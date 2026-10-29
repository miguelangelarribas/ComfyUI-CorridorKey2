"""In-memory CorridorKey 2.0 inference for ComfyUI tensors.

Mirrors ``corridorkey.runtime.run_inference`` (RVM prepass -> Hann-windowed
stage 1 -> streamed stage 2) but reads frames from a ComfyUI IMAGE batch
[T, H, W, C] and returns tensors instead of writing video/EXR files.

Only the frame source, hint source and output sink differ from upstream; the
math (color transforms, padding, Hann carry, hint-quality conditioning,
rendering) is kept 1:1 with the reference runtime.
"""
from __future__ import annotations

import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from .corridorkey import color_utils as cu
from .corridorkey.runtime import (
    InferenceConfig,
    _build_foundation_inputs,
    configure_low_vram_inference,
    select_vram_profile,
)
from .corridorkey.video_io import ceil_to_multiple, taper_weights, window_starts

SHAPE_QUANTUM = 128  # InferenceConfig.shape_quantum


@dataclass
class KeyerSettings:
    input_linear: bool = False        # IMAGE holds scene-linear RGB (e.g. EXR plates)
    output_linear: bool = False       # foreground returned as unclamped scene-linear
    hint_quality: float = 0.95
    carry_hint: bool = True
    carry_hint_quality: float = 0.95
    memory_mode: str = "auto"         # auto | normal | low_vram
    hann_chunk: int = 24
    hann_stride: int = 0              # 0 -> hann_chunk // 2
    stage2_batch: int = 1
    despill_strength: float = 0.5
    temp_dir: str | None = None


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def build_meta(T: int, H: int, W: int) -> dict:
    h_pad = ceil_to_multiple(H, SHAPE_QUANTUM)
    w_pad = ceil_to_multiple(W, SHAPE_QUANTUM)
    pad_l = (w_pad - W) // 2
    pad_r = w_pad - W - pad_l
    pad_t = (h_pad - H) // 2
    pad_b = h_pad - H - pad_t
    return {
        "num_frames": int(T),
        "native_hw": (H, W),
        "output_hw": (H, W),
        "model_native_hw": (H, W),
        "shape_hw": (h_pad, w_pad),
        "pad": (pad_l, pad_r, pad_t, pad_b),
    }


def _pad_nchw(x: torch.Tensor, pad) -> torch.Tensor:
    """Center pad, equivalent to cv2.BORDER_REFLECT_101 used upstream."""
    l, r, t, b = pad
    if not (l or r or t or b):
        return x
    H, W = x.shape[-2:]
    mode = "reflect" if (max(l, r) < W and max(t, b) < H) else "replicate"
    return F.pad(x, (l, r, t, b), mode=mode)


def _crop_native(x: torch.Tensor, meta: dict) -> torch.Tensor:
    h, w = meta["native_hw"]
    pad_l, _, pad_t, _ = meta["pad"]
    return x[..., pad_t:pad_t + h, pad_l:pad_l + w]


def _despill(fg: torch.Tensor, strength: float) -> torch.Tensor:
    # Same as runtime._despill_average_srgb_torch
    if strength <= 0.0:
        return fg
    r, g, b = fg[:, 0:1], fg[:, 1:2], fg[:, 2:3]
    spill = (g - (r + b) * 0.5).clamp(min=0.0)
    full = torch.cat([r + spill * 0.5, g - spill, b + spill * 0.5], dim=1)
    if strength >= 1.0:
        return full
    return fg * (1.0 - strength) + full * strength


# ---------------------------------------------------------------------------
# Frame / hint sources
# ---------------------------------------------------------------------------

class _FrameSource:
    """Builds the model (asinh) and display (SDR proxy) views for a frame range.

    Matches video_io.preprocess_frame_dir_path / preprocess_video_frame_views
    at model_scale=1.0.
    """

    def __init__(self, images: torch.Tensor, meta: dict, input_linear: bool,
                 device: torch.device):
        self.images = images
        self.meta = meta
        self.input_linear = bool(input_linear)
        self.device = device

    @torch.no_grad()
    def get(self, t0: int, t1: int, need_model: bool = True):
        x = self.images[t0:t1, :, :, :3].to(self.device, non_blocking=True)
        x = x.float().permute(0, 3, 1, 2).contiguous()
        if self.input_linear:
            linear = x
            display = cu.linear_to_srgb(linear.clamp(min=0.0)).clamp(0.0, 1.0)
        else:
            display = x.clamp(0.0, 1.0)
            linear = cu.srgb_to_linear(display)
        pad = self.meta["pad"]
        display_t = _pad_nchw(display, pad).to(torch.bfloat16)
        if not need_model:
            return None, display_t
        model_t = cu.linear_to_asinh(_pad_nchw(linear, pad)).to(torch.bfloat16)
        return model_t, display_t


def prep_hint(masks: torch.Tensor, meta: dict) -> torch.Tensor:
    """[N,h,w] matte (1 = subject) -> [N,1,Hpad,Wpad] bf16 on CPU."""
    m = masks.float()
    if m.dim() == 2:
        m = m[None]
    m = m[:, None]
    if tuple(m.shape[-2:]) != tuple(meta["native_hw"]):
        m = F.interpolate(m, size=meta["native_hw"], mode="bilinear",
                          align_corners=False)
    m = _pad_nchw(m.clamp(0.0, 1.0), meta["pad"])
    return m.to(device="cpu", dtype=torch.bfloat16)


# ---------------------------------------------------------------------------
# Prepasses
# ---------------------------------------------------------------------------

def _autocast(device: torch.device):
    return torch.autocast(device.type, dtype=torch.bfloat16)


@torch.no_grad()
def _precompute_rvm(model, frames: _FrameSource, meta: dict, out_path: Path,
                    *, chunk: int, tick: Callable[[], None]):
    rvm = model.foundations.rvm
    device = next(rvm.parameters()).device
    T_all = int(meta["num_frames"])
    H, W = meta["shape_hw"]
    H_e, W_e = H // 4, W // 4
    D = rvm.embed_dim
    out = np.memmap(out_path, dtype=np.float16, mode="w+",
                    shape=(T_all, D, H_e // 16, W_e // 16))
    mean = model.foundations.rvm_mean.view(1, 1, -1, 1, 1).float().cpu()
    std = model.foundations.rvm_std.view(1, 1, -1, 1, 1).float().cpu().clamp(min=1e-6)
    state = None
    for t0 in range(0, T_all, chunk):
        t1 = min(T_all, t0 + chunk)
        Tc = t1 - t0
        _, display = frames.get(t0, t1, need_model=False)
        rgb = display.to(device).float()
        rgb_rvm = F.interpolate(rgb, size=(H_e, W_e), mode="bilinear",
                                align_corners=False)
        rgb_rvm = rgb_rvm.reshape(1, Tc, 3, H_e, W_e).clamp(0.0, 1.0)
        with _autocast(device):
            feats, state = rvm(rgb_rvm, state=state)
        out[t0:t1] = ((feats.float().cpu() - mean) / std)[0].numpy().astype(np.float16)
        del rgb, rgb_rvm, feats, display
        tick()
    out.flush()
    del out


@torch.no_grad()
def _precompute_foundations(model, frames: _FrameSource, meta: dict, prefix: Path,
                            *, chunk: int, tick: Callable[[], None]) -> dict:
    device = next(model.foundations.parameters()).device
    T_all = int(meta["num_frames"])
    H, W = meta["shape_hw"]
    h64, w64 = H // 64, W // 64
    C_cradio = int(getattr(model.stage1, "cradio_dim", 0) or 0)
    C_moge = int(getattr(model.stage1, "moge_dim", 0) or 0)
    if C_moge <= 0:
        raise RuntimeError("MoGe is required for this model")
    paths = {"moge": str(prefix) + "_moge.f32", "moge_shape": (T_all, C_moge, h64, w64)}
    moge_out = np.memmap(paths["moge"], dtype=np.float32, mode="w+",
                         shape=paths["moge_shape"])
    cradio_out = None
    if C_cradio > 0:
        paths["cradio"] = str(prefix) + "_cradio.f32"
        paths["cradio_shape"] = (T_all, C_cradio, h64, w64)
        cradio_out = np.memmap(paths["cradio"], dtype=np.float32, mode="w+",
                               shape=paths["cradio_shape"])
    for t0 in range(0, T_all, chunk):
        t1 = min(T_all, t0 + chunk)
        Tc = t1 - t0
        _, display = frames.get(t0, t1, need_model=False)
        rgb_radio, rgb_moge = _build_foundation_inputs(display.to(device).unsqueeze(0))
        with _autocast(device):
            ff = model.foundations(
                rgb_cradio=rgb_radio.reshape(Tc, 3, *rgb_radio.shape[-2:]),
                rgb_moge=rgb_moge.reshape(Tc, 3, *rgb_moge.shape[-2:]),
            )
        moge_out[t0:t1] = ff["moge"].float().cpu().numpy()
        if cradio_out is not None:
            cradio_out[t0:t1] = ff["cradio"].float().cpu().numpy()
        del display, rgb_radio, rgb_moge, ff
        tick()
    moge_out.flush()
    del moge_out
    if cradio_out is not None:
        cradio_out.flush()
        del cradio_out
    return paths


# ---------------------------------------------------------------------------
# Stage 1 (Hann windows) and stage 2 (render)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _hann_stage1(model, frames: _FrameSource, meta: dict, *, chunk: int, stride: int,
                 first_hint, full_hints, hint_quality: float,
                 carry_hint: bool, carry_hint_quality: float,
                 rvm_path, foundation_paths, baton_path,
                 on_ready, tick: Callable[[], None]):
    """Port of runtime.hann_stage1_stream with in-memory frame/hint sources."""
    device = next(model.stage1.parameters()).device
    T_all = int(meta["num_frames"])
    H, W = meta["shape_hw"]
    d_16 = model.stage1.d_16
    h16, w16 = H // 16, W // 16
    starts = window_starts(T_all, chunk, stride)
    overlap = max(0, chunk - stride)
    baton_store = None
    if on_ready is None:
        baton_store = np.memmap(baton_path, dtype=np.float16, mode="w+",
                                shape=(T_all, d_16, h16, w16))
    live_acc, live_w, carry_cache = {}, {}, {}
    print(f"[CorridorKey2] stage1: chunk={chunk}, stride={stride}, "
          f"windows={len(starts)}, baton_grid={w16}x{h16}")

    rvm_map = None
    if rvm_path is not None:
        D = model.foundations.rvm.embed_dim
        rvm_map = np.memmap(rvm_path, dtype=np.float16, mode="r",
                            shape=(T_all, D, H // 64, W // 64))
    cradio_map = moge_map = None
    if foundation_paths is not None:
        moge_map = np.memmap(foundation_paths["moge"], dtype=np.float32, mode="r",
                             shape=tuple(foundation_paths["moge_shape"]))
        if model.stage1.cradio_dim > 0:
            cradio_map = np.memmap(foundation_paths["cradio"], dtype=np.float32,
                                   mode="r", shape=tuple(foundation_paths["cradio_shape"]))

    def flush_before(t_limit: int):
        ready = []
        for k in sorted(k for k in live_acc if k < t_limit):
            frame = (live_acc.pop(k) / max(live_w.pop(k), 1e-6)).astype(np.float16)
            if baton_store is not None:
                baton_store[k] = frame
            else:
                ready.append((k, frame))
        if ready:
            on_ready(ready)

    for i, t0 in enumerate(starts):
        t1 = min(t0 + chunk, T_all)
        Tc = t1 - t0
        hint_cpu = torch.zeros(1, Tc, 1, H, W, dtype=torch.bfloat16)
        hint_quality_cpu = torch.zeros(1, Tc, dtype=torch.bfloat16)
        hint_present = torch.zeros(1, Tc, dtype=torch.bool)
        if full_hints is not None:
            hint_cpu[0] = prep_hint(full_hints[t0:t1], meta)
            hint_quality_cpu.fill_(float(hint_quality))
            hint_present.fill_(True)
        for j, t_abs in enumerate(range(t0, t1)):
            if t_abs in carry_cache:
                hint_cpu[0, j] = carry_cache[t_abs]
                hint_quality_cpu[0, j] = float(carry_hint_quality)
                hint_present[0, j] = True
        if first_hint is not None and t0 == 0:
            hint_cpu[0, 0] = first_hint
            hint_quality_cpu[0, 0] = float(hint_quality)
            hint_present[0, 0] = True
        if not bool(hint_present.any().item()):
            hint_quality_cpu.fill_(-1.0)  # "no hint": the model picks the subject

        model_chunk, display_chunk = frames.get(t0, t1)
        rgb_c = model_chunk.to(device).unsqueeze(0)
        hint_c = hint_cpu.to(device, non_blocking=True)
        hint_quality_c = hint_quality_cpu.to(device, non_blocking=True)
        if foundation_paths is None:
            rgb_radio, rgb_moge = _build_foundation_inputs(display_chunk.to(device).unsqueeze(0))
            with _autocast(device):
                ff = model.foundations(
                    rgb_cradio=rgb_radio.reshape(Tc, 3, *rgb_radio.shape[-2:]),
                    rgb_moge=rgb_moge.reshape(Tc, 3, *rgb_moge.shape[-2:]),
                )
            cradio_feat = ff["cradio"].reshape(1, Tc, -1, H // 64, W // 64)
            moge_feat = ff["moge"].reshape(1, Tc, -1, H // 64, W // 64)
            del ff, rgb_radio, rgb_moge
        else:
            moge_feat = torch.from_numpy(np.asarray(moge_map[t0:t1]).copy())
            moge_feat = moge_feat.unsqueeze(0).to(device).float()
            cradio_feat = torch.from_numpy(np.asarray(cradio_map[t0:t1]).copy())
            cradio_feat = cradio_feat.unsqueeze(0).to(device).float()
        rvm_feat = None
        if rvm_map is not None:
            rvm_feat = torch.from_numpy(np.asarray(rvm_map[t0:t1]).copy())
            rvm_feat = rvm_feat.unsqueeze(0).to(device).float()
        with _autocast(device):
            s1 = model.stage1(rgb_c, hint_c, moge_feat, cradio_feat=cradio_feat,
                              rvm_feat=rvm_feat, t_stride=1.0,
                              hint_quality=hint_quality_c)
        baton_gpu = s1["baton"][0].float()
        next_start = starts[i + 1] if i + 1 < len(starts) else T_all
        if carry_hint:
            aux_alpha = s1["aux"][0, :, 0:1].float()
            aux_full = F.interpolate(aux_alpha, size=(H, W), mode="bilinear",
                                     align_corners=False).clamp(0.0, 1.0)
            aux_cpu = aux_full.to("cpu", dtype=torch.bfloat16)
            for j, t_abs in enumerate(range(t0, t1)):
                carry_cache[t_abs] = aux_cpu[j].clone()
            for old_t in [k for k in carry_cache if k < next_start]:
                del carry_cache[old_t]
            del aux_alpha, aux_full, aux_cpu
        w = taper_weights(Tc, overlap, is_first=(i == 0),
                          is_last=(i == len(starts) - 1),
                          device="cpu", dtype=torch.float32)
        weighted = (baton_gpu * w.to(baton_gpu.device).view(Tc, 1, 1, 1)).cpu().numpy()
        w_np = w.numpy()
        for j, t_abs in enumerate(range(t0, t1)):
            if t_abs not in live_acc:
                live_acc[t_abs] = weighted[j].copy()
                live_w[t_abs] = float(w_np[j])
            else:
                live_acc[t_abs] += weighted[j]
                live_w[t_abs] += float(w_np[j])
        del s1, baton_gpu, weighted, cradio_feat, moge_feat, rvm_feat
        del rgb_c, hint_c, hint_quality_c, hint_cpu, hint_quality_cpu
        del model_chunk, display_chunk
        flush_before(next_start)
        tick()
    flush_before(T_all + 1)
    if baton_store is not None:
        baton_store.flush()
    del baton_store, rvm_map, cradio_map, moge_map


@torch.no_grad()
def _stage2_render(model, frames: _FrameSource, meta: dict, items, s: KeyerSettings,
                   out_alpha: torch.Tensor, out_fg: torch.Tensor):
    device = next(model.stage2.parameters()).device
    bs = max(1, int(s.stage2_batch))
    for offset in range(0, len(items), bs):
        sub = items[offset:offset + bs]
        ts = [t for t, _ in sub]
        rgb_t = torch.cat([frames.get(t, t + 1)[0] for t in ts]).to(device).unsqueeze(0)
        baton_t = torch.from_numpy(np.stack([b for _, b in sub])).unsqueeze(0)
        baton_t = baton_t.to(device).float()
        with torch.inference_mode(), _autocast(device):
            logits = model.stage2(rgb_t, baton_t)
            if isinstance(logits, tuple):
                logits = logits[0]
        logits = logits[0]
        alpha = _crop_native(logits[:, 0:1].float(), meta).clamp(0.0, 1.0)
        fg_lin = cu.asinh_to_linear(_crop_native(logits[:, 1:4].float(), meta))
        if s.output_linear:
            # EXR path upstream: unclamped scene-linear, despill in linear light.
            fg = _despill(fg_lin, s.despill_strength)
        else:
            # Video path upstream: sRGB, despill in sRGB, clamped.
            fg = cu.linear_to_srgb(fg_lin.clamp(min=0.0)).clamp(0.0, 1.0)
            fg = _despill(fg, s.despill_strength).clamp(0.0, 1.0)
        idx = torch.tensor(ts, dtype=torch.long)
        out_alpha[idx] = alpha[:, 0].cpu()
        out_fg[idx] = fg.permute(0, 2, 3, 1).cpu()
        del logits, rgb_t, baton_t, alpha, fg_lin, fg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _set_low_vram(model, enabled: bool):
    cfg = InferenceConfig()
    if not enabled:
        cfg.low_vram_enc_frame_chunk = 0
        cfg.low_vram_mlp_chunk_tokens = 0
        cfg.low_vram_swin_window_batch = 0
        cfg.low_vram_full_attn_query_chunk_tokens = 0
    configure_low_vram_inference(model, cfg)


def _safe_remove(p):
    try:
        os.remove(p)
    except OSError:
        pass


def resolve_memory_profile(s: KeyerSettings, meta: dict, device: torch.device):
    T_all = int(meta["num_frames"])
    if s.memory_mode == "auto":
        if device.type != "cuda":
            return 8, 4, False
        free_bytes, _ = torch.cuda.mem_get_info(device)
        budget_gb = (free_bytes + torch.cuda.memory_reserved(device)) / 2**30
        chunk, stride, low_vram, est = select_vram_profile(meta["shape_hw"], T_all, budget_gb)
        h, w = meta["shape_hw"]
        print(f"[CorridorKey2] auto VRAM: {budget_gb:.1f} GB at {w}x{h} -> "
              f"hann_chunk={chunk}, hann_stride={stride}, low_vram={low_vram} "
              f"(estimated peak {est:.1f} GB)")
        return chunk, stride, low_vram
    chunk = max(1, int(s.hann_chunk))
    stride = int(s.hann_stride) if int(s.hann_stride) > 0 else max(1, chunk // 2)
    stride = min(max(1, stride), chunk)  # stride > chunk would skip frames
    return chunk, stride, s.memory_mode == "low_vram"


def run_keyer(model, images: torch.Tensor, s: KeyerSettings, *,
              hint_masks: torch.Tensor | None = None,
              device: torch.device | str = "cuda",
              progress: Callable[[int, int], None] | None = None,
              check_interrupt: Callable[[], None] | None = None):
    """images: [T,H,W,C] float. hint_masks: None, [1,H,W] (first frame) or [T,H,W].

    Returns (alpha [T,H,W] float32, foreground [T,H,W,3] float32), both on CPU.
    """
    device = torch.device(device)
    if images.dim() != 4 or images.shape[-1] < 3:
        raise ValueError(f"expected IMAGE [T,H,W,3], got {tuple(images.shape)}")
    T_all, H, W = (int(v) for v in images.shape[:3])
    meta = build_meta(T_all, H, W)

    first_hint = full_hints = None
    if hint_masks is not None:
        hm = hint_masks if hint_masks.dim() == 3 else hint_masks[None]
        if hm.shape[0] == 1:
            first_hint = prep_hint(hm[0:1], meta)[0]
        elif hm.shape[0] == T_all:
            full_hints = hm
        else:
            raise ValueError(
                f"hint_mask has {hm.shape[0]} frames; expected 1 (first-frame hint) "
                f"or {T_all} (one per frame)")

    chunk, stride, low_vram = resolve_memory_profile(s, meta, device)
    n_windows = len(window_starts(T_all, chunk, stride))
    rvm_chunk = max(1, min(chunk, 8))
    found_chunk = InferenceConfig().low_vram_foundation_chunk
    has_rvm = model.foundations.rvm is not None
    total = n_windows
    total += -(-T_all // rvm_chunk) if has_rvm else 0
    total += (-(-T_all // found_chunk) + T_all) if low_vram else 0
    done = [0]

    def tick(n: int = 1):
        if check_interrupt is not None:
            check_interrupt()
        done[0] += n
        if progress is not None:
            progress(done[0], total)

    tmp = Path(s.temp_dir or tempfile.gettempdir()) / "corridorkey2"
    tmp.mkdir(parents=True, exist_ok=True)
    tag = uuid.uuid4().hex[:10]
    rvm_path = tmp / f"{tag}_rvm_feat.f16"
    baton_path = tmp / f"{tag}_baton.f16"
    found_prefix = tmp / f"{tag}_foundation"

    out_alpha = torch.zeros(T_all, H, W, dtype=torch.float32)
    out_fg = torch.zeros(T_all, H, W, 3, dtype=torch.float32)
    frames = _FrameSource(images, meta, s.input_linear, device)
    h_pad, w_pad = meta["shape_hw"]
    print(f"[CorridorKey2] {T_all} frames {W}x{H} (padded {w_pad}x{h_pad}), "
          f"hint={'full' if full_hints is not None else 'first' if first_hint is not None else 'none'}, "
          f"input={'linear' if s.input_linear else 'sRGB'}, "
          f"output={'linear' if s.output_linear else 'sRGB'}")

    try:
        # Phase 1 placement (runtime._phase1_setup)
        model.foundations.to(device)
        model.stage1.to("cpu" if low_vram else device)
        model.stage2.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _set_low_vram(model, low_vram)

        if has_rvm:
            _precompute_rvm(model, frames, meta, rvm_path, chunk=rvm_chunk, tick=tick)
        foundation_paths = None
        if low_vram:
            foundation_paths = _precompute_foundations(
                model, frames, meta, found_prefix, chunk=found_chunk, tick=tick)
            model.foundations.to("cpu")
            model.stage1.to(device)
        else:
            model.stage2.to(device)
        model.eval()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        def on_ready(items):
            _stage2_render(model, frames, meta, items, s, out_alpha, out_fg)

        _hann_stage1(
            model, frames, meta, chunk=chunk, stride=stride,
            first_hint=first_hint, full_hints=full_hints,
            hint_quality=s.hint_quality, carry_hint=s.carry_hint,
            carry_hint_quality=s.carry_hint_quality,
            rvm_path=rvm_path if has_rvm else None,
            foundation_paths=foundation_paths, baton_path=baton_path,
            on_ready=None if low_vram else on_ready, tick=tick)

        if low_vram:
            # Phase 2 (runtime._phase2_setup): stage 2 alone on the GPU.
            model.foundations.to("cpu")
            model.stage1.to("cpu")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            model.stage2.to(device)
            baton = np.memmap(baton_path, dtype=np.float16, mode="r",
                              shape=(T_all, model.stage1.d_16, h_pad // 16, w_pad // 16))
            bs = max(1, int(s.stage2_batch))
            for t0 in range(0, T_all, bs):
                t1 = min(T_all, t0 + bs)
                items = [(t, np.asarray(baton[t]).copy()) for t in range(t0, t1)]
                _stage2_render(model, frames, meta, items, s, out_alpha, out_fg)
                tick(t1 - t0)
            del baton
    finally:
        for p in (rvm_path, baton_path, Path(str(found_prefix) + "_cradio.f32"),
                  Path(str(found_prefix) + "_moge.f32")):
            _safe_remove(p)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out_alpha, out_fg

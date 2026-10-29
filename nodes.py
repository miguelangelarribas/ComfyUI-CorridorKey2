from __future__ import annotations

import os
from pathlib import Path

import torch

import comfy.model_management as mm
import comfy.utils
import folder_paths

from .ck_inference import KeyerSettings, run_keyer
from .ck_loader import download_assets, load_corridorkey, missing_assets
from .corridorkey import color_utils as cu

MODEL_FOLDER = "corridorkey2"
_default_dir = os.path.join(folder_paths.models_dir, MODEL_FOLDER)
if MODEL_FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths[MODEL_FOLDER] = ([_default_dir], set())
else:
    folder_paths.add_model_folder_path(MODEL_FOLDER, _default_dir)


def _model_root() -> Path:
    roots = [Path(p) for p in folder_paths.get_folder_paths(MODEL_FOLDER)]
    for r in roots:
        if (r / "corridorkey_v2.safetensors").is_file():
            return r
    return roots[0] if roots else Path(_default_dir)


class CorridorKey2ModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "auto_download": ("BOOLEAN", {"default": True, "tooltip":
                "Download missing weights (~5 GB) into models/corridorkey2 on first run."}),
        }}

    RETURN_TYPES = ("CORRIDORKEY2_MODEL",)
    RETURN_NAMES = ("ck2_model",)
    FUNCTION = "load"
    CATEGORY = "CorridorKey2"

    def load(self, auto_download):
        root = _model_root()
        missing = missing_assets(root)
        if missing:
            if not auto_download:
                raise FileNotFoundError(
                    f"CorridorKey2: missing {missing} under {root}. "
                    "Enable auto_download or place the files manually (see README).")
            download_assets(root)
        print(f"[CorridorKey2] loading from {root}")
        return (load_corridorkey(root),)


class CorridorKey2Keyer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ck2_model": ("CORRIDORKEY2_MODEL",),
                "images": ("IMAGE",),
                "input_colorspace": (["sRGB", "linear"], {"tooltip":
                    "sRGB: display-coded plates (video, PNG/JPG). linear: scene-linear "
                    "RGB (e.g. linear EXR); HDR values above 1.0 are preserved."}),
                "output_colorspace": (["sRGB", "linear"], {"tooltip":
                    "sRGB: clamped [0,1] foreground for preview/video. linear: unclamped "
                    "scene-linear foreground for EXR/comp."}),
                "despill_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "0 = raw model foreground, 1 = full green despill."}),
                "memory_mode": (["auto", "normal", "low_vram"], {"tooltip":
                    "auto picks Hann window + low-VRAM from free GPU memory. "
                    "normal/low_vram use hann_chunk / hann_stride below."}),
                "hann_chunk": ("INT", {"default": 24, "min": 1, "max": 160, "tooltip":
                    "Frames per stage-1 temporal window (ignored in auto)."}),
                "hann_stride": ("INT", {"default": 0, "min": 0, "max": 160, "tooltip":
                    "Frames between windows; 0 = hann_chunk/2 (ignored in auto)."}),
                "hint_quality": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Trust in the hint mask: 0.95 clean matte, 0.75 rough, <=0.5 loose."}),
                "carry_hint": ("BOOLEAN", {"default": True, "tooltip":
                    "Feed the model's own matte from overlapping windows back as a hint."}),
                "carry_hint_quality": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "stage2_batch": ("INT", {"default": 1, "min": 1, "max": 16}),
                "offload_after": ("BOOLEAN", {"default": True, "tooltip":
                    "Move the model back to RAM after keying to free VRAM."}),
            },
            "optional": {
                "hint_mask": ("MASK", {"tooltip":
                    "1 frame = first-frame hint; N frames (= images) = full hint per frame. "
                    "White = subject."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("foreground", "alpha")
    FUNCTION = "key"
    CATEGORY = "CorridorKey2"
    DESCRIPTION = ("CorridorKey 2.0 green-screen keyer. Returns a straight (un-premultiplied) "
                   "foreground and its alpha matte (1 = foreground).")

    def key(self, ck2_model, images, input_colorspace, output_colorspace, despill_strength,
            memory_mode, hann_chunk, hann_stride, hint_quality, carry_hint,
            carry_hint_quality, stage2_batch, offload_after, hint_mask=None):
        device = mm.get_torch_device()
        if device.type != "cuda":
            raise RuntimeError("CorridorKey2 requires a CUDA GPU.")
        # The model lives outside ComfyUI's model manager: free VRAM first.
        mm.unload_all_models()
        mm.soft_empty_cache()

        settings = KeyerSettings(
            input_linear=input_colorspace == "linear",
            output_linear=output_colorspace == "linear",
            hint_quality=hint_quality, carry_hint=carry_hint,
            carry_hint_quality=carry_hint_quality, memory_mode=memory_mode,
            hann_chunk=hann_chunk, hann_stride=hann_stride,
            stage2_batch=stage2_batch, despill_strength=despill_strength,
            temp_dir=folder_paths.get_temp_directory(),
        )
        pbar = comfy.utils.ProgressBar(100)

        def progress(done, total):
            pbar.update_absolute(min(done, total), total)

        try:
            alpha, fg = run_keyer(
                ck2_model, images, settings, hint_masks=hint_mask, device=device,
                progress=progress,
                check_interrupt=mm.throw_exception_if_processing_interrupted)
        finally:
            if offload_after:
                ck2_model.to("cpu")
                mm.soft_empty_cache()
        return (fg, alpha)


class CorridorKey2CheckerPreview:
    """Composite foreground over a checkerboard in linear light (as upstream)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "foreground": ("IMAGE",),
            "alpha": ("MASK",),
            "foreground_colorspace": (["sRGB", "linear"],),
            "checker_size": ("INT", {"default": 64, "min": 4, "max": 512}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview",)
    FUNCTION = "preview"
    CATEGORY = "CorridorKey2"

    def preview(self, foreground, alpha, foreground_colorspace, checker_size):
        T, H, W, _ = foreground.shape
        y = torch.arange(H).view(H, 1) // checker_size
        x = torch.arange(W).view(1, W) // checker_size
        board = torch.where((y + x) % 2 == 0, 0.75, 0.5).float()
        bg_lin = cu.srgb_to_linear(board)[None, :, :, None]
        a = alpha.float()
        if a.dim() == 2:
            a = a[None]
        if a.shape[0] == 1 and T > 1:
            a = a.expand(T, -1, -1)
        a = a[..., None].clamp(0.0, 1.0)
        fg = foreground[..., :3].float()
        fg_lin = fg if foreground_colorspace == "linear" else cu.srgb_to_linear(fg.clamp(0.0, 1.0))
        comp = fg_lin * a + bg_lin * (1.0 - a)
        return (cu.linear_to_srgb(comp.clamp(min=0.0)).clamp(0.0, 1.0),)


NODE_CLASS_MAPPINGS = {
    "CorridorKey2ModelLoader": CorridorKey2ModelLoader,
    "CorridorKey2Keyer": CorridorKey2Keyer,
    "CorridorKey2CheckerPreview": CorridorKey2CheckerPreview,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CorridorKey2ModelLoader": "CorridorKey 2.0 Model Loader",
    "CorridorKey2Keyer": "CorridorKey 2.0 Keyer",
    "CorridorKey2CheckerPreview": "CorridorKey 2.0 Checker Preview",
}

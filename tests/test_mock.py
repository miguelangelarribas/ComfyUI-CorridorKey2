"""Mock-model test of the in-memory CorridorKey2 pipeline (CPU, no weights)."""
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG.parent))
pkg = types.ModuleType("ck2pkg")
pkg.__path__ = [str(PKG)]
sys.modules["ck2pkg"] = pkg

from ck2pkg.ck_inference import KeyerSettings, run_keyer, build_meta  # noqa: E402
from ck2pkg.corridorkey import color_utils as cu  # noqa: E402

D16, DR, CM, CC = 8, 6, 5, 7


class RVM(nn.Module):
    embed_dim = DR

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))

    def forward(self, x, state=None):
        B, T, _, H, W = x.shape
        f = F.avg_pool2d(x.reshape(B * T, 3, H, W), 16).mean(1, keepdim=True)
        f = f.expand(-1, DR, -1, -1).reshape(B, T, DR, H // 16, W // 16)
        n = (state if state is not None else 0) + T
        return f, n


class Found(nn.Module):
    def __init__(self):
        super().__init__()
        self.rvm = RVM()
        self.register_buffer("rvm_mean", torch.zeros(DR))
        self.register_buffer("rvm_std", torch.ones(DR))

    def forward(self, rgb_cradio=None, rgb_moge=None):
        N, _, h, w = rgb_cradio.shape
        g = F.adaptive_avg_pool2d(rgb_cradio.float(), (h // 16, w // 16))
        return {"cradio": g[:, 1:2].expand(-1, CC, -1, -1),
                "moge": g[:, 0:1].expand(-1, CM, -1, -1)}


class Stage1(nn.Module):
    d_16, cradio_dim, moge_dim = D16, CC, CM

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))
        self.calls = []

    def forward(self, rgb, hint, moge_feat, cradio_feat=None, rvm_feat=None,
                t_stride=1.0, hint_quality=None):
        B, T, _, H, W = rgb.shape
        assert moge_feat.shape[-2:] == (H // 64, W // 64)
        assert rvm_feat.shape[-2:] == (H // 64, W // 64)
        self.calls.append(hint_quality[0].float().tolist())
        # baton ch0 = per-frame /16 green of the decoded linear plate (deterministic per frame)
        lin = cu.asinh_to_linear(rgb.float())
        g = F.avg_pool2d(lin[0, :, 1:2], 16)  # [T,1,H/16,W/16]
        baton = g.expand(-1, D16, -1, -1).unsqueeze(0)
        aux = F.interpolate(lin[0, :, 0:4 - 1].mean(1, keepdim=True), scale_factor=0.5)
        aux = aux.expand(-1, 4, -1, -1).unsqueeze(0)
        return {"baton": baton, "aux": aux}


class Stage2(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))

    def forward(self, rgb, baton):
        B, T, _, H, W = rgb.shape
        a = F.interpolate(baton[0, :, 0:1].float(), size=(H, W), mode="nearest")
        return torch.cat([a, rgb[0].float()], 1).unsqueeze(0)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.foundations, self.stage1, self.stage2 = Found(), Stage1(), Stage2()


def run(model, imgs, **kw):
    hint = kw.pop("hint", None)
    s = KeyerSettings(**kw)
    model.stage1.calls.clear()
    return run_keyer(model, imgs, s, hint_masks=hint, device="cpu")


def main():
    torch.manual_seed(0)
    T, H, W = 11, 150, 200          # non-multiple of 128 -> exercises padding/crop
    imgs = torch.rand(T, H, W, 3)
    m = Model()

    # 1) normal, explicit window: fg round-trips the plate, alpha matches per-frame baton
    a1, fg1 = run(m, imgs, memory_mode="normal", hann_chunk=4, hann_stride=2, despill_strength=0.0)
    assert a1.shape == (T, H, W) and fg1.shape == (T, H, W, 3)
    err = (fg1 - imgs).abs().max().item()
    print("sRGB roundtrip max err", err)
    assert err < 0.02, err
    ref = F.avg_pool2d(cu.srgb_to_linear(imgs.permute(0, 3, 1, 2))[:, 1:2], 1)
    # alpha is nearest-upsampled /16 average of padded plate; check frame independence instead
    a_single, _ = run(m, imgs, memory_mode="normal", hann_chunk=1, hann_stride=1, despill_strength=0.0)
    d = (a1 - a_single).abs().max().item()
    print("hann blend vs single-frame alpha diff", d)
    assert d < 1e-2, d
    print("windows (hint quality per call):", m.stage1.calls[:3])

    # 2) low_vram matches normal
    a2, fg2 = run(m, imgs, memory_mode="low_vram", hann_chunk=4, hann_stride=2, despill_strength=0.0)
    assert torch.allclose(a1, a2, atol=1e-3) and torch.allclose(fg1, fg2, atol=1e-3)
    print("low_vram == normal: OK")

    # 3) hint semantics
    run(m, imgs, memory_mode="normal", hann_chunk=4, hann_stride=2, carry_hint=False)
    assert all(q == -1.0 for c in m.stage1.calls for q in c), "no hint -> -1"
    run(m, imgs, memory_mode="normal", hann_chunk=4, hann_stride=2, carry_hint=False,
        hint=torch.ones(1, H, W))
    c = m.stage1.calls
    assert abs(c[0][0] - 0.95) < 1e-2 and c[0][1] == 0.0 and all(q == -1.0 for q in c[1]), c[:2]
    run(m, imgs, memory_mode="normal", hann_chunk=4, hann_stride=2, carry_hint=True,
        hint=torch.ones(1, H // 2, W // 2))
    c = m.stage1.calls
    assert abs(c[1][0] - 0.95) < 1e-2 and abs(c[1][1] - 0.95) < 1e-2 and c[1][2] == 0.0, c[:2]
    run(m, imgs, memory_mode="normal", hann_chunk=4, hann_stride=2, carry_hint=False,
        hint=torch.ones(T, H, W), hint_quality=0.75)
    assert all(abs(q - 0.75) < 1e-2 for cc in m.stage1.calls for q in cc)
    try:
        run(m, imgs, hint=torch.ones(3, H, W), memory_mode="normal")
        raise AssertionError("bad hint length accepted")
    except ValueError:
        pass
    print("hint semantics: OK")

    # 4) linear in/out with HDR values
    lin = torch.rand(T, H, W, 3) * 4.0
    _, fgl = run(m, lin, memory_mode="normal", hann_chunk=4, input_linear=True,
                 output_linear=True, despill_strength=0.0)
    rel = ((fgl - lin).abs() / (lin.abs() + 1e-2)).max().item()
    print("linear HDR roundtrip max rel err", rel)
    assert rel < 0.03 and fgl.max() > 3.5

    # 5) stride > chunk is clamped, T < chunk, single frame
    a5, _ = run(m, imgs, memory_mode="normal", hann_chunk=3, hann_stride=9)
    assert (a5 != 0).any(dim=(1, 2)).all(), "every frame rendered"
    run(m, imgs[:2], memory_mode="normal", hann_chunk=24)
    run(m, imgs[:1], memory_mode="auto")
    print("edge cases: OK")
    _ = ref, build_meta
    print("ALL OK")


if __name__ == "__main__":
    main()

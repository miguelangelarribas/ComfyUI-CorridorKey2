"""Compare ck_inference.run_keyer against upstream runtime.run_inference (EXR path)."""
import os
import sys
import tempfile
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import test_mock as tm  # registers ck2pkg + mock model  # noqa: E402
from ck2pkg.ck_inference import KeyerSettings, run_keyer  # noqa: E402
from ck2pkg.corridorkey import runtime as rt  # noqa: E402
from ck2pkg.corridorkey.video_io import WriterConfig  # noqa: E402

# Make the upstream CUDA-only runtime run on CPU.
torch.nn.Module.cuda = lambda self, *a, **k: self
torch.cuda.empty_cache = lambda: None
torch.cuda.synchronize = lambda *a, **k: None
_autocast = torch.autocast
torch.autocast = lambda device_type, **k: _autocast("cpu", **k)


def main():
    rng = np.random.default_rng(1)
    T, H, W = 9, 140, 210
    frames = (rng.random((T, H, W, 3)) * 255).astype(np.uint8)
    hint = np.zeros((H, W), np.uint8)
    hint[30:110, 50:160] = 255
    tmp = Path(tempfile.mkdtemp())
    (tmp / "plate").mkdir()
    for i, f in enumerate(frames):
        cv2.imwrite(str(tmp / "plate" / f"f_{i:04d}.png"), f[..., ::-1])
    cv2.imwrite(str(tmp / "hint.png"), hint)

    for carry, low in ((True, False), (False, True)):
        model = tm.Model()
        cfg = rt.InferenceConfig(hann_chunk=4, hann_stride=2, carry_hint=carry,
                                 low_vram=low, outputs={"alpha", "fg"},
                                 output_format="exr", exr_dtype="float",
                                 despill_strength=0.0, temp_dir=tmp / "t")
        out = tmp / f"up_{carry}_{low}"
        rt.run_inference(model, str(tmp / "plate"), str(out), cfg, WriterConfig(),
                         initial_hint_path=str(tmp / "hint.png"), hint_video_path=None)
        up_calls = list(model.stage1.calls)
        up_a = np.stack([cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                         for p in sorted((out / "plate_alpha_exr").glob("*.exr"))])
        up_fg = np.stack([cv2.imread(str(p), cv2.IMREAD_UNCHANGED)[..., ::-1]
                          for p in sorted((out / "plate_fg_exr").glob("*.exr"))])

        model = tm.Model()
        imgs = torch.from_numpy(frames.astype(np.float32) / 255.0)
        s = KeyerSettings(output_linear=True, despill_strength=0.0, memory_mode="low_vram" if low else "normal",
                          hann_chunk=4, hann_stride=2, carry_hint=carry)
        a, fg = run_keyer(model, imgs, s, hint_masks=torch.from_numpy(hint[None] / 255.0),
                          device="cpu")
        da = np.abs(a.numpy() - up_a).max()
        dfg = np.abs(fg.numpy() - up_fg).max() / max(1e-6, float(np.abs(up_fg).max()))
        same_q = np.allclose(np.array(sum(up_calls, [])), np.array(sum(model.stage1.calls, [])), atol=1e-2)
        print(f"carry={carry} low_vram={low}: alpha max diff {da:.4g}, fg rel diff {dfg:.4g}, "
              f"hint-quality schedule identical: {same_q}")
        assert da < 2e-2 and dfg < 2e-2 and same_q
    print("MATCHES UPSTREAM")


if __name__ == "__main__":
    main()

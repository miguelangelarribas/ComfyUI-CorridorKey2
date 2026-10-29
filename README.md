# ComfyUI-CorridorKey2

ComfyUI custom nodes for **CorridorKey 2.0**, the green/blue-screen keyer by Corridor Digital
(model designed and trained by Zachary Harris). Upstream: https://github.com/nikopueringer/CorridorKey2.0
(vendored at commit `485f5e9`, 2026-10-15).

The model code (`corridorkey/`, `third_party/MoGe/`) is upstream's, unmodified. The ComfyUI layer
(`ck_inference.py`, `ck_loader.py`, `nodes.py`) replaces the CLI's file reader/writers with an
in-memory `IMAGE` batch → `IMAGE` + `MASK` path. The pipeline is the same as upstream: RVM prepass →
Hann-windowed stage 1 → streamed stage 2. On a mock model the port matches upstream's
`run_inference` bit for bit (`tests/test_vs_upstream.py`).

## Install

**ComfyUI-Manager:** search for *CorridorKey 2.0*, or use *Install via Git URL* with this repo.

**Manual:**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/miguelangelarribas/ComfyUI-CorridorKey2
# portable / embedded python:
../../python_embeded/python.exe -m pip install -r ComfyUI-CorridorKey2/requirements.txt
```
Requirements: NVIDIA GPU (RTX 30-series or newer, CUDA PyTorch), ≥4 GB VRAM up to 1440p, ≥6 GB for 4K.

Tested with ComfyUI 0.41, torch 2.8.0+cu128, Python 3.12, transformers 5.17 and huggingface_hub 1.33,
on an RTX 5090. Upstream pins `transformers<5`, but 5.x works here, so you don't need to downgrade.
C-RADIO imports `distutils`, which is missing in Python 3.12 / embedded Pythons. `utils3d` is pinned to a
git commit upstream. Both are handled by shims in `ck_loader.py`, so nothing extra needs installing.

### Weights (~5 GB)
The loader downloads them on first run into `ComfyUI/models/corridorkey2/` (`auto_download`).
To place them manually:
```
corridorkey_v2.safetensors       HF nikopueringer/CorridorKey2.0
foundation_stats.safetensors     HF nikopueringer/CorridorKey2.0
vfms/cradio/C-RADIOv4-SO400M/    HF nvidia/C-RADIOv4-SO400M @ c0457f5 (snapshot, skip *.pth.tar)
vfms/moge/model.pt               HF Ruicheng/moge-2-vitb-normal @ ca5f0e0
vfms/rvm/pretrain_rvm_small16_256_204031069.npz
    https://storage.googleapis.com/representations4d/checkpoints/pretrain_rvm_small16_256_204031069.npz
```
A `corridorkey2:` entry in `extra_model_paths.yaml` is honoured.

## Nodes

**CorridorKey 2.0 Model Loader** → `CORRIDORKEY2_MODEL`. The model is kept in RAM, and the Keyer moves
each stage to the GPU only while it runs.

**CorridorKey 2.0 Keyer**
- `images` (IMAGE, the whole clip as one batch) → `foreground` (IMAGE, straight / un-premultiplied)
  + `alpha` (MASK, 1 = subject).
- `input_colorspace`: `sRGB` for video/PNG/JPG, `linear` for scene-linear plates (EXR).
  This is the same as upstream's `--frame_dir_linear`.
- `output_colorspace`: `sRGB` (clamped 0–1) or `linear` (unclamped HDR, for EXR/comp). These match the
  CLI's `video` and `exr` paths.
- `despill_strength`: 0 = raw model foreground, 0.5 = CLI default, 1 = full despill.
- `memory_mode`: `auto` picks the Hann window and low-VRAM mode from free GPU memory. `normal` and
  `low_vram` use `hann_chunk` / `hann_stride` (stride 0 = chunk/2).
- `hint_mask` (optional MASK, white = subject):
  - 1 frame = first-frame hint (`--hint_first_frame`).
  - N frames = per-frame hint (`--hint_video`).
  - It is resized to the plate's resolution if needed.
- `hint_quality`: how much the model trusts the hint. 0.95 = clean matte, 0.75 = rough automatic,
  ≤0.5 = loose/garbage.
- `carry_hint`: feeds the model's own matte back between overlapping windows. Disable it
  (`--no_carry_hint`) when your per-frame hint is authoritative.

**CorridorKey 2.0 Checker Preview** composites the foreground over a checkerboard in linear light,
like the CLI's `checker` output.

An example workflow is in `example_workflows/CorridorKey2_example.json`.

### Notes
- ComfyUI's `JoinImageWithAlpha` inverts the mask (`alpha = 1 - mask`), so put an `InvertMask` before it.
- The whole clip is held in RAM as float32 (4K × 300 frames ≈ 30 GB for the foreground alone). Split
  long shots, e.g. with VHS `skip_first_frames` / `frame_load_cap`.
- Low-VRAM memmaps go to ComfyUI's temp folder and are deleted after each run.
- Reference numbers (RTX 5090, 96 frames 480×832, synthetic plate):

  | mode | speed | peak VRAM | IoU vs reference |
  |---|---|---|---|
  | auto, no hint | 8.4 fps | 9.3 GB | 0.990 |
  | auto, first-frame hint | 9.0 fps | 9.3 GB | 0.999 |
  | low_vram, window 8 | 6.1 fps | 2.5 GB | 0.998 |

## License
Same license as upstream CorridorKey 2.0 (a variant of CC BY-NC-SA 4.0 with extra terms, see `LICENSE`):
- Free to use, including for commercial work. What you make with it is yours.
- You may not sell it, charge for access to it, or offer it as a paid API or service without
  permission from Corridor Digital.
- Derivatives must stay freely accessible, use this same license and keep the CorridorKey name.
- Organizations with more than US $20M in annual revenue need one CorridorKey plugin seat per user
  (https://corridorkey.com).

The foundation models are not redistributed here. They are downloaded from their publishers under
their own licenses: C-RADIO (NVIDIA Open Model License), MoGe-2 (MIT) and RVM (Apache 2.0 / CC-BY 4.0).
See `THIRD_PARTY_NOTICES.md`.

## Credits
CorridorKey 2.0 by Zachary Harris, Niko Pueringer and Corridor Digital. ComfyUI wrapper by
Miguel Ángel Arribas.

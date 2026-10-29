"""Asset download + model loading for the ComfyUI wrapper.

Layout under <ComfyUI>/models/corridorkey2/ (same names as upstream):
    corridorkey_v2.safetensors
    foundation_stats.safetensors
    vfms/cradio/C-RADIOv4-SO400M/   (HF snapshot)
    vfms/moge/model.pt
    vfms/rvm/pretrain_rvm_small16_256_204031069.npz
"""
from __future__ import annotations

import base64
import hashlib
import os
import shutil
import sys
import types
import urllib.request
from pathlib import Path

NODE_ROOT = Path(__file__).resolve().parent

CORRIDORKEY_REPO = "nikopueringer/CorridorKey2.0"
CORRIDORKEY_FILES = ("corridorkey_v2.safetensors", "foundation_stats.safetensors")
CRADIO_REPO = "nvidia/C-RADIOv4-SO400M"
CRADIO_REVISION = "c0457f5dc26ca145f954cd4fc5bb6114e5705ad8"
MOGE_REPO = "Ruicheng/moge-2-vitb-normal"
MOGE_REVISION = "ca5f0e07ff01d3e5a364c1d954ed12ee1814b368"
RVM_FILE = "pretrain_rvm_small16_256_204031069.npz"
RVM_URL = f"https://storage.googleapis.com/representations4d/checkpoints/{RVM_FILE}"
RVM_MD5_B64 = "ZJb6kj8+tasiJbVeplXu7Q=="


def asset_paths(root: Path) -> dict:
    root = Path(root)
    return {
        "weights": root / "corridorkey_v2.safetensors",
        "foundation_stats": root / "foundation_stats.safetensors",
        "cradio_dir": root / "vfms/cradio/C-RADIOv4-SO400M",
        "moge_checkpoint": root / "vfms/moge/model.pt",
        "rvm_ckpt": root / "vfms/rvm" / RVM_FILE,
    }


def missing_assets(root: Path) -> list[str]:
    p = asset_paths(root)
    out = [k for k, v in p.items() if not v.exists()]
    if p["cradio_dir"].exists() and not (p["cradio_dir"] / "hf_model.py").is_file():
        out.append("cradio_dir")
    return out


def _md5_b64(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode()


def download_assets(root: Path) -> None:
    """Port of scripts/download_weights.py targeting the ComfyUI models folder."""
    if os.environ.get("HF_HUB_OFFLINE", "0") not in ("0", "", "false", "False"):
        raise RuntimeError("HF_HUB_OFFLINE is set; download the weights manually "
                           "(see README) or unset it.")
    from huggingface_hub import hf_hub_download, snapshot_download

    root = Path(root)
    p = asset_paths(root)
    root.mkdir(parents=True, exist_ok=True)
    for name in CORRIDORKEY_FILES:
        if not (root / name).is_file():
            print(f"[CorridorKey2] downloading {CORRIDORKEY_REPO}/{name}")
            hf_hub_download(CORRIDORKEY_REPO, name, local_dir=root)
    if not (p["cradio_dir"] / "hf_model.py").is_file():
        print(f"[CorridorKey2] downloading {CRADIO_REPO}@{CRADIO_REVISION[:7]}")
        snapshot_download(CRADIO_REPO, revision=CRADIO_REVISION,
                          local_dir=p["cradio_dir"], ignore_patterns=["*.pth.tar"])
    if not p["moge_checkpoint"].is_file():
        print(f"[CorridorKey2] downloading {MOGE_REPO}@{MOGE_REVISION[:7]}")
        hf_hub_download(MOGE_REPO, "model.pt", revision=MOGE_REVISION,
                        local_dir=p["moge_checkpoint"].parent)
    dst = p["rvm_ckpt"]
    if not (dst.is_file() and _md5_b64(dst) == RVM_MD5_B64):
        print(f"[CorridorKey2] downloading {RVM_URL}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        with urllib.request.urlopen(RVM_URL) as response, open(tmp, "wb") as f:
            shutil.copyfileobj(response, f, length=1 << 20)
        if _md5_b64(tmp) != RVM_MD5_B64:
            tmp.unlink()
            raise RuntimeError(f"checksum mismatch downloading {RVM_URL}")
        tmp.replace(dst)


def _ensure_utils3d():
    """MoGe imports utils3d at module level but CorridorKey only uses the
    DINOv2 encoder, which never touches it. If the pinned git package is not
    installed, register an empty stub so the import succeeds."""
    try:
        import utils3d  # noqa: F401
    except ImportError:
        stub = types.ModuleType("utils3d")
        stub.__corridorkey_stub__ = True
        sys.modules["utils3d"] = stub


def _ensure_distutils():
    """C-RADIO's remote code does `from distutils.version import LooseVersion`.
    distutils is gone in Python 3.12 and embedded Pythons (ComfyUI portable)
    don't activate setuptools' shim, so alias setuptools' vendored copy."""
    try:
        from distutils.version import LooseVersion  # noqa: F401
        return
    except ImportError:
        pass
    try:
        import setuptools._distutils as _d
        import setuptools._distutils.version as _v
        sys.modules["distutils"] = _d
        sys.modules["distutils.version"] = _v
        return
    except ImportError:
        pass
    from packaging.version import Version

    class LooseVersion(Version):
        def __init__(self, v):
            super().__init__(str(v).split("+")[0])

    mod = types.ModuleType("distutils")
    ver = types.ModuleType("distutils.version")
    ver.LooseVersion = LooseVersion
    mod.version = ver
    sys.modules["distutils"] = mod
    sys.modules["distutils.version"] = ver


def _check_moge_namespace():
    """Another custom node may already have imported a different `moge`."""
    mod = sys.modules.get("moge")
    if mod is None:
        return
    ours = (NODE_ROOT / "third_party" / "MoGe").resolve()
    origin = Path(getattr(mod, "__file__", "") or "").resolve()
    if ours not in origin.parents:
        print(f"[CorridorKey2] WARNING: module 'moge' already imported from {origin}; "
              f"CorridorKey expects the vendored MoGe-2 at {ours}.")


def load_corridorkey(root: Path):
    """Load the model on CPU (moved to GPU per phase at inference time)."""
    from .corridorkey.runtime import RuntimePaths, load_model

    p = asset_paths(root)
    paths = RuntimePaths(package_root=NODE_ROOT, **p)
    _ensure_utils3d()
    _ensure_distutils()
    _check_moge_namespace()
    # load_model() sets HF/Transformers offline flags process-wide; restore them
    # so other custom nodes can still download.
    saved = {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    try:
        model = load_model(paths, device="cpu")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    model.requires_grad_(False)
    return model

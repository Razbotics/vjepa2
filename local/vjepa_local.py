"""Helpers for running V-JEPA 2 / 2.1 / 2-AC from locally downloaded checkpoints.

The upstream hub loaders (src/hub/backbones.py) always fetch from a URL and copy
the file into the torch hub cache. On a laptop with 5-15 GB checkpoints that
doubles disk usage, so this module builds the same architectures with
``pretrained=False`` and loads a state dict from an explicit path instead.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import torch

import src.hub.backbones as hub
from src.hub.backbones import _clean_backbone_key

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(REPO_ROOT, "checkpoints")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Spec:
    builder: str  # function name in src.hub.backbones
    ckpt_file: str  # file name under checkpoints/
    encoder_key: str  # which encoder weights to load from the checkpoint
    img_size: int
    strict: bool  # upstream loads V-JEPA 2 non-strict because of pos_embed vs RoPE
    ac: bool = False  # action-conditioned predictor


SPECS = {
    # V-JEPA 2.1 (distilled from ViT-G, 384px)
    "vjepa2_1_vit_base_384": Spec("vjepa2_1_vit_base_384", "vjepa2_1_vitb_dist_vitG_384.pt", "ema_encoder", 384, True),
    "vjepa2_1_vit_large_384": Spec(
        "vjepa2_1_vit_large_384", "vjepa2_1_vitl_dist_vitG_384.pt", "ema_encoder", 384, True
    ),
    "vjepa2_1_vit_giant_384": Spec("vjepa2_1_vit_giant_384", "vjepa2_1_vitg_384.pt", "target_encoder", 384, True),
    "vjepa2_1_vit_gigantic_384": Spec(
        "vjepa2_1_vit_gigantic_384", "vjepa2_1_vitG_384.pt", "target_encoder", 384, True
    ),
    # V-JEPA 2 (256px)
    "vjepa2_vit_large": Spec("vjepa2_vit_large", "vitl.pt", "target_encoder", 256, False),
    "vjepa2_vit_huge": Spec("vjepa2_vit_huge", "vith.pt", "target_encoder", 256, False),
    "vjepa2_vit_giant": Spec("vjepa2_vit_giant", "vitg.pt", "target_encoder", 256, False),
    "vjepa2_vit_giant_384": Spec("vjepa2_vit_giant_384", "vitg-384.pt", "target_encoder", 384, False),
    # V-JEPA 2-AC (action-conditioned world model, ViT-g encoder)
    "vjepa2_ac_vit_giant": Spec("vjepa2_ac_vit_giant", "vjepa2-ac-vitg.pt", "encoder", 256, False, ac=True),
}


def pick_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(name: str, ckpt_path: str | None = None, device: torch.device | str = "cpu", **kwargs):
    """Build encoder+predictor for ``name`` and load weights from a local checkpoint.

    ``kwargs`` are forwarded to the hub builder (e.g. ``num_frames=16``).
    Returns (encoder, predictor). Both are in eval mode on ``device``.
    """
    spec = SPECS[name]
    ckpt_path = ckpt_path or os.path.join(CKPT_DIR, spec.ckpt_file)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"{ckpt_path} missing: curl -L -o {ckpt_path} {hub.VJEPA_BASE_URL}/{spec.ckpt_file}")

    builder = getattr(hub, spec.builder)
    encoder, predictor = builder(pretrained=False, **kwargs)

    t0 = time.time()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
    enc_msg = encoder.load_state_dict(_clean_backbone_key(state[spec.encoder_key]), strict=spec.strict)
    pred_msg = predictor.load_state_dict(_clean_backbone_key(state["predictor"]), strict=spec.strict or spec.ac)
    print(f"[load_model] {name}: keys={list(state.keys())} load={time.time() - t0:.1f}s")
    print(f"[load_model] encoder: {enc_msg}")
    print(f"[load_model] predictor: {pred_msg}")
    del state

    encoder = encoder.to(device).eval()
    predictor = predictor.to(device).eval()
    return encoder, predictor


def n_params(m: torch.nn.Module) -> float:
    return sum(p.numel() for p in m.parameters()) / 1e6


def load_video(path: str, num_frames: int, size: int) -> torch.Tensor:
    """Read ``num_frames`` evenly spaced frames and return a [1, C, T, H, W] float tensor."""
    from decord import VideoReader, cpu

    import src.datasets.utils.video.transforms as video_transforms
    import src.datasets.utils.video.volume_transforms as volume_transforms

    vr = VideoReader(path, num_threads=-1, ctx=cpu(0))
    idx = np.linspace(0, len(vr) - 1, num=num_frames).astype(np.int64)
    frames = vr.get_batch(idx).asnumpy()  # T H W C uint8
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # T C H W

    short_side = int(256.0 / 224 * size)
    tfm = video_transforms.Compose(
        [
            video_transforms.Resize(short_side, interpolation="bilinear"),
            video_transforms.CenterCrop(size=(size, size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return tfm(frames).unsqueeze(0)  # 1 C T H W


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def peak_mem_gb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**3
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 1024**3
    return float("nan")

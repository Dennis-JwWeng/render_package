"""
DINOv2 ViT-L/14 (registers) singleton loader.
Local weights + local repo checkout. No network access required.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_dinov2_model = None


def load_dinov2(weights_path: str, repo_path: str, device: str = "cuda"):
    """Build ViT-L/14+reg and load pretrained weights from local files."""
    repo = str(Path(repo_path).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)

    from dinov2.hub.backbones import dinov2_vitl14_reg

    model = dinov2_vitl14_reg(pretrained=True, weights=str(weights_path))
    model.eval().to(device)
    return model


def get_dinov2(cfg: dict) -> torch.nn.Module:
    """Process-wide singleton to avoid reloading ~1.2G weights per object."""
    global _dinov2_model
    if _dinov2_model is None:
        weights = cfg["weights"]["dinov2"]
        repo = cfg["weights"]["dinov2_repo"]
        device = cfg.get("encode", {}).get("device", "cuda:0")
        _dinov2_model = load_dinov2(weights, repo, device)
    return _dinov2_model


def release_dinov2():
    """Free the singleton for GPU memory management."""
    global _dinov2_model
    if _dinov2_model is not None:
        del _dinov2_model
        _dinov2_model = None
        torch.cuda.empty_cache()

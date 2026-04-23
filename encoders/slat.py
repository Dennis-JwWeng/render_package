"""
Stage 2b: TRELLIS SLAT encoder -- dino_features.npz -> latents/slat.npz
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

from .utils import atomic_save_npz

DONE_FILE = "slat.npz"

_MIN_VOXELS = 50
_EXPECTED_FEAT_DIM = 8
_EXPECTED_COORD_DIM = 4

_encoder = None


def _get_encoder(cfg: dict):
    global _encoder
    if _encoder is None:
        os.environ.setdefault("SPCONV_ALGO", "native")
        trellis_path = cfg.get("third_party", {}).get("trellis")
        if trellis_path:
            parent = str(os.path.dirname(os.path.abspath(trellis_path)))
            if parent not in sys.path:
                sys.path.insert(0, parent)

        import trellis.models as models

        device = cfg.get("encode", {}).get("device", "cuda:0")
        ckpt_prefix = cfg["weights"]["slat_encoder"]
        _encoder = models.from_pretrained(ckpt_prefix).eval().to(device)
    return _encoder


def _validate_slat(feats: torch.Tensor, coords: torch.Tensor, name: str):
    """Validate SLAT tensors before saving."""
    n = feats.shape[0]
    if n < _MIN_VOXELS:
        raise ValueError(f"{name}: too few voxels ({n} < {_MIN_VOXELS})")
    if feats.shape[1] != _EXPECTED_FEAT_DIM:
        raise ValueError(f"{name}: feat dim {feats.shape[1]}, expected {_EXPECTED_FEAT_DIM}")
    if coords.shape[1] != _EXPECTED_COORD_DIM:
        raise ValueError(f"{name}: coord dim {coords.shape[1]}, expected {_EXPECTED_COORD_DIM}")
    if feats.shape[0] != coords.shape[0]:
        raise ValueError(f"{name}: feats/coords row mismatch")
    if not torch.isfinite(feats).all():
        raise ValueError(f"{name}: non-finite values in feats")
    if (feats == 0).all():
        raise ValueError(f"{name}: all-zero feats (degenerate)")


def encode_slat(scene_dir: str, cfg: dict) -> str | None:
    """Encode TRELLIS SLAT for one object.
    Returns path to saved npz, or None if skipped/failed."""
    latents_dir = os.path.join(scene_dir, "latents")
    out_path = os.path.join(latents_dir, DONE_FILE)
    if os.path.isfile(out_path):
        return out_path

    feat_path = os.path.join(latents_dir, "dino_features.npz")
    if not os.path.isfile(feat_path):
        return None

    os.environ.setdefault("SPCONV_ALGO", "native")
    trellis_path = cfg.get("third_party", {}).get("trellis")
    if trellis_path:
        parent = str(os.path.dirname(os.path.abspath(trellis_path)))
        if parent not in sys.path:
            sys.path.insert(0, parent)
    from trellis.modules import sparse as sp

    data = np.load(feat_path)
    features = data["features"]
    voxel_indices = data["voxel_indices"]

    device = cfg.get("encode", {}).get("device", "cuda:0")

    indices = torch.from_numpy(voxel_indices).long().to(device)
    batch_idx = torch.zeros(indices.shape[0], 1, dtype=torch.int32, device=device)
    full_coords = torch.cat([batch_idx, indices.int()], dim=1)

    aggregated = sp.SparseTensor(
        feats=torch.from_numpy(features.astype(np.float32)).to(device),
        coords=full_coords,
    )

    encoder = _get_encoder(cfg)
    with torch.no_grad():
        latent = encoder(aggregated, sample_posterior=False)

    name = os.path.basename(scene_dir)
    _validate_slat(latent.feats, latent.coords, name)

    atomic_save_npz(
        out_path,
        feats=latent.feats.cpu().float().numpy(),
        coords=latent.coords.cpu().int().numpy(),
    )
    return out_path

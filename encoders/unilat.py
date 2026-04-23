"""
Stage 2a: UniLat encoder -- dino_features.npz -> latents/unilat.npz
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

from .utils import atomic_save_npz

DONE_FILE = "unilat.npz"

_encoder = None


def _get_encoder(cfg: dict):
    global _encoder
    if _encoder is None:
        unilat_path = cfg.get("third_party", {}).get("unilat3d")
        if unilat_path:
            parent = str(os.path.dirname(os.path.abspath(unilat_path)))
            if parent not in sys.path:
                sys.path.insert(0, parent)

        from unilat3d import models as unilat_models

        device = cfg.get("encode", {}).get("device", "cuda:0")
        ckpt_prefix = cfg["weights"]["unilat_encoder"]
        _encoder = unilat_models.from_pretrained(ckpt_prefix).eval().to(device)
    return _encoder


def encode_unilat(scene_dir: str, cfg: dict) -> str | None:
    """Encode UniLat latent for one object.
    Returns path to saved npz, or None if skipped/failed."""
    latents_dir = os.path.join(scene_dir, "latents")
    out_path = os.path.join(latents_dir, DONE_FILE)
    if os.path.isfile(out_path):
        return out_path

    feat_path = os.path.join(latents_dir, "dino_features.npz")
    if not os.path.isfile(feat_path):
        return None

    data = np.load(feat_path)
    features = data["features"]
    voxel_indices = data["voxel_indices"]

    device = cfg.get("encode", {}).get("device", "cuda:0")

    unilat_path = cfg.get("third_party", {}).get("unilat3d")
    if unilat_path:
        parent = str(os.path.dirname(os.path.abspath(unilat_path)))
        if parent not in sys.path:
            sys.path.insert(0, parent)
    from unilat3d.modules.sparse import SparseTensor

    encoder = _get_encoder(cfg)

    indices = torch.from_numpy(voxel_indices).long().to(device)
    batch_idx = torch.zeros(indices.shape[0], 1, dtype=torch.long, device=device)
    full_coords = torch.cat([batch_idx, indices], dim=1).int()
    sparse_input = SparseTensor(
        feats=torch.from_numpy(features).float().to(device),
        coords=full_coords,
    )

    with torch.no_grad():
        latent = encoder(sparse_input, sample_posterior=False)

    lat_np = latent.detach().cpu().numpy()
    atomic_save_npz(out_path, latent=lat_np)
    return out_path

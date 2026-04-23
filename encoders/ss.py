"""
Stage 2c: TRELLIS SS (Sparse Structure) encoder -- slat.npz coords -> latents/ss.npz
Must run after SLAT encoding (needs slat.npz).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

from .utils import atomic_save_npz

DONE_FILE = "ss.npz"

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
        ckpt_prefix = cfg["weights"]["ss_encoder"]
        _encoder = models.from_pretrained(ckpt_prefix).eval().to(device)
    return _encoder


@torch.no_grad()
def _encode_ss(encoder, coords: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    """coords [N,4] (batch_idx, x, y, z) -> z_s [C, R, R, R]."""
    occ = torch.zeros(1, 1, 64, 64, 64, device=device)
    occ[0, 0, coords[:, 1], coords[:, 2], coords[:, 3]] = 1
    z_s = encoder(occ)
    return z_s.squeeze(0)


def encode_ss(scene_dir: str, cfg: dict) -> str | None:
    """Encode TRELLIS SS for one object.
    Returns path to saved npz, or None if skipped/failed."""
    latents_dir = os.path.join(scene_dir, "latents")
    out_path = os.path.join(latents_dir, DONE_FILE)
    if os.path.isfile(out_path):
        return out_path

    slat_path = os.path.join(latents_dir, "slat.npz")
    if not os.path.isfile(slat_path):
        return None

    device = cfg.get("encode", {}).get("device", "cuda:0")
    data = np.load(slat_path)
    coords = torch.from_numpy(data["coords"]).int().to(device)

    encoder = _get_encoder(cfg)
    z_s = _encode_ss(encoder, coords, device)

    atomic_save_npz(out_path, ss=z_s.cpu().float().numpy())
    return out_path

"""
Stage 1: Shared feature extraction -- voxelize mesh + DINOv2 multiview aggregation.
Saves latents/dino_features.npz per object.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from .dinov2_loader import get_dinov2
from .utils import (
    atomic_save_npz,
    ensure_utils3d,
    intrinsics_from_fov_xy,
    project_cv,
    voxelize_mesh,
)

DONE_FILE = "dino_features.npz"


def _build_frame_tensors(frames, images_dir, dino_norm):
    """NeRF transforms.json frames -> list of {image, extrinsics, intrinsics}."""
    tensors = []
    for fr in frames:
        fp = fr["file_path"]
        if fp.startswith("./"):
            fp = fp[2:]
        img_path = os.path.join(images_dir, os.path.basename(fp))
        if not os.path.isfile(img_path):
            img_path = os.path.join(images_dir, fp)
        if not os.path.isfile(img_path):
            continue

        img = Image.open(img_path).convert("RGBA").resize(
            (518, 518), Image.Resampling.LANCZOS
        )
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr[:, :, :3] * arr[:, :, 3:4]
        img_t = dino_norm(torch.from_numpy(arr).permute(2, 0, 1).float())

        c2w = torch.tensor(fr["transform_matrix"], dtype=torch.float32)
        c2w[:3, 1:3] *= -1
        extr = torch.inverse(c2w)
        fov = float(fr["camera_angle_x"])
        intr = intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))

        tensors.append({"image": img_t, "extrinsics": extr, "intrinsics": intr})
    return tensors


@torch.no_grad()
def _aggregate_dino(frame_tensors, positions, dinov2, device, batch_size=8):
    """Run DINOv2 on all views and aggregate per-voxel features."""
    n_patch = 518 // 14
    pos_t = torch.from_numpy(positions).float().to(device)
    acc = torch.zeros(positions.shape[0], 1024, device=device, dtype=torch.float32)
    n_views = len(frame_tensors)

    for i in tqdm(range(0, n_views, batch_size), desc="DINOv2 views", leave=False):
        batch = frame_tensors[i: i + batch_size]
        bs = len(batch)
        imgs = torch.stack([b["image"] for b in batch]).to(device)
        extrs = torch.stack([b["extrinsics"] for b in batch]).to(device)
        intrs = torch.stack([b["intrinsics"] for b in batch]).to(device)

        feats = dinov2(imgs, is_training=True)
        uv = project_cv(pos_t, extrs, intrs)[0] * 2 - 1
        pt = (
            feats["x_prenorm"][:, dinov2.num_register_tokens + 1:]
            .permute(0, 2, 1)
            .reshape(bs, 1024, n_patch, n_patch)
        )
        sampled = (
            F.grid_sample(pt, uv.unsqueeze(1), mode="bilinear", align_corners=False)
            .squeeze(2)
            .permute(0, 2, 1)
        )
        acc += sampled.sum(dim=0).float()

    return (acc / float(n_views)).half()


def extract_features(scene_dir: str, cfg: dict) -> str | None:
    """Extract DINOv2 per-voxel features for one object.
    Returns path to saved npz, or None if skipped/failed."""
    latents_dir = os.path.join(scene_dir, "latents")
    out_path = os.path.join(latents_dir, DONE_FILE)
    if os.path.isfile(out_path):
        return out_path

    mesh_path = os.path.join(scene_dir, "mesh.ply")
    transforms_path = os.path.join(scene_dir, "transforms.json")
    images_dir = os.path.join(scene_dir, "images")
    if not os.path.isdir(images_dir):
        images_dir = scene_dir

    if not os.path.isfile(mesh_path) or not os.path.isfile(transforms_path):
        return None

    ensure_utils3d(cfg)

    vox, positions = voxelize_mesh(mesh_path)

    with open(transforms_path) as f:
        meta = json.load(f)
    frames = meta["frames"]
    max_views = cfg.get("encode", {}).get("max_views", -1)
    if max_views > 0:
        frames = frames[:max_views]

    dino_norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    frame_tensors = _build_frame_tensors(frames, images_dir, dino_norm)
    if not frame_tensors:
        return None

    device = cfg.get("encode", {}).get("device", "cuda:0")
    batch_size = cfg.get("encode", {}).get("batch_size", 8)
    dinov2 = get_dinov2(cfg)

    dino_feats = _aggregate_dino(frame_tensors, positions, dinov2, device, batch_size)

    max_voxels = cfg.get("encode", {}).get("max_voxels", 32768)
    if dino_feats.shape[0] > max_voxels:
        seed = cfg.get("encode", {}).get("subsample_seed", 0)
        rng = np.random.default_rng(seed)
        choice = rng.choice(dino_feats.shape[0], size=max_voxels, replace=False)
        vox = vox[choice]
        dino_feats = dino_feats[choice]

    atomic_save_npz(
        out_path,
        features=dino_feats.cpu().numpy(),
        voxel_indices=vox,
    )
    return out_path

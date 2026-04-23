#!/usr/bin/env python3
"""
Unified multi-GPU encoder for ALL data sources (ABO, HSSD, GitHub, etc.).

Scans render roots for objects (any directory containing mesh.ply),
distributes them across GPU workers, runs DINOv2 + latent encoding.
Handles legacy ABO format (dino_voxel_mean.pt + voxels.ply) automatically.
Fully idempotent: completed objects are skipped on resume.

Usage:
    # Encode everything, auto-detect GPUs
    SPCONV_ALGO=native python encode_all.py \
        --render_root /path/to/renders \
        --render_root /path/to/github/render

    # Use specific GPUs
    SPCONV_ALGO=native python encode_all.py \
        --render_root /path/to/renders \
        --gpus 0,1,2,3

    # Single GPU (legacy behavior)
    SPCONV_ALGO=native python encode_all.py \
        --render_root /path/to/renders \
        --num_gpus 1

    # Dry run: count objects per dataset
    SPCONV_ALGO=native python encode_all.py \
        --render_root /path/to/renders --dry_run

    # Filter by shard or stage
    SPCONV_ALGO=native python encode_all.py \
        --render_root /path/to/renders --shards 0,1,2 --stages slat,ss
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time
import traceback
from collections import Counter
from multiprocessing import Process, Queue

import numpy as np
import torch

os.environ.setdefault("SPCONV_ALGO", "native")

RENDER_PKG = os.path.dirname(os.path.abspath(__file__))
if RENDER_PKG not in sys.path:
    sys.path.insert(0, RENDER_PKG)

_SENTINEL = None  # poison pill


# ── Discovery helpers ────────────────────────────────────────────────

def _has_objects(dirpath: str) -> bool:
    try:
        for e in os.listdir(dirpath)[:20]:
            if os.path.isfile(os.path.join(dirpath, e, "mesh.ply")):
                return True
    except OSError:
        pass
    return False


def discover_tree(root: str, shard_filter: set[str] | None = None
                  ) -> list[tuple[str, str]]:
    """Find shard-level directories whose children have mesh.ply.
    Returns list of (label, shard_path)."""
    results = []

    def _walk(path: str, label: str):
        if _has_objects(path):
            if shard_filter is None or any(part in shard_filter for part in label.split("/")):
                results.append((label, path))
            return
        try:
            entries = sorted(os.listdir(path))
        except OSError:
            return
        for entry in entries:
            full = os.path.join(path, entry)
            if os.path.isdir(full) and not entry.startswith("."):
                _walk(full, f"{label}/{entry}" if label else entry)

    _walk(root, "")
    return results


# ── Legacy feature conversion (ABO) ─────────────────────────────────

def _convert_legacy_features(scene_dir: str) -> bool:
    """Convert ABO-style dino_voxel_mean.pt + voxels.ply -> latents/dino_features.npz."""
    from encoders.utils import atomic_save_npz

    out_path = os.path.join(scene_dir, "latents", "dino_features.npz")
    if os.path.isfile(out_path):
        return True

    pt_path = os.path.join(scene_dir, "dino_voxel_mean.pt")
    ply_path = os.path.join(scene_dir, "voxels.ply")
    if not os.path.isfile(pt_path) or not os.path.isfile(ply_path):
        return False

    features = torch.load(pt_path, map_location="cpu", weights_only=True).numpy()

    from plyfile import PlyData
    p = PlyData.read(ply_path)
    v = p["vertex"]
    pts = np.vstack([v["x"], v["y"], v["z"]]).T
    voxel_indices = ((pts + 0.5) * 64).astype(np.int64)

    n = min(features.shape[0], voxel_indices.shape[0])
    if features.shape[0] != voxel_indices.shape[0]:
        features, voxel_indices = features[:n], voxel_indices[:n]

    atomic_save_npz(out_path, features=features, voxel_indices=voxel_indices)
    return True


# ── Encode worker (one per GPU) ─────────────────────────────────────

def _encode_worker(
    gpu_id: int,
    task_queue: Queue,
    result_queue: Queue,
    cfg_dict: dict,
    stages: list[str],
):
    """Worker process: owns one GPU, pulls objects from shared queue."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("SPCONV_ALGO", "native")

    cfg = copy.deepcopy(cfg_dict)
    cfg["encode"]["device"] = "cuda:0"

    sys.path.insert(0, RENDER_PKG)

    from encoders import encode_object

    while True:
        item = task_queue.get()
        if item is _SENTINEL:
            break

        idx, total, label, scene_dir = item
        obj_name = os.path.basename(scene_dir)
        print(f"  [GPU{gpu_id}] [{idx + 1}/{total}] {label}/{obj_name}", flush=True)

        result = {"label": label, "obj": obj_name}
        dino_npz = os.path.join(scene_dir, "latents", "dino_features.npz")
        downstream_done = all(
            os.path.isfile(os.path.join(scene_dir, "latents", f))
            for f in ("unilat.npz", "slat.npz", "ss.npz")
        )
        need_features = "dino_features" in stages and not os.path.isfile(dino_npz) and not downstream_done

        if need_features:
            try:
                if _convert_legacy_features(scene_dir):
                    result["legacy_convert"] = True
                    need_features = False
            except Exception:
                pass

            if need_features:
                try:
                    from encoders.features import extract_features
                    r = extract_features(scene_dir, cfg)
                    if r:
                        result["dino_features"] = "ok"
                    else:
                        result["dino_features"] = "fail"
                        result_queue.put(result)
                        continue
                except Exception as e:
                    print(f"    [GPU{gpu_id}] [dino FAIL] {obj_name}: {e}", flush=True)
                    result["dino_features"] = "fail"
                    result_queue.put(result)
                    continue
        elif "dino_features" in stages:
            result["dino_features"] = "skip"

        encode_stages = [s for s in stages if s != "dino_features"]
        if encode_stages:
            results = encode_object(scene_dir, cfg, encode_stages)
            result.update(results)

        result_queue.put(result)

    result_queue.put({"_done": True, "gpu_id": gpu_id})


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU encode for all rendered objects"
    )
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--render_root", type=str, action="append", required=True,
                        help="Root directory to scan (can specify multiple times)")
    parser.add_argument("--shards", type=str, default=None,
                        help="Comma-separated shard names to process (default: all)")
    parser.add_argument("--stages", type=str, default=None,
                        help="Comma-separated stages: dino_features,unilat,slat,ss")
    parser.add_argument("--num_gpus", type=int, default=None,
                        help="Number of GPUs (default: auto from config)")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs, e.g. 0,1,2,3")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only count objects, don't encode")
    args = parser.parse_args()

    from encoders.config import load_config
    from encoders import ALL_STAGES

    cfg = load_config(args.config)

    stages = None
    if args.stages:
        stages = [s.strip() for s in args.stages.split(",")]
    else:
        stages = [s for s in ALL_STAGES if cfg.get("stages", {}).get(s, True)]

    shard_filter = None
    if args.shards:
        shard_filter = set(s.strip() for s in args.shards.split(","))

    # Resolve GPU list
    if args.gpus:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
    else:
        gpu_ids = cfg.get("gpus", {}).get("encode", [0])

    if args.num_gpus is not None:
        gpu_ids = gpu_ids[:args.num_gpus]

    num_gpus = len(gpu_ids)

    # Phase 1: discover all objects
    all_jobs: list[tuple[str, str]] = []  # (shard_label, scene_dir)
    for root in args.render_root:
        root = os.path.abspath(root)
        shards = discover_tree(root, shard_filter)
        for shard_label, shard_path in shards:
            objects = [
                d for d in sorted(os.listdir(shard_path))
                if os.path.isdir(os.path.join(shard_path, d))
                and os.path.isfile(os.path.join(shard_path, d, "mesh.ply"))
            ]
            for obj_name in objects:
                scene_dir = os.path.join(shard_path, obj_name)
                all_jobs.append((shard_label, scene_dir))

    total = len(all_jobs)
    print(f"[ENCODE-ALL] {total} objects, {num_gpus} GPU(s) {gpu_ids}, stages={stages}", flush=True)

    if args.dry_run:
        by_shard = Counter(s for s, _ in all_jobs)
        datasets: dict[str, int] = {}
        for shard_label, count in sorted(by_shard.items()):
            ds = shard_label.split("/")[0] if "/" in shard_label else shard_label
            datasets[ds] = datasets.get(ds, 0) + count
        for ds, n in sorted(datasets.items()):
            n_shards = sum(1 for s in by_shard if s.startswith(ds))
            print(f"  {ds:30s}  {n:5d} objects  ({n_shards} shards)")
        return

    if total == 0:
        print("[ENCODE-ALL] Nothing to do.", flush=True)
        return

    t0 = time.time()

    # Phase 2: multi-GPU encode
    task_queue: Queue = Queue()
    result_queue: Queue = Queue()

    for idx, (label, scene_dir) in enumerate(all_jobs):
        task_queue.put((idx, total, label, scene_dir))

    for _ in range(num_gpus):
        task_queue.put(_SENTINEL)

    workers = []
    for gid in gpu_ids:
        p = Process(
            target=_encode_worker,
            args=(gid, task_queue, result_queue, cfg, stages),
            name=f"encode-gpu{gid}",
            daemon=True,
        )
        p.start()
        workers.append(p)

    # Phase 3: collect results
    counts: dict[str, int] = Counter()
    done_workers = 0
    processed = 0

    while done_workers < num_gpus:
        result = result_queue.get()
        if result.get("_done"):
            done_workers += 1
            continue

        processed += 1
        if result.get("legacy_convert"):
            counts["legacy_convert"] += 1
        for stage in ALL_STAGES:
            status = result.get(stage)
            if status in ("ok", "skip", "fail"):
                counts[f"{stage}_{status}"] += 1

    for p in workers:
        p.join(timeout=10)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}", flush=True)
    print(f"[ENCODE-ALL] Done: {processed}/{total} objects in {elapsed:.0f}s ({elapsed / 60:.1f}min)", flush=True)
    print(f"[ENCODE-ALL] GPUs: {gpu_ids}, legacy_convert={counts.get('legacy_convert', 0)}", flush=True)
    for stage in stages:
        ok = counts.get(f"{stage}_ok", 0)
        skip = counts.get(f"{stage}_skip", 0)
        fail = counts.get(f"{stage}_fail", 0)
        print(f"  {stage:20s}  ok={ok:5d}  skip={skip:5d}  fail={fail:5d}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()

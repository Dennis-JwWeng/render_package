"""
Unified 3D latent encoding package.

Public API:
    encode_object(scene_dir, cfg, stages=None) - encode one object
    encode_shard(shard_dir, cfg, stages=None)  - encode all objects in a shard
"""
from __future__ import annotations

import json
import os
import time
import traceback
from typing import Sequence

ALL_STAGES = ("dino_features", "unilat", "slat", "ss")

STAGE_DONE_FILES = {
    "dino_features": "dino_features.npz",
    "unilat": "unilat.npz",
    "slat": "slat.npz",
    "ss": "ss.npz",
}
SKIP_FILE = "encode_skipped.json"


def _is_done(scene_dir: str, stage: str) -> bool:
    return os.path.isfile(os.path.join(scene_dir, "latents", STAGE_DONE_FILES[stage]))


def mark_object_skipped(scene_dir: str, reason: str, stage: str | None = None) -> str:
    """Persist a deterministic data-quality skip for objects that cannot be encoded."""
    latents_dir = os.path.join(scene_dir, "latents")
    os.makedirs(latents_dir, exist_ok=True)
    path = os.path.join(latents_dir, SKIP_FILE)
    payload = {
        "reason": reason,
        "stage": stage,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def object_is_skipped(scene_dir: str) -> bool:
    path = os.path.join(scene_dir, "latents", SKIP_FILE)
    if not os.path.isfile(path):
        return False
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("reason") in {"too_few_voxels", "invalid_mesh"}
    except Exception:
        return False


def _skip_reason_from_exception(exc: Exception) -> str | None:
    msg = str(exc)
    if "too few voxels" in msg:
        return "too_few_voxels"
    if "Empty or invalid mesh" in msg or "Voxelization subprocess failed" in msg:
        return "invalid_mesh"
    return None


def encode_object(
    scene_dir: str,
    cfg: dict,
    stages: Sequence[str] | None = None,
) -> dict[str, str]:
    """Run encoding stages for one object. Returns {stage: "ok"/"skip"/"fail"/"disabled"}."""
    if stages is None:
        stages = [s for s in ALL_STAGES if cfg.get("stages", {}).get(s, True)]

    results = {}

    if "dino_features" in stages:
        downstream_done = all(_is_done(scene_dir, s) for s in ("unilat", "slat", "ss"))
        if _is_done(scene_dir, "dino_features") or downstream_done:
            results["dino_features"] = "skip"
        else:
            try:
                from .features import extract_features
                r = extract_features(scene_dir, cfg)
                results["dino_features"] = "ok" if r else "fail"
            except Exception as e:
                print(f"  [features FAIL] {os.path.basename(scene_dir)}: {e}", flush=True)
                traceback.print_exc()
                skip_reason = _skip_reason_from_exception(e)
                if skip_reason:
                    mark_object_skipped(scene_dir, skip_reason, "dino_features")
                    results["dino_features"] = "skip"
                else:
                    results["dino_features"] = "fail"
    else:
        results["dino_features"] = "disabled"

    if "unilat" in stages:
        if _is_done(scene_dir, "unilat"):
            results["unilat"] = "skip"
        else:
            try:
                from .unilat import encode_unilat
                r = encode_unilat(scene_dir, cfg)
                results["unilat"] = "ok" if r else "fail"
            except Exception as e:
                print(f"  [unilat FAIL] {os.path.basename(scene_dir)}: {e}", flush=True)
                results["unilat"] = "fail"
    else:
        results["unilat"] = "disabled"

    if "slat" in stages:
        if _is_done(scene_dir, "slat"):
            results["slat"] = "skip"
        else:
            try:
                from .slat import encode_slat
                r = encode_slat(scene_dir, cfg)
                results["slat"] = "ok" if r else "fail"
            except Exception as e:
                print(f"  [slat FAIL] {os.path.basename(scene_dir)}: {e}", flush=True)
                skip_reason = _skip_reason_from_exception(e)
                if skip_reason:
                    mark_object_skipped(scene_dir, skip_reason, "slat")
                    results["slat"] = "skip"
                else:
                    results["slat"] = "fail"
    else:
        results["slat"] = "disabled"

    if "ss" in stages:
        if _is_done(scene_dir, "ss"):
            results["ss"] = "skip"
        else:
            try:
                from .ss import encode_ss
                r = encode_ss(scene_dir, cfg)
                results["ss"] = "ok" if r else "fail"
            except Exception as e:
                print(f"  [ss FAIL] {os.path.basename(scene_dir)}: {e}", flush=True)
                results["ss"] = "fail"
    else:
        results["ss"] = "disabled"

    if not cfg.get("stages", {}).get("save_dino_features", False):
        dino_path = os.path.join(scene_dir, "latents", STAGE_DONE_FILES["dino_features"])
        if os.path.isfile(dino_path) and all(
            _is_done(scene_dir, s) for s in ("unilat", "slat", "ss")
        ):
            os.remove(dino_path)
            results["dino_features_cleaned"] = True

    return results


def encode_shard(
    shard_dir: str,
    cfg: dict,
    stages: Sequence[str] | None = None,
) -> dict[str, int]:
    """Encode all rendered objects in a shard directory.
    Returns summary counts {ok, skip, fail} per stage."""
    if not os.path.isdir(shard_dir):
        raise FileNotFoundError(f"Shard dir not found: {shard_dir}")

    objects = sorted([
        d for d in os.listdir(shard_dir)
        if os.path.isdir(os.path.join(shard_dir, d))
        and os.path.isfile(os.path.join(shard_dir, d, "mesh.ply"))
    ])

    print(f"[ENCODE] Shard {os.path.basename(shard_dir)}: {len(objects)} objects", flush=True)
    t0 = time.time()

    counts = {}
    for stage in ALL_STAGES:
        counts[f"{stage}_ok"] = 0
        counts[f"{stage}_skip"] = 0
        counts[f"{stage}_fail"] = 0

    for i, obj_name in enumerate(objects):
        scene_dir = os.path.join(shard_dir, obj_name)
        print(f"  [{i+1}/{len(objects)}] {obj_name}", flush=True)
        results = encode_object(scene_dir, cfg, stages)
        for stage, status in results.items():
            if status in ("ok", "skip", "fail"):
                counts[f"{stage}_{status}"] += 1

    elapsed = time.time() - t0
    print(f"[ENCODE] Done in {elapsed:.0f}s", flush=True)
    for stage in ALL_STAGES:
        ok = counts[f"{stage}_ok"]
        skip = counts[f"{stage}_skip"]
        fail = counts[f"{stage}_fail"]
        if ok + skip + fail > 0:
            print(f"  {stage}: ok={ok} skip={skip} fail={fail}", flush=True)

    return counts

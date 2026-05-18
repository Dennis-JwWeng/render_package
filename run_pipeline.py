#!/usr/bin/env python3
"""
Interleaved two-stage pipeline: render .tar.zst shards → encode latents.

Render and encode run concurrently on separate GPU partitions.
As each shard finishes rendering, its objects are immediately fed to the
encode worker pool. Both stages are fully resumable via per-object done files.

Usage:
    # Full interleaved pipeline (GPUs 0-1 render, GPUs 2-3 encode)
    SPCONV_ALGO=native python run_pipeline.py \
        --config config/default.yaml \
        --render_gpus 0,1 --encode_gpus 2,3

    # Encode-only (skip render, encode existing rendered data)
    SPCONV_ALGO=native python run_pipeline.py \
        --config config/default.yaml \
        --encode_only --render_dir /path/to/renders

    # Render-only (skip encode)
    python run_pipeline.py --config config/default.yaml --render_only
"""
from __future__ import annotations

import argparse
import copy
import glob
import os
import sys
import threading
import time
from collections import Counter
from multiprocessing import Process, Queue

os.environ.setdefault("SPCONV_ALGO", "native")

RENDER_PKG = os.path.dirname(os.path.abspath(__file__))
if RENDER_PKG not in sys.path:
    sys.path.insert(0, RENDER_PKG)

_SENTINEL = "__DONE__"


# ── Encode worker (same pattern as encode_all.py) ───────────────────

def _encode_worker(
    gpu_id: int,
    task_queue: Queue,
    result_queue: Queue,
    cfg_dict: dict,
    stages: list[str],
):
    """Worker process: owns one GPU, encodes objects from shared queue."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("SPCONV_ALGO", "native")

    cfg = copy.deepcopy(cfg_dict)
    cfg["encode"]["device"] = "cuda:0"

    sys.path.insert(0, RENDER_PKG)
    from encoders import encode_object

    while True:
        item = task_queue.get()
        if item == _SENTINEL:
            task_queue.put(_SENTINEL)  # propagate to other workers
            break

        scene_dir = item
        obj_name = os.path.basename(scene_dir)
        print(f"  [ENC:GPU{gpu_id}] {obj_name}", flush=True)

        try:
            results = encode_object(scene_dir, cfg, stages)
            result_queue.put(results)
        except Exception as e:
            print(f"  [ENC:GPU{gpu_id}] FAIL {obj_name}: {e}", flush=True)
            result_queue.put({s: "fail" for s in stages})

    result_queue.put({"_done": True, "gpu_id": gpu_id})


# ── Render producer thread ───────────────────────────────────────────

def _render_thread(
    zst_files: list[str],
    encode_queue: Queue,
    cfg: dict,
    render_gpus: list[int],
    no_delete_shard: bool,
):
    """Render shards one by one, push rendered objects to encode_queue."""
    from render_github import render_shard_from_zst

    render_dir = cfg["paths"]["render_dir"]
    raw_dir = cfg["paths"]["raw_dir"]
    render_tmp = cfg["paths"].get("render_tmp")
    num_workers = cfg.get("render", {}).get("num_workers", 4)
    num_views = cfg.get("render", {}).get("num_views", 40)

    for i, zst_path in enumerate(zst_files):
        shard_name = os.path.basename(zst_path)
        print(f"\n[RENDER] Shard {i + 1}/{len(zst_files)}: {shard_name}", flush=True)

        rendered_dirs = render_shard_from_zst(
            zst_path=zst_path,
            render_dir=render_dir,
            raw_dir=raw_dir,
            gpu_ids=render_gpus,
            num_workers=num_workers,
            num_views=num_views,
            render_tmp=render_tmp,
            no_delete_shard=no_delete_shard,
        )

        print(f"[RENDER] Shard {shard_name}: {len(rendered_dirs)} objects → encode queue", flush=True)
        for d in rendered_dirs:
            encode_queue.put(d)

    encode_queue.put(_SENTINEL)
    print("[RENDER] All shards rendered, signalling encode workers.", flush=True)


# ── Shard state scanner ─────────────────────────────────────────────

def object_encode_complete(scene_dir: str, stages: list[str], cfg: dict | None = None) -> bool:
    """True when all enabled encode stages have outputs on disk (same rules as encode_object skips).

    If save_dino_features is false, dino_features.npz may be absent once unilat+slat+ss exist.
    """
    from encoders import STAGE_DONE_FILES, object_is_skipped

    if object_is_skipped(scene_dir):
        return True
    save_dino = True
    if cfg is not None:
        save_dino = bool(cfg.get("stages", {}).get("save_dino_features", False))
    lat = os.path.join(scene_dir, "latents")
    for s in stages:
        if s not in STAGE_DONE_FILES:
            continue
        name = STAGE_DONE_FILES[s]
        p = os.path.join(lat, name)
        if s == "dino_features" and not save_dino:
            if os.path.isfile(p):
                continue
            if all(
                os.path.isfile(os.path.join(lat, STAGE_DONE_FILES[x]))
                for x in ("unilat", "slat", "ss")
            ):
                continue
            return False
        if not os.path.isfile(p):
            return False
    return True


def object_has_encode_progress(scene_dir: str, stages: list[str], cfg: dict | None = None) -> bool:
    """True if any enabled stage latent artifact exists (encode was started / partial success)."""
    from encoders import STAGE_DONE_FILES

    lat = os.path.join(scene_dir, "latents")
    if not os.path.isdir(lat):
        return False
    for s in stages:
        if s not in STAGE_DONE_FILES:
            continue
        if os.path.isfile(os.path.join(lat, STAGE_DONE_FILES[s])):
            return True
    return False


def object_needs_encode(
    scene_dir: str,
    stages: list[str],
    cfg: dict | None,
    *,
    prior_shard_encode_passes: int,
    retry_policy: str,
) -> bool:
    """Whether this object should still be queued for encode (per watchdog policy).

    * ``any_incomplete``: any missing stage output → keep retrying (legacy).
    * ``never_started_only``: after at least one full shard encode pass
      (``prior_shard_encode_passes >= 1``), only retry objects that still have
      **no** latent files — i.e. never got a real encode product. Objects with
      partial ``latents/*.npz`` are treated as "tried" and not re-queued.
    """
    from encoders import object_is_skipped

    if not os.path.isfile(os.path.join(scene_dir, "mesh.ply")):
        return False
    if object_is_skipped(scene_dir):
        return False
    if object_encode_complete(scene_dir, stages, cfg):
        return False
    if retry_policy != "never_started_only":
        return True
    if prior_shard_encode_passes < 1:
        return True
    return not object_has_encode_progress(scene_dir, stages, cfg)


def _classify_shards(
    render_dir: str,
    stages: list[str],
    cfg: dict | None = None,
    shard_encode_passes: dict[str, int] | None = None,
) -> dict[str, str]:
    """Classify each **shard** directory (one tar → one folder under render_dir).

    Status is **per shard**, but it is computed from **every object subfolder** inside
    that shard: ``encode_done`` means each object has finished all encode stages
    (strict), or under watchdog relaxed policy that nothing left to retry
    (``shard_encode_passes`` + ``encode_retry_objects``). ``render_done`` means
    every object has ``mesh.ply`` but encode is not fully done / still needs work.
    ``partial`` means at least one object is missing ``mesh.ply``.
    """
    status: dict[str, str] = {}
    if not os.path.isdir(render_dir):
        return status

    wd = (cfg.get("pipeline") or {}).get("watchdog") or {} if cfg else {}
    retry_policy = (wd.get("encode_retry_objects") or "any_incomplete").strip().lower()
    if retry_policy not in ("any_incomplete", "never_started_only"):
        retry_policy = "any_incomplete"
    relaxed = shard_encode_passes is not None

    for shard_name in sorted(os.listdir(render_dir)):
        shard_path = os.path.join(render_dir, shard_name)
        if not os.path.isdir(shard_path):
            continue

        objects = [
            d for d in os.listdir(shard_path)
            if os.path.isdir(os.path.join(shard_path, d))
        ]
        if not objects:
            continue

        all_rendered = all(
            os.path.isfile(os.path.join(shard_path, d, "mesh.ply"))
            for d in objects
        )
        passes = int((shard_encode_passes or {}).get(shard_name, 0)) if shard_encode_passes else 0

        if not relaxed:
            all_encoded = all_rendered and all(
                object_encode_complete(os.path.join(shard_path, d), stages, cfg)
                for d in objects
            )
            if all_encoded:
                status[shard_name] = "encode_done"
            elif all_rendered:
                status[shard_name] = "render_done"
            else:
                status[shard_name] = "partial"
            continue

        all_encoded_strict = all_rendered and all(
            object_encode_complete(os.path.join(shard_path, d), stages, cfg)
            for d in objects
        )
        if all_encoded_strict:
            status[shard_name] = "encode_done"
            continue

        any_needs = False
        for d in objects:
            scene_dir = os.path.join(shard_path, d)
            if object_needs_encode(
                scene_dir,
                stages,
                cfg,
                prior_shard_encode_passes=passes,
                retry_policy=retry_policy,
            ):
                any_needs = True
                break

        if all_rendered and not any_needs:
            status[shard_name] = "encode_done"
        elif all_rendered:
            status[shard_name] = "render_done"
        else:
            status[shard_name] = "partial"

    return status


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Interleaved render + encode pipeline")
    parser.add_argument("--config", type=str, default="config/default.yaml")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--encode_only", action="store_true",
                            help="Skip render, encode existing rendered shards")
    mode_group.add_argument("--render_only", action="store_true",
                            help="Render only, no encoding")

    parser.add_argument("--render_dir", type=str, default=None,
                        help="Override render output directory")
    parser.add_argument("--shard_dir", type=str, default=None,
                        help="Override .tar.zst source directory")
    parser.add_argument("--num_shards", type=int, default=None,
                        help="Max shards to process (default: from config)")

    parser.add_argument("--render_gpus", type=str, default=None,
                        help="Comma-separated GPU IDs for render (e.g. 0,1)")
    parser.add_argument("--encode_gpus", type=str, default=None,
                        help="Comma-separated GPU IDs for encode (e.g. 2,3)")

    parser.add_argument("--stages", type=str, default=None,
                        help="Comma-separated encode stages")
    parser.add_argument("--no_delete_shard", action="store_true", default=True)
    args = parser.parse_args()

    from encoders.config import load_config
    from encoders import ALL_STAGES

    cfg = load_config(args.config)

    if args.render_dir:
        cfg["paths"]["render_dir"] = os.path.abspath(args.render_dir)
    if args.shard_dir:
        cfg["paths"]["shard_dir"] = os.path.abspath(args.shard_dir)

    render_gpus = (
        [int(x) for x in args.render_gpus.split(",")]
        if args.render_gpus
        else cfg["gpus"]["render"]
    )
    encode_gpus = (
        [int(x) for x in args.encode_gpus.split(",")]
        if args.encode_gpus
        else cfg["gpus"]["encode"]
    )

    stages = None
    if args.stages:
        stages = [s.strip() for s in args.stages.split(",")]
    else:
        stages = [s for s in ALL_STAGES if cfg.get("stages", {}).get(s, True)]

    num_shards = args.num_shards or cfg.get("pipeline", {}).get("num_shards", 100)
    render_dir = cfg["paths"]["render_dir"]
    shard_dir = cfg["paths"]["shard_dir"]

    print(f"[PIPELINE] render_gpus={render_gpus}  encode_gpus={encode_gpus}", flush=True)
    print(f"[PIPELINE] stages={stages}", flush=True)
    print(f"[PIPELINE] render_dir={render_dir}", flush=True)

    # ── Encode-only mode ─────────────────────────────────────────────
    if args.encode_only:
        print(f"[PIPELINE] Encode-only mode", flush=True)

        shard_status = _classify_shards(render_dir, stages, cfg)
        skip_count = sum(1 for v in shard_status.values() if v == "encode_done")
        todo_shards = [k for k, v in shard_status.items() if v != "encode_done"]

        print(f"[PIPELINE] {len(shard_status)} shards: {skip_count} done, {len(todo_shards)} to encode", flush=True)

        encode_queue: Queue = Queue()
        result_queue: Queue = Queue()

        obj_count = 0
        for shard_name in todo_shards:
            shard_path = os.path.join(render_dir, shard_name)
            for obj_name in sorted(os.listdir(shard_path)):
                obj_dir = os.path.join(shard_path, obj_name)
                if os.path.isdir(obj_dir) and os.path.isfile(os.path.join(obj_dir, "mesh.ply")):
                    if object_encode_complete(obj_dir, stages, cfg):
                        continue
                    encode_queue.put(obj_dir)
                    obj_count += 1

        encode_queue.put(_SENTINEL)
        print(f"[PIPELINE] Queued {obj_count} objects for encoding on GPUs {encode_gpus}", flush=True)

        if obj_count == 0:
            print("[PIPELINE] Nothing pending (all objects already encoded).", flush=True)
            return

        workers = []
        for gid in encode_gpus:
            p = Process(
                target=_encode_worker,
                args=(gid, encode_queue, result_queue, cfg, stages),
                daemon=True,
            )
            p.start()
            workers.append(p)

        t0 = time.time()
        done_workers = 0
        counts: Counter = Counter()
        while done_workers < len(encode_gpus):
            result = result_queue.get()
            if result.get("_done"):
                done_workers += 1
                continue
            for stage in ALL_STAGES:
                st = result.get(stage)
                if st in ("ok", "skip", "fail"):
                    counts[f"{stage}_{st}"] += 1

        for p in workers:
            p.join(timeout=10)

        elapsed = time.time() - t0
        print(f"\n[PIPELINE] Encode done in {elapsed:.0f}s ({elapsed / 60:.1f}min)", flush=True)
        for stage in stages:
            print(f"  {stage}: ok={counts.get(f'{stage}_ok', 0)} "
                  f"skip={counts.get(f'{stage}_skip', 0)} "
                  f"fail={counts.get(f'{stage}_fail', 0)}", flush=True)
        return

    # ── Render-only mode ─────────────────────────────────────────────
    if args.render_only:
        print(f"[PIPELINE] Render-only mode", flush=True)
        from render_github import render_shard_from_zst, get_ready_zst_files

        os.makedirs(shard_dir, exist_ok=True)
        zst_files = get_ready_zst_files(num_shards)
        print(f"[PIPELINE] Found {len(zst_files)} .tar.zst shards", flush=True)

        for i, zst_path in enumerate(zst_files):
            print(f"\n[PIPELINE] Shard {i + 1}/{len(zst_files)}: {os.path.basename(zst_path)}", flush=True)
            render_shard_from_zst(
                zst_path=zst_path,
                render_dir=render_dir,
                raw_dir=cfg["paths"]["raw_dir"],
                gpu_ids=render_gpus,
                num_workers=cfg.get("render", {}).get("num_workers", 4),
                num_views=cfg.get("render", {}).get("num_views", 40),
                render_tmp=cfg["paths"].get("render_tmp"),
                no_delete_shard=args.no_delete_shard,
            )
        print(f"\n[PIPELINE] Render complete.", flush=True)
        return

    # ── Full interleaved pipeline ────────────────────────────────────
    print(f"[PIPELINE] Interleaved render+encode mode", flush=True)
    from render_github import get_ready_zst_files

    zst_files = get_ready_zst_files(num_shards)
    if not zst_files:
        print("[PIPELINE] No .tar.zst shards found. Nothing to do.", flush=True)
        return

    # Scan for already-completed shards to skip
    shard_status = _classify_shards(render_dir, stages, cfg)
    skip_names = {k for k, v in shard_status.items() if v == "encode_done"}
    pending_zst = [
        f for f in zst_files
        if os.path.basename(f).replace(".tar.zst", "") not in skip_names
    ]
    print(f"[PIPELINE] {len(zst_files)} total shards, {len(skip_names)} already done, "
          f"{len(pending_zst)} to process", flush=True)

    if not pending_zst:
        print("[PIPELINE] All shards already encoded. Nothing to do.", flush=True)
        return

    encode_queue: Queue = Queue()
    result_queue: Queue = Queue()

    t0 = time.time()

    # Start encode workers
    enc_workers = []
    for gid in encode_gpus:
        p = Process(
            target=_encode_worker,
            args=(gid, encode_queue, result_queue, cfg, stages),
            daemon=True,
        )
        p.start()
        enc_workers.append(p)

    # Start render producer thread
    render_t = threading.Thread(
        target=_render_thread,
        args=(pending_zst, encode_queue, cfg, render_gpus, args.no_delete_shard),
        daemon=True,
    )
    render_t.start()

    # Collect encode results
    done_workers = 0
    counts: Counter = Counter()
    encoded_objs = 0

    while done_workers < len(encode_gpus):
        result = result_queue.get()
        if result.get("_done"):
            done_workers += 1
            continue
        encoded_objs += 1
        for stage in ALL_STAGES:
            st = result.get(stage)
            if st in ("ok", "skip", "fail"):
                counts[f"{stage}_{st}"] += 1

    render_t.join(timeout=30)
    for p in enc_workers:
        p.join(timeout=10)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}", flush=True)
    print(f"[PIPELINE] Complete: {encoded_objs} objects encoded in {elapsed:.0f}s ({elapsed / 60:.1f}min)", flush=True)
    print(f"[PIPELINE] Render GPUs: {render_gpus}  Encode GPUs: {encode_gpus}", flush=True)
    for stage in stages:
        print(f"  {stage}: ok={counts.get(f'{stage}_ok', 0)} "
              f"skip={counts.get(f'{stage}_skip', 0)} "
              f"fail={counts.get(f'{stage}_fail', 0)}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()

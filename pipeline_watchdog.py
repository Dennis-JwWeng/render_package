#!/usr/bin/env python3
"""
Watchdog entrypoint: validate config → download (range) → render → encode → upload, in a loop, with JSON state.

**Modes** (pick one; split work across two processes if you want render-first, encode-later):

- **Default (full):** download → render → encode → (optional) upload — same state file.
- **`--render-only`:** download → render only. When a shard is fully rendered (all `mesh.ply`),
  mark `render_encode` = `render_only_done`, optionally delete `shards/github/<stem>.tar.zst`
  (`pipeline.watchdog.delete_source_shard_tar_after_render: true`). No encode / no HF upload.
- **`--encode-only`:** no download, no Blender; only encode shards that are `render_done` on disk
  (e.g. after a render-only pass). Then same upload rules as full mode if `hf.upload.enabled`.

Uses the same YAML as run_pipeline / upload_hf_encoded_shards (paths, hf.download, hf.upload, gpus, stages).

State file (default: <data_root>/github/pipeline_state.json):
  - Tracks per-shard download / render_encode / upload and last_error
  - `render_only_done` = render phase finished under `--render-only`, waiting for encode watchdog

Download: only when disk says the shard still needs a local .tar.zst for decompress/render
  (partial / not started). encode_done and render_done skip download; render_done runs encode only.
  Set pipeline.watchdog.prefetch_downloads: true for bulk parallel prefetch at cycle start.

Encode: uses all IDs in gpus.encode (same worker pool as run_pipeline.py). Prefer gpus.pool in YAML so
  render and encode share one list (full GPUs for Blender, then the same GPUs for encode).

Usage:
  export HUGGINGFACE_HUB_TOKEN=...   # if upload enabled (or hf.upload.token in <config>.local.yaml)
  export SPCONV_ALGO=native
  python pipeline_watchdog.py --config config/default.yaml
  python pipeline_watchdog.py --config config/default.yaml --render-only
  python pipeline_watchdog.py --config config/default.yaml --encode-only

  python pipeline_watchdog.py --config config/default.yaml --once   # single cycle then exit
"""
from __future__ import annotations

import argparse
import json
import queue
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any

RENDER_PKG = os.path.dirname(os.path.abspath(__file__))
if RENDER_PKG not in sys.path:
    sys.path.insert(0, RENDER_PKG)

from encoders.config import hf_hub_token  # noqa: E402

STATE_VERSION = 1
STOP = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _state_path(cfg: dict) -> str:
    wd = cfg.get("pipeline", {}).get("watchdog", {}) or {}
    p = wd.get("state_file")
    if p:
        return os.path.abspath(os.path.expanduser(p))
    dr = cfg["paths"]["data_root"]
    return os.path.join(dr, "github", "pipeline_state.json")


def load_state(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        return {"version": STATE_VERSION, "cycle": 0, "updated": None, "shards": {}}
    with open(path) as f:
        data = json.load(f)
    if data.get("version") != STATE_VERSION:
        data = {"version": STATE_VERSION, "cycle": data.get("cycle", 0), "updated": None, "shards": data.get("shards", {})}
    data.setdefault("shards", {})
    return data


def save_state(path: str, state: dict[str, Any]) -> None:
    state["updated"] = _utc_now()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _shard_entry() -> dict[str, Any]:
    return {
        "download": "pending",
        "render_encode": "pending",
        "upload": "pending",
        "last_error": None,
        "encode_passes": 0,
    }


def validate_config(cfg: dict, config_path: str, *, watchdog_mode: str = "full") -> None:
    paths = cfg["paths"]
    hf_dl = (cfg.get("hf") or {}).get("download") or {}
    if not hf_dl.get("repo_pattern") or hf_dl.get("suffix") is None:
        raise SystemExit("hf.download.repo_pattern and suffix are required")
    s0 = int(hf_dl["shard_index_start"])
    s1 = int(hf_dl["shard_index_end"])
    if s0 > s1:
        raise SystemExit(f"Invalid shard index range [{s0}, {s1}]")
    dest = hf_dl.get("dest") or paths["data_root"]
    if not os.path.isdir(dest):
        try:
            os.makedirs(dest, exist_ok=True)
        except OSError as e:
            raise SystemExit(f"Cannot create data_root/dest {dest}: {e}") from e
    for key in ("shard_dir", "render_dir", "raw_dir"):
        os.makedirs(paths[key], exist_ok=True)
    if watchdog_mode != "encode_only" and cfg.get("stages", {}).get("render", True):
        b = paths.get("blender_bin")
        if not b or not os.path.isfile(b):
            raise SystemExit(f"Blender not found at paths.blender_bin: {b}")
    up = (cfg.get("hf") or {}).get("upload") or {}
    if up.get("enabled", False) and watchdog_mode != "render_only":
        if not hf_hub_token(cfg):
            raise SystemExit(
                "hf.upload.enabled but no token: set HUGGINGFACE_HUB_TOKEN / HF_TOKEN or hf.upload.token in .local.yaml"
            )
        if not up.get("repo_id"):
            raise SystemExit("hf.upload.repo_id empty")


def expected_stems(cfg: dict) -> list[str]:
    hf_dl = cfg["hf"]["download"]
    repo = hf_dl["repo_pattern"].format(hf_dl["suffix"])
    mirror = hf_dl.get("mirror", "https://hf-mirror.com")
    import download_trellis as dlt

    prev = dlt.MIRROR
    dlt.MIRROR = mirror
    try:
        return dlt.expected_shard_stems(repo, int(hf_dl["shard_index_start"]), int(hf_dl["shard_index_end"]))
    finally:
        dlt.MIRROR = prev


def run_download_phase(cfg: dict) -> None:
    hf_dl = cfg["hf"]["download"]
    wd = cfg.get("pipeline", {}).get("watchdog", {}) or {}
    repo = hf_dl["repo_pattern"].format(hf_dl["suffix"])
    dest = hf_dl.get("dest") or cfg["paths"]["data_root"]
    mirror = hf_dl.get("mirror", "https://hf-mirror.com")
    workers = int(wd.get("download_workers", 4))
    retries = int(wd.get("download_retries", 5))
    shards_only = bool(wd.get("shards_only_download", True))
    manifest_only = bool(wd.get("manifest_only_download", False))

    import download_trellis as dlt

    prev = dlt.MIRROR
    dlt.MIRROR = mirror
    try:
        tasks = dlt.build_download_task_list(
            repo,
            int(hf_dl["shard_index_start"]),
            int(hf_dl["shard_index_end"]),
            manifest_only=manifest_only,
            shards_only=shards_only,
            skip_names=None,
        )
        if not tasks:
            print("[WATCHDOG] Download: no tasks in range.", flush=True)
            return
        print(f"[WATCHDOG] Download: {len(tasks)} task(s) (shards_only={shards_only}, manifest_only={manifest_only})", flush=True)
        stats = dlt.run_download_tasks(repo, dest, tasks, workers=workers, retries=retries, quiet=False)
        print(f"[WATCHDOG] Download done: ok={stats['ok']} skip={stats['skip']} fail={stats['fail']}", flush=True)
        if stats["fail"] > 0:
            print("[WATCHDOG] WARN: some downloads failed; will retry next cycle.", flush=True)
    finally:
        dlt.MIRROR = prev


def download_one_shard_tar(cfg: dict, stem: str) -> bool:
    """Fetch shards/github/<stem>.tar.zst when missing (one file, mirror + single worker)."""
    hf_dl = cfg["hf"]["download"]
    wd = cfg.get("pipeline", {}).get("watchdog", {}) or {}
    repo = hf_dl["repo_pattern"].format(hf_dl["suffix"])
    dest = hf_dl.get("dest") or cfg["paths"]["data_root"]
    mirror = hf_dl.get("mirror", "https://hf-mirror.com")
    retries = int(wd.get("download_retries", 5))
    import download_trellis as dlt

    prev = dlt.MIRROR
    dlt.MIRROR = mirror
    try:
        return dlt.download_shard_tar(repo, dest, stem, retries=retries, quiet=False)
    finally:
        dlt.MIRROR = prev


def run_encode_shard_multigpu(
    config_path: str,
    shard_render_dir: str,
    encode_gpus: list[int],
    *,
    prior_encode_passes: int = 0,
    encode_retry_policy: str = "any_incomplete",
) -> None:
    """Encode one shard directory using one worker process per GPU (shared task queue)."""
    from collections import Counter
    from multiprocessing import Process, Queue

    from encoders import ALL_STAGES
    from encoders.config import load_config
    from run_pipeline import _SENTINEL, _encode_worker, object_encode_complete, object_needs_encode

    if not os.path.isdir(shard_render_dir):
        print(f"[WATCHDOG] Encode skipped: missing render dir {shard_render_dir}", flush=True)
        return

    cfg = load_config(config_path)
    stages = [s for s in ALL_STAGES if cfg.get("stages", {}).get(s, True)]
    gpus = list(encode_gpus) if encode_gpus else [0]

    objects = sorted(
        [
            os.path.join(shard_render_dir, d)
            for d in os.listdir(shard_render_dir)
            if os.path.isdir(os.path.join(shard_render_dir, d))
            and os.path.isfile(os.path.join(shard_render_dir, d, "mesh.ply"))
        ]
    )
    if not objects:
        print(f"[WATCHDOG] No objects with mesh.ply under {shard_render_dir}", flush=True)
        return

    pol = (encode_retry_policy or "any_incomplete").strip().lower()
    if pol not in ("any_incomplete", "never_started_only"):
        pol = "any_incomplete"
    todo = [
        p
        for p in objects
        if object_needs_encode(
            p,
            stages,
            cfg,
            prior_shard_encode_passes=prior_encode_passes,
            retry_policy=pol,
        )
    ]
    stem = os.path.basename(shard_render_dir)
    if not todo:
        n_inc = sum(1 for p in objects if not object_encode_complete(p, stages, cfg))
        if n_inc == 0:
            print(
                f"[WATCHDOG] Encode shard {stem}: all {len(objects)} objects already complete (skipped worker pool)",
                flush=True,
            )
        else:
            print(
                f"[WATCHDOG] Encode shard {stem}: {n_inc} object(s) still incomplete but none queued "
                f"(retry policy={pol!r}, prior_passes={prior_encode_passes}; skipped worker pool)",
                flush=True,
            )
        return

    print(
        f"[WATCHDOG] Encode shard {stem}: {len(todo)}/{len(objects)} objects need work on GPUs {gpus}",
        flush=True,
    )

    encode_queue: Queue = Queue()
    result_queue: Queue = Queue()
    for scene_dir in todo:
        encode_queue.put(scene_dir)
    encode_queue.put(_SENTINEL)

    workers: list[Process] = []
    for gid in gpus:
        p = Process(
            target=_encode_worker,
            args=(gid, encode_queue, result_queue, cfg, stages),
            daemon=True,
        )
        p.start()
        workers.append(p)

    done_workers = 0
    counts: Counter = Counter()
    while done_workers < len(gpus):
        result = result_queue.get()
        if result.get("_done"):
            done_workers += 1
            continue
        for stage in ALL_STAGES:
            st = result.get(stage)
            if st in ("ok", "skip", "fail"):
                counts[f"{stage}_{st}"] += 1

    for p in workers:
        p.join(timeout=60)

    for stage in stages:
        ok = counts.get(f"{stage}_ok", 0)
        sk = counts.get(f"{stage}_skip", 0)
        fl = counts.get(f"{stage}_fail", 0)
        if ok + sk + fl > 0:
            print(f"[WATCHDOG]   {stage}: ok={ok} skip={sk} fail={fl}", flush=True)


def run_upload_subprocess(config_path: str, shard_stem: str) -> None:
    script = os.path.join(RENDER_PKG, "upload_hf_encoded_shards.py")
    subprocess.check_call(
        [sys.executable, script, "--config", config_path, "--shards", shard_stem],
        env=os.environ.copy(),
    )


def _classify_all(
    render_dir: str,
    stages: list[str],
    cfg: dict,
    state: dict[str, Any] | None = None,
) -> dict[str, str]:
    from run_pipeline import _classify_shards

    shard_passes: dict[str, int] | None = None
    if state is not None:
        shard_passes = {
            k: int(v.get("encode_passes", 0))
            for k, v in state.get("shards", {}).items()
        }
    return _classify_shards(render_dir, stages, cfg, shard_encode_passes=shard_passes)


def _remote_tar_set(
    repo_id: str, repo_type: str, path_in_repo: str, token: str | None = None
) -> set[str]:
    from huggingface_hub import HfApi

    prefix = path_in_repo.strip("/") + "/"
    names: set[str] = set()
    for p in HfApi(token=token).list_repo_files(repo_id, repo_type=repo_type):
        if p.startswith(prefix) and p.endswith(".tar.zst"):
            names.add(os.path.basename(p).replace(".tar.zst", ""))
    return names


def _delete_source_shard_tar_after_render_if_enabled(cfg: dict, shard_stem: str) -> None:
    """Remove shards/github/<stem>.tar.zst when pipeline.watchdog.delete_source_shard_tar_after_render is true."""
    wd = (cfg.get("pipeline", {}).get("watchdog", {}) or {})
    if not wd.get("delete_source_shard_tar_after_render", False):
        return
    shard_dir = cfg["paths"]["shard_dir"]
    zst = os.path.join(shard_dir, f"{shard_stem}.tar.zst")
    if os.path.isfile(zst):
        os.remove(zst)
        print(f"[CLEANUP] removed source shard tar after render {zst}", flush=True)


def _try_upload_shard_after_encode_done(
    config_path: str,
    cfg: dict,
    stem: str,
    ent: dict[str, Any],
    remote_shards: set[str] | None,
    hf_token: str | None,
) -> None:
    """Run HF upload + tar cleanup when hf.upload.enabled and shard is encode_done on disk."""
    hf_up = (cfg.get("hf") or {}).get("upload") or {}
    upload_enabled = bool(hf_up.get("enabled", False))
    path_in_repo = (hf_up.get("path_in_repo") or "github/render").strip("/")

    from huggingface_hub import HfApi
    from upload_hf_encoded_shards import _remote_size, delete_source_shard_tar_if_enabled

    hf_api = HfApi(token=hf_token)

    if not upload_enabled:
        ent["upload"] = "skipped"
        delete_source_shard_tar_if_enabled(cfg, stem)
        return

    if remote_shards is not None and stem in remote_shards:
        ent["upload"] = "done"
        ent["last_error"] = None
        if cfg.get("pipeline", {}).get("delete_source_shard_tar"):
            rel = f"{path_in_repo}/{stem}.tar.zst"
            rsz = _remote_size(
                hf_api,
                hf_up["repo_id"],
                hf_up.get("repo_type", "dataset"),
                rel,
            )
            if rsz and rsz > 0:
                delete_source_shard_tar_if_enabled(cfg, stem)
        return

    print(f"[WATCHDOG] {stem}: upload", flush=True)
    try:
        run_upload_subprocess(os.path.abspath(config_path), stem)
        if remote_shards is not None:
            remote_shards.add(stem)
        ent["upload"] = "done"
        ent["last_error"] = None
    except Exception as e:
        ent["upload"] = "error"
        ent["last_error"] = str(e)
        print(f"[WATCHDOG] upload ERROR {stem}: {e}", flush=True)


def _encode_upload_one_shard(
    config_path: str,
    cfg: dict,
    state: dict[str, Any],
    stem: str,
    encode_gpus: list[int],
    stages: list[str],
    remote_shards: set[str] | None,
    hf_token: str | None,
    max_encode_passes: int,
    encode_retry_policy: str,
    incomplete_as: str,
) -> None:
    render_dir = cfg["paths"]["render_dir"]
    shard_render = os.path.join(render_dir, stem)
    ent = state["shards"].setdefault(stem, _shard_entry())
    passes = int(ent.get("encode_passes", 0))
    if max_encode_passes >= 0 and passes >= max_encode_passes:
        print(
            f"[WATCHDOG] {stem}: skip encode (encode_passes={passes} >= max_encode_passes={max_encode_passes}); "
            f"keeping existing outputs on disk",
            flush=True,
        )
        ent["render_encode"] = "partial"
        ent["last_error"] = None if incomplete_as == "partial" else "max_encode_passes_reached"
        return

    try:
        if not os.path.isdir(shard_render):
            raise FileNotFoundError(f"render_done but render dir missing: {shard_render}")
        print(f"[WATCHDOG] {stem}: encode/upload worker start", flush=True)
        run_encode_shard_multigpu(
            os.path.abspath(config_path),
            shard_render,
            encode_gpus,
            prior_encode_passes=int(ent.get("encode_passes", 0)),
            encode_retry_policy=encode_retry_policy,
        )
        ent["encode_passes"] = int(ent.get("encode_passes", 0)) + 1
        classify = _classify_all(render_dir, stages, cfg, state)
        st = classify.get(stem)
        if st == "encode_done":
            ent["render_encode"] = "done"
            ent["last_error"] = None
            ent["encode_passes"] = 0
            _try_upload_shard_after_encode_done(
                config_path, cfg, stem, ent, remote_shards, hf_token
            )
        elif incomplete_as == "partial":
            ent["render_encode"] = "partial"
            ent["last_error"] = None
            print(
                f"[WATCHDOG] {stem}: encode still incomplete after pass — "
                f"treating as partial (disk outputs kept; no re-download). "
                f"status={st!r}",
                flush=True,
            )
        else:
            ent["render_encode"] = "error"
            ent["last_error"] = "encode_incomplete_after_pass"
    except Exception as e:
        ent["render_encode"] = "error"
        ent["last_error"] = str(e)
        print(f"[WATCHDOG] ERROR {stem}: {e}", flush=True)


def _has_mesh_outputs(shard_render: str) -> bool:
    if not os.path.isdir(shard_render):
        return False
    return any(
        os.path.isdir(os.path.join(shard_render, d))
        and os.path.isfile(os.path.join(shard_render, d, "mesh.ply"))
        for d in os.listdir(shard_render)
    )


def one_cycle_full_overlap(
    config_path: str,
    cfg: dict,
    state: dict[str, Any],
    remote_shards: set[str] | None,
    hf_token: str | None,
) -> None:
    paths = cfg["paths"]
    shard_dir = paths["shard_dir"]
    render_dir = paths["render_dir"]
    raw_dir = paths["raw_dir"]
    render_tmp = paths.get("render_tmp")

    from encoders import ALL_STAGES
    import render_github as rg

    stages = [s for s in ALL_STAGES if cfg.get("stages", {}).get(s, True)]
    stems = expected_stems(cfg)
    if not stems and state.get("shards"):
        stems = sorted(state["shards"].keys())
        print(
            f"[WATCHDOG] expected_stems empty (HF mirror/API list failed); "
            f"falling back to {len(stems)} id(s) from pipeline_state.json",
            flush=True,
        )
    classify = _classify_all(render_dir, stages, cfg, state)

    render_gpus = cfg["gpus"]["render"]
    encode_gpus = cfg["gpus"]["encode"]
    if set(render_gpus) & set(encode_gpus):
        print(
            "[WATCHDOG] WARN: overlap_render_encode is enabled but render and encode GPU lists overlap; "
            "this may increase memory pressure.",
            flush=True,
        )
    num_workers = int(cfg.get("render", {}).get("num_workers", 4))
    num_views = int(cfg.get("render", {}).get("num_views", 40))
    no_delete_shard = bool(cfg.get("pipeline", {}).get("no_delete_shard", True))
    wd = cfg.get("pipeline", {}).get("watchdog", {}) or {}
    incomplete_as = (wd.get("incomplete_encode_as") or "error").strip().lower()
    if incomplete_as not in ("error", "partial"):
        incomplete_as = "error"
    max_encode_passes = int(wd.get("max_encode_passes", -1))
    encode_retry_policy = (wd.get("encode_retry_objects") or "any_incomplete").strip().lower()
    if encode_retry_policy not in ("any_incomplete", "never_started_only"):
        encode_retry_policy = "any_incomplete"

    encode_queue: queue.Queue[str | None] = queue.Queue()

    def _worker() -> None:
        while True:
            stem_or_none = encode_queue.get()
            try:
                if stem_or_none is None:
                    return
                _encode_upload_one_shard(
                    config_path,
                    cfg,
                    state,
                    stem_or_none,
                    encode_gpus,
                    stages,
                    remote_shards,
                    hf_token,
                    max_encode_passes,
                    encode_retry_policy,
                    incomplete_as,
                )
            finally:
                encode_queue.task_done()

    worker = threading.Thread(target=_worker, name="watchdog-encode-upload", daemon=True)
    worker.start()
    queued: set[str] = set()

    try:
        for stem in stems:
            if STOP:
                break
            zst = os.path.join(shard_dir, stem + ".tar.zst")
            ent = state["shards"].setdefault(stem, _shard_entry())

            if ent.get("skip_reason") == "empty_tar_no_models":
                print(
                    f"[WATCHDOG] {stem}: skip (tar produced no models last time; clear shard in pipeline_state to retry)",
                    flush=True,
                )
                continue

            st = classify.get(stem)
            if st == "encode_done":
                ent["encode_passes"] = 0
                ent["last_error"] = None
                ent.pop("skip_reason", None)
                ent["render_encode"] = "done"
                ent["download"] = "done" if os.path.isfile(zst) and os.path.getsize(zst) > 0 else "not_required"
                print(f"[WATCHDOG] {stem}: encode_done — skip tar download and render", flush=True)
                _try_upload_shard_after_encode_done(config_path, cfg, stem, ent, remote_shards, hf_token)
                continue

            needs_tar = st not in ("encode_done", "render_done")
            if needs_tar:
                if not (os.path.isfile(zst) and os.path.getsize(zst) > 0):
                    print(f"[WATCHDOG] {stem}: download .tar.zst (needed for decompress/render)", flush=True)
                    if not download_one_shard_tar(cfg, stem):
                        ent["download"] = "pending"
                        ent["last_error"] = "download_failed"
                        continue
                    if not (os.path.isfile(zst) and os.path.getsize(zst) > 0):
                        ent["download"] = "pending"
                        ent["last_error"] = "missing_local_tar_after_download"
                        continue
                ent["download"] = "done"
            else:
                ent["download"] = "done" if os.path.isfile(zst) and os.path.getsize(zst) > 0 else "not_required"
                print(f"[WATCHDOG] {stem}: render_done — skip tar download; queue encode", flush=True)

            if st != "render_done":
                print(f"[WATCHDOG] {stem}: render (overlap mode; status was {st})", flush=True)
                shard_render = os.path.join(render_dir, stem)
                try:
                    ent["render_encode"] = "in_progress"
                    rg.render_shard_from_zst(
                        zst_path=zst,
                        render_dir=render_dir,
                        raw_dir=raw_dir,
                        gpu_ids=render_gpus,
                        num_workers=num_workers,
                        num_views=num_views,
                        render_tmp=render_tmp,
                        no_delete_shard=no_delete_shard,
                    )
                    if not os.path.isdir(shard_render):
                        print(
                            f"[WATCHDOG] {stem}: no render output (0 models or render failed); skip encode — "
                            f"check tarball or clear skip_reason in state to retry",
                            flush=True,
                        )
                        ent["skip_reason"] = "empty_tar_no_models"
                        ent["render_encode"] = "partial"
                        ent["last_error"] = None if incomplete_as == "partial" else "empty_tar_no_models"
                        continue
                    if not _has_mesh_outputs(shard_render):
                        print(f"[WATCHDOG] {stem}: render dir has no mesh.ply; skip encode", flush=True)
                        ent["skip_reason"] = "empty_tar_no_models"
                        ent["render_encode"] = "partial"
                        ent["last_error"] = None if incomplete_as == "partial" else "empty_tar_no_models"
                        continue
                except Exception as e:
                    ent["render_encode"] = "error"
                    ent["last_error"] = str(e)
                    print(f"[WATCHDOG] ERROR {stem}: {e}", flush=True)
                    continue

            if stem not in queued:
                ent["render_encode"] = "encode_queued"
                ent["last_error"] = None
                queued.add(stem)
                encode_queue.put(stem)
                print(f"[WATCHDOG] {stem}: queued encode/upload; continuing render pipeline", flush=True)
    finally:
        encode_queue.put(None)
        encode_queue.join()
        worker.join()


def one_cycle(
    config_path: str,
    cfg: dict,
    state: dict[str, Any],
    remote_shards: set[str] | None,
    hf_token: str | None = None,
    *,
    mode: str = "full",
) -> None:
    paths = cfg["paths"]
    shard_dir = paths["shard_dir"]
    render_dir = paths["render_dir"]
    raw_dir = paths["raw_dir"]
    render_tmp = paths.get("render_tmp")

    from encoders import ALL_STAGES

    stages = [s for s in ALL_STAGES if cfg.get("stages", {}).get(s, True)]

    stems = expected_stems(cfg)
    if not stems and state.get("shards"):
        stems = sorted(state["shards"].keys())
        print(
            f"[WATCHDOG] expected_stems empty (HF mirror/API list failed); "
            f"falling back to {len(stems)} id(s) from pipeline_state.json",
            flush=True,
        )
    classify = _classify_all(render_dir, stages, cfg, state)

    render_gpus = cfg["gpus"]["render"]
    encode_gpus = cfg["gpus"]["encode"]
    num_workers = int(cfg.get("render", {}).get("num_workers", 4))
    num_views = int(cfg.get("render", {}).get("num_views", 40))
    no_delete_shard = bool(cfg.get("pipeline", {}).get("no_delete_shard", True))
    wd = cfg.get("pipeline", {}).get("watchdog", {}) or {}
    incomplete_as = (wd.get("incomplete_encode_as") or "error").strip().lower()
    if incomplete_as not in ("error", "partial"):
        incomplete_as = "error"
    max_encode_passes = int(wd.get("max_encode_passes", -1))
    encode_retry_policy = (wd.get("encode_retry_objects") or "any_incomplete").strip().lower()
    if encode_retry_policy not in ("any_incomplete", "never_started_only"):
        encode_retry_policy = "any_incomplete"

    import render_github as rg

    for stem in stems:
        if STOP:
            break
        zst = os.path.join(shard_dir, stem + ".tar.zst")
        ent = state["shards"].setdefault(stem, _shard_entry())

        if ent.get("skip_reason") == "empty_tar_no_models":
            print(
                f"[WATCHDOG] {stem}: skip (tar produced no models last time; clear shard in pipeline_state to retry)",
                flush=True,
            )
            continue

        st = classify.get(stem)

        # ── Render-only watchdog: download + Blender only; optional delete .tar.zst after full render ──
        if mode == "render_only":
            if st == "encode_done":
                ent["encode_passes"] = 0
                ent["last_error"] = None
                ent.pop("skip_reason", None)
                ent["render_encode"] = "done"
                print(f"[WATCHDOG] {stem}: encode_done — skip (render-only mode)", flush=True)
                ent["upload"] = "skipped"
                continue
            if ent.get("render_encode") == "render_only_done":
                print(f"[WATCHDOG] {stem}: render_only_done — skip", flush=True)
                continue

            needs_tar = st not in ("encode_done", "render_done")
            if needs_tar:
                if not (os.path.isfile(zst) and os.path.getsize(zst) > 0):
                    print(f"[WATCHDOG] {stem}: download .tar.zst (render-only)", flush=True)
                    if not download_one_shard_tar(cfg, stem):
                        ent["download"] = "pending"
                        ent["last_error"] = "download_failed"
                        continue
                    if not (os.path.isfile(zst) and os.path.getsize(zst) > 0):
                        ent["download"] = "pending"
                        ent["last_error"] = "missing_local_tar_after_download"
                        continue
                ent["download"] = "done"
            else:
                if os.path.isfile(zst) and os.path.getsize(zst) > 0:
                    ent["download"] = "done"
                else:
                    ent["download"] = "not_required"
                print(f"[WATCHDOG] {stem}: skip tar download (render-only; status={st})", flush=True)

            if st == "render_done":
                ent["render_encode"] = "render_only_done"
                ent["last_error"] = None
                _delete_source_shard_tar_after_render_if_enabled(cfg, stem)
                ent["upload"] = "skipped"
                print(f"[WATCHDOG] {stem}: render-only checkpoint (all mesh on disk)", flush=True)
                continue

            print(f"[WATCHDOG] {stem}: render-only render (status was {st})", flush=True)
            try:
                shard_render = os.path.join(render_dir, stem)
                rg.render_shard_from_zst(
                    zst_path=zst,
                    render_dir=render_dir,
                    raw_dir=raw_dir,
                    gpu_ids=render_gpus,
                    num_workers=num_workers,
                    num_views=num_views,
                    render_tmp=render_tmp,
                    no_delete_shard=no_delete_shard,
                )
                classify = _classify_all(render_dir, stages, cfg, state)
                st2 = classify.get(stem)
                if st2 not in ("render_done", "encode_done"):
                    if not os.path.isdir(shard_render):
                        ent["skip_reason"] = "empty_tar_no_models"
                    ent["render_encode"] = "partial"
                    ent["last_error"] = (
                        None if incomplete_as == "partial" else "render_incomplete_render_only"
                    )
                    print(
                        f"[WATCHDOG] {stem}: render-only incomplete after render (status={st2!r})",
                        flush=True,
                    )
                    continue
                if st2 == "encode_done":
                    ent["render_encode"] = "done"
                else:
                    ent["render_encode"] = "render_only_done"
                ent["last_error"] = None
                ent["encode_passes"] = 0
                _delete_source_shard_tar_after_render_if_enabled(cfg, stem)
                ent["upload"] = "skipped"
            except Exception as e:
                ent["render_encode"] = "error"
                ent["last_error"] = str(e)
                print(f"[WATCHDOG] ERROR {stem}: {e}", flush=True)
            continue

        # ── Encode-only watchdog: no download / no Blender; encode + optional upload ──
        if mode == "encode_only":
            if st == "encode_done":
                ent["encode_passes"] = 0
                ent["last_error"] = None
                ent.pop("skip_reason", None)
                ent["render_encode"] = "done"
                print(f"[WATCHDOG] {stem}: encode_done — skip tar/render", flush=True)
                _try_upload_shard_after_encode_done(
                    config_path, cfg, stem, ent, remote_shards, hf_token
                )
                continue
            if st != "render_done":
                print(
                    f"[WATCHDOG] {stem}: encode-only skip (need render_done on disk; status={st!r})",
                    flush=True,
                )
                continue

            if os.path.isfile(zst) and os.path.getsize(zst) > 0:
                ent["download"] = "done"
            else:
                ent["download"] = "not_required"
            print(f"[WATCHDOG] {stem}: encode-only", flush=True)
            shard_render = os.path.join(render_dir, stem)
            passes = int(ent.get("encode_passes", 0))
            if max_encode_passes >= 0 and passes >= max_encode_passes:
                print(
                    f"[WATCHDOG] {stem}: skip encode (encode_passes={passes} >= max_encode_passes={max_encode_passes})",
                    flush=True,
                )
                ent["render_encode"] = "partial"
                ent["last_error"] = (
                    None if incomplete_as == "partial" else "max_encode_passes_reached"
                )
                continue
            try:
                if not os.path.isdir(shard_render):
                    raise FileNotFoundError(f"render_done but render dir missing: {shard_render}")
                run_encode_shard_multigpu(
                    os.path.abspath(config_path),
                    shard_render,
                    encode_gpus,
                    prior_encode_passes=int(ent.get("encode_passes", 0)),
                    encode_retry_policy=encode_retry_policy,
                )
                ent["encode_passes"] = int(ent.get("encode_passes", 0)) + 1
                classify = _classify_all(render_dir, stages, cfg, state)
                st = classify.get(stem)
                if st == "encode_done":
                    ent["render_encode"] = "done"
                    ent["last_error"] = None
                    ent["encode_passes"] = 0
                    _try_upload_shard_after_encode_done(
                        config_path, cfg, stem, ent, remote_shards, hf_token
                    )
                elif incomplete_as == "partial":
                    ent["render_encode"] = "partial"
                    ent["last_error"] = None
                    print(
                        f"[WATCHDOG] {stem}: encode still incomplete after pass — status={st!r}",
                        flush=True,
                    )
                else:
                    ent["render_encode"] = "error"
                    ent["last_error"] = "encode_incomplete_after_pass"
            except Exception as e:
                ent["render_encode"] = "error"
                ent["last_error"] = str(e)
                print(f"[WATCHDOG] ERROR {stem}: {e}", flush=True)
            continue

        # ── Full pipeline (render + encode + upload) ──
        if st == "encode_done":
            ent["encode_passes"] = 0
            ent["last_error"] = None
            ent.pop("skip_reason", None)

        # Tar is only required to decompress + render. encode_done / render_done use render_dir on disk.
        needs_tar = st not in ("encode_done", "render_done")

        if needs_tar:
            if not (os.path.isfile(zst) and os.path.getsize(zst) > 0):
                print(f"[WATCHDOG] {stem}: download .tar.zst (needed for decompress/render)", flush=True)
                if not download_one_shard_tar(cfg, stem):
                    ent["download"] = "pending"
                    ent["last_error"] = "download_failed"
                    continue
                if not (os.path.isfile(zst) and os.path.getsize(zst) > 0):
                    ent["download"] = "pending"
                    ent["last_error"] = "missing_local_tar_after_download"
                    continue
            ent["download"] = "done"
        else:
            if os.path.isfile(zst) and os.path.getsize(zst) > 0:
                ent["download"] = "done"
            else:
                ent["download"] = "not_required"
            if st == "encode_done":
                print(f"[WATCHDOG] {stem}: encode_done — skip tar download and render", flush=True)
            else:
                print(f"[WATCHDOG] {stem}: render_done — skip tar download; encode only", flush=True)

        if st == "encode_done":
            ent["render_encode"] = "done"
        elif st == "render_done":
            ent["render_encode"] = "in_progress"
        elif st == "partial":
            ent["render_encode"] = "in_progress"
        else:
            ent["render_encode"] = "pending"

        if st != "encode_done":
            passes = int(ent.get("encode_passes", 0))
            if max_encode_passes >= 0 and passes >= max_encode_passes:
                print(
                    f"[WATCHDOG] {stem}: skip encode (encode_passes={passes} >= max_encode_passes={max_encode_passes}); "
                    f"keeping existing outputs on disk",
                    flush=True,
                )
                ent["render_encode"] = "partial"
                ent["last_error"] = (
                    None
                    if incomplete_as == "partial"
                    else "max_encode_passes_reached"
                )
                continue

            print(f"[WATCHDOG] {stem}: render+encode (status was {st})", flush=True)
            try:
                shard_render = os.path.join(render_dir, stem)
                if st == "render_done":
                    if not os.path.isdir(shard_render):
                        raise FileNotFoundError(f"render_done but render dir missing: {shard_render}")
                else:
                    rg.render_shard_from_zst(
                        zst_path=zst,
                        render_dir=render_dir,
                        raw_dir=raw_dir,
                        gpu_ids=render_gpus,
                        num_workers=num_workers,
                        num_views=num_views,
                        render_tmp=render_tmp,
                        no_delete_shard=no_delete_shard,
                    )
                if st != "render_done":
                    if not os.path.isdir(shard_render):
                        print(
                            f"[WATCHDOG] {stem}: no render output (0 models or render failed); skip encode — "
                            f"check tarball or clear skip_reason in state to retry",
                            flush=True,
                        )
                        ent["skip_reason"] = "empty_tar_no_models"
                        ent["render_encode"] = "partial"
                        ent["last_error"] = (
                            None if incomplete_as == "partial" else "empty_tar_no_models"
                        )
                        continue
                    n_mesh = sum(
                        1
                        for d in os.listdir(shard_render)
                        if os.path.isdir(os.path.join(shard_render, d))
                        and os.path.isfile(os.path.join(shard_render, d, "mesh.ply"))
                    )
                    if n_mesh == 0:
                        print(
                            f"[WATCHDOG] {stem}: render dir has no mesh.ply; skip encode",
                            flush=True,
                        )
                        ent["skip_reason"] = "empty_tar_no_models"
                        ent["render_encode"] = "partial"
                        ent["last_error"] = (
                            None if incomplete_as == "partial" else "empty_tar_no_models"
                        )
                        continue
                run_encode_shard_multigpu(
                    os.path.abspath(config_path),
                    shard_render,
                    encode_gpus,
                    prior_encode_passes=int(ent.get("encode_passes", 0)),
                    encode_retry_policy=encode_retry_policy,
                )
                ent["encode_passes"] = int(ent.get("encode_passes", 0)) + 1
                classify = _classify_all(render_dir, stages, cfg, state)
                st = classify.get(stem)
                if st == "encode_done":
                    ent["render_encode"] = "done"
                    ent["last_error"] = None
                    ent["encode_passes"] = 0
                elif incomplete_as == "partial":
                    ent["render_encode"] = "partial"
                    ent["last_error"] = None
                    print(
                        f"[WATCHDOG] {stem}: encode still incomplete after pass — "
                        f"treating as partial (disk outputs kept; no re-download). "
                        f"status={st!r}",
                        flush=True,
                    )
                else:
                    ent["render_encode"] = "error"
                    ent["last_error"] = "encode_incomplete_after_pass"
            except Exception as e:
                ent["render_encode"] = "error"
                ent["last_error"] = str(e)
                print(f"[WATCHDOG] ERROR {stem}: {e}", flush=True)
                continue

        if classify.get(stem) != "encode_done":
            continue

        ent["render_encode"] = "done"
        _try_upload_shard_after_encode_done(
            config_path, cfg, stem, ent, remote_shards, hf_token
        )


def main() -> None:
    global STOP
    parser = argparse.ArgumentParser(description="Pipeline watchdog (download → render → encode → upload loop)")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit")
    wg = parser.add_mutually_exclusive_group()
    wg.add_argument(
        "--render-only",
        action="store_true",
        help="Only download + Blender render; mark render_only_done; optional delete .tar.zst (watchdog YAML)",
    )
    wg.add_argument(
        "--encode-only",
        action="store_true",
        help="Only encode (and optional upload); requires render_done on disk (e.g. after --render-only)",
    )
    args = parser.parse_args()

    if args.render_only:
        watchdog_mode = "render_only"
    elif args.encode_only:
        watchdog_mode = "encode_only"
    else:
        watchdog_mode = "full"

    os.environ.setdefault("SPCONV_ALGO", "native")

    def _sig(_s, _f):
        global STOP
        STOP = True
        print("[WATCHDOG] stop requested", flush=True)

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    from encoders.config import load_config

    config_path = os.path.abspath(os.path.expanduser(args.config))
    cfg = load_config(config_path)
    validate_config(cfg, config_path, watchdog_mode=watchdog_mode)

    state_path = _state_path(cfg)
    wd = cfg.get("pipeline", {}).get("watchdog", {}) or {}
    interval = float(wd.get("interval_seconds", 600))

    print(f"[WATCHDOG] config={config_path}", flush=True)
    print(f"[WATCHDOG] state={state_path}", flush=True)
    print(f"[WATCHDOG] mode={watchdog_mode} interval={interval}s once={args.once}", flush=True)

    while not STOP:
        state = load_state(state_path)
        state["cycle"] = int(state.get("cycle", 0)) + 1
        print(f"\n{'='*60}\n[WATCHDOG] cycle {state['cycle']} {_utc_now()}\n{'='*60}", flush=True)

        try:
            validate_config(cfg, config_path, watchdog_mode=watchdog_mode)
        except SystemExit as e:
            print(f"[WATCHDOG] config validation failed: {e}", flush=True)
            if args.once:
                raise
            time.sleep(interval)
            continue

        wd_opts = cfg.get("pipeline", {}).get("watchdog", {}) or {}
        if watchdog_mode != "encode_only" and wd_opts.get("prefetch_downloads", False):
            run_download_phase(cfg)

        hf_up = (cfg.get("hf") or {}).get("upload") or {}
        remote: set[str] | None = None
        hf_tok = hf_hub_token(cfg)
        if watchdog_mode != "render_only" and hf_up.get("enabled", False):
            try:
                remote = _remote_tar_set(
                    hf_up["repo_id"],
                    hf_up.get("repo_type", "dataset"),
                    hf_up.get("path_in_repo", "github/render"),
                    token=hf_tok,
                )
                print(f"[WATCHDOG] Hub has {len(remote)} shard archive(s) under {hf_up.get('path_in_repo')}", flush=True)
            except Exception as e:
                print(f"[WATCHDOG] WARN: could not list remote: {e}", flush=True)
                remote = set()

        try:
            if watchdog_mode == "full" and bool(wd_opts.get("overlap_render_encode", False)):
                one_cycle_full_overlap(config_path, cfg, state, remote, hf_token=hf_tok)
            else:
                one_cycle(config_path, cfg, state, remote, hf_token=hf_tok, mode=watchdog_mode)
        except Exception as e:
            print(f"[WATCHDOG] cycle error: {e}", flush=True)
            import traceback

            traceback.print_exc()

        save_state(state_path, state)

        done_dl = sum(
            1 for s in state["shards"].values() if s.get("download") in ("done", "not_required")
        )
        done_re = sum(1 for s in state["shards"].values() if s.get("render_encode") == "done")
        done_ro = sum(
            1 for s in state["shards"].values() if s.get("render_encode") == "render_only_done"
        )
        done_up = sum(1 for s in state["shards"].values() if s.get("upload") in ("done", "skipped"))
        if watchdog_mode == "render_only":
            print(
                f"[WATCHDOG] summary (render-only): download_ok={done_dl} "
                f"render_only_done={done_ro} encode_done={done_re} "
                f"upload_skipped_or_done={done_up} / {len(state['shards'])}",
                flush=True,
            )
        elif watchdog_mode == "encode_only":
            print(
                f"[WATCHDOG] summary (encode-only): download_ok={done_dl} encode_done={done_re} "
                f"upload_ok_or_skip={done_up} / {len(state['shards'])}",
                flush=True,
            )
        else:
            print(
                f"[WATCHDOG] summary shards: download_ok={done_dl} encode_ok={done_re} "
                f"upload_ok_or_skip={done_up} / {len(state['shards'])}",
                flush=True,
            )

        if args.once:
            break
        if STOP:
            break
        print(f"[WATCHDOG] sleep {interval}s ...", flush=True)
        t0 = time.time()
        while time.time() - t0 < interval and not STOP:
            time.sleep(min(30.0, interval))


if __name__ == "__main__":
    main()

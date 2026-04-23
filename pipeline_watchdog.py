#!/usr/bin/env python3
"""
Watchdog entrypoint: validate config → download (range) → render → encode → upload, in a loop, with JSON state.

Uses the same YAML as run_pipeline / upload_hf_encoded_shards (paths, hf.download, hf.upload, gpus, stages).

State file (default: <data_root>/github/pipeline_state.json):
  - Tracks per-shard download / render_encode / upload and last_error
  - Reconciles with disk + Hub when possible

Usage:
  export HUGGINGFACE_HUB_TOKEN=...   # if upload enabled
  export SPCONV_ALGO=native
  python pipeline_watchdog.py --config config/default.yaml

  python pipeline_watchdog.py --config config/default.yaml --once   # single cycle then exit
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any

RENDER_PKG = os.path.dirname(os.path.abspath(__file__))
if RENDER_PKG not in sys.path:
    sys.path.insert(0, RENDER_PKG)

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
    }


def validate_config(cfg: dict, config_path: str) -> None:
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
    if cfg.get("stages", {}).get("render", True):
        b = paths.get("blender_bin")
        if not b or not os.path.isfile(b):
            raise SystemExit(f"Blender not found at paths.blender_bin: {b}")
    up = (cfg.get("hf") or {}).get("upload") or {}
    if up.get("enabled", False):
        if not (os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")):
            raise SystemExit("hf.upload.enabled but HUGGINGFACE_HUB_TOKEN / HF_TOKEN not set")
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


def run_encode_shard_subprocess(config_path: str, shard_render_dir: str, cuda_visible: str) -> None:
    code = (
        "import os,sys;"
        f"os.environ['CUDA_VISIBLE_DEVICES']={cuda_visible!r};"
        "os.environ.setdefault('SPCONV_ALGO','native');"
        f"sys.path.insert(0,{RENDER_PKG!r});"
        "from encoders.config import load_config;"
        "from encoders import encode_shard, ALL_STAGES;"
        f"cfg=load_config({config_path!r});"
        f"st=[s for s in ALL_STAGES if cfg.get('stages',{{}}).get(s,True)];"
        f"encode_shard({shard_render_dir!r}, cfg, st)"
    )
    subprocess.check_call([sys.executable, "-c", code])


def run_upload_subprocess(config_path: str, shard_stem: str) -> None:
    script = os.path.join(RENDER_PKG, "upload_hf_encoded_shards.py")
    subprocess.check_call(
        [sys.executable, script, "--config", config_path, "--shards", shard_stem],
        env=os.environ.copy(),
    )


def _classify_all(render_dir: str, stages: list[str], cfg: dict) -> dict[str, str]:
    from run_pipeline import _classify_shards

    return _classify_shards(render_dir, stages, cfg)


def _remote_tar_set(repo_id: str, repo_type: str, path_in_repo: str) -> set[str]:
    from huggingface_hub import HfApi

    prefix = path_in_repo.strip("/") + "/"
    names: set[str] = set()
    for p in HfApi().list_repo_files(repo_id, repo_type=repo_type):
        if p.startswith(prefix) and p.endswith(".tar.zst"):
            names.add(os.path.basename(p).replace(".tar.zst", ""))
    return names


def one_cycle(config_path: str, cfg: dict, state: dict[str, Any], remote_shards: set[str] | None) -> None:
    paths = cfg["paths"]
    shard_dir = paths["shard_dir"]
    render_dir = paths["render_dir"]
    raw_dir = paths["raw_dir"]
    render_tmp = paths.get("render_tmp")
    hf_up = (cfg.get("hf") or {}).get("upload") or {}
    upload_enabled = bool(hf_up.get("enabled", False))
    path_in_repo = (hf_up.get("path_in_repo") or "github/render").strip("/")

    from encoders import ALL_STAGES

    stages = [s for s in ALL_STAGES if cfg.get("stages", {}).get(s, True)]

    stems = expected_stems(cfg)
    classify = _classify_all(render_dir, stages, cfg)

    render_gpus = cfg["gpus"]["render"]
    encode_gpus = cfg["gpus"]["encode"]
    num_workers = int(cfg.get("render", {}).get("num_workers", 4))
    num_views = int(cfg.get("render", {}).get("num_views", 40))
    no_delete_shard = bool(cfg.get("pipeline", {}).get("no_delete_shard", True))

    import render_github as rg
    from huggingface_hub import HfApi
    from upload_hf_encoded_shards import _remote_size, delete_source_shard_tar_if_enabled

    hf_api = HfApi()

    for stem in stems:
        if STOP:
            break
        zst = os.path.join(shard_dir, stem + ".tar.zst")
        ent = state["shards"].setdefault(stem, _shard_entry())

        if os.path.isfile(zst) and os.path.getsize(zst) > 0:
            ent["download"] = "done"
        else:
            ent["download"] = "pending"
            ent["last_error"] = "missing_local_tar"
            continue

        st = classify.get(stem)
        if st == "encode_done":
            ent["render_encode"] = "done"
        elif st == "render_done":
            ent["render_encode"] = "in_progress"
        elif st == "partial":
            ent["render_encode"] = "in_progress"
        else:
            ent["render_encode"] = "pending"

        if st != "encode_done":
            print(f"[WATCHDOG] {stem}: render+encode (status was {st})", flush=True)
            try:
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
                shard_render = os.path.join(render_dir, stem)
                enc_gpu = str(encode_gpus[0]) if encode_gpus else "0"
                run_encode_shard_subprocess(os.path.abspath(config_path), shard_render, enc_gpu)
                classify = _classify_all(render_dir, stages, cfg)
                st = classify.get(stem)
                if st == "encode_done":
                    ent["render_encode"] = "done"
                    ent["last_error"] = None
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

        if not upload_enabled:
            ent["upload"] = "skipped"
            delete_source_shard_tar_if_enabled(cfg, stem)
            continue

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
            continue

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


def main() -> None:
    global STOP
    parser = argparse.ArgumentParser(description="Pipeline watchdog (download → render → encode → upload loop)")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit")
    args = parser.parse_args()

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
    validate_config(cfg, config_path)

    state_path = _state_path(cfg)
    wd = cfg.get("pipeline", {}).get("watchdog", {}) or {}
    interval = float(wd.get("interval_seconds", 600))

    print(f"[WATCHDOG] config={config_path}", flush=True)
    print(f"[WATCHDOG] state={state_path}", flush=True)
    print(f"[WATCHDOG] interval={interval}s once={args.once}", flush=True)

    while not STOP:
        state = load_state(state_path)
        state["cycle"] = int(state.get("cycle", 0)) + 1
        print(f"\n{'='*60}\n[WATCHDOG] cycle {state['cycle']} {_utc_now()}\n{'='*60}", flush=True)

        try:
            validate_config(cfg, config_path)
        except SystemExit as e:
            print(f"[WATCHDOG] config validation failed: {e}", flush=True)
            if args.once:
                raise
            time.sleep(interval)
            continue

        run_download_phase(cfg)

        hf_up = (cfg.get("hf") or {}).get("upload") or {}
        remote: set[str] | None = None
        if hf_up.get("enabled", False):
            try:
                remote = _remote_tar_set(
                    hf_up["repo_id"],
                    hf_up.get("repo_type", "dataset"),
                    hf_up.get("path_in_repo", "github/render"),
                )
                print(f"[WATCHDOG] Hub has {len(remote)} shard archive(s) under {hf_up.get('path_in_repo')}", flush=True)
            except Exception as e:
                print(f"[WATCHDOG] WARN: could not list remote: {e}", flush=True)
                remote = set()

        try:
            one_cycle(config_path, cfg, state, remote)
        except Exception as e:
            print(f"[WATCHDOG] cycle error: {e}", flush=True)
            import traceback

            traceback.print_exc()

        save_state(state_path, state)

        done_dl = sum(1 for s in state["shards"].values() if s.get("download") == "done")
        done_re = sum(1 for s in state["shards"].values() if s.get("render_encode") == "done")
        done_up = sum(1 for s in state["shards"].values() if s.get("upload") in ("done", "skipped"))
        print(
            f"[WATCHDOG] summary shards: download_ok={done_dl} encode_ok={done_re} upload_ok_or_skip={done_up} / {len(state['shards'])}",
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

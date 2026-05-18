#!/usr/bin/env python3
"""Upload encoded shards one at a time with timeout/retry protection.

This is a thin watchdog around upload_hf_encoded_shards.py. It keeps the
single-shard upload behavior isolated so one large or stalled shard does not
block the whole batch forever.
"""
from __future__ import annotations

import argparse
import glob
import os
import signal
import subprocess
import sys
import time

RENDER_PKG = os.path.dirname(os.path.abspath(__file__))
if RENDER_PKG not in sys.path:
    sys.path.insert(0, RENDER_PKG)


def _remote_completed_stems(cfg: dict) -> set[str]:
    from encoders.config import hf_hub_token
    from huggingface_hub import HfApi
    from upload_hf_encoded_shards import _plan_archives

    up = (cfg.get("hf") or {}).get("upload") or {}
    prefix = (up.get("path_in_repo") or "github/render").strip("/") + "/"
    repo_id = up.get("repo_id") or ""
    if not repo_id:
        return set()
    files = HfApi(token=hf_hub_token(cfg)).list_repo_files(
        repo_id,
        repo_type=up.get("repo_type", "dataset"),
        revision=up.get("revision") or "main",
    )
    remote_names = {
        os.path.basename(p)
        for p in files
        if p.startswith(prefix) and p.endswith(".tar.zst")
    }

    completed: set[str] = set()
    render_dir = cfg["paths"]["render_dir"]
    arch = up.get("archive") or {}
    includes: list[str] = list(arch.get("include") or ["latents", "transforms.json", "mesh.ply"])
    exclude_globs: list[str] = list(arch.get("exclude_globs") or [])
    max_part_bytes = int(arch.get("max_part_bytes") or 0)
    for single in remote_names:
        if ".part_" not in single:
            completed.add(single[: -len(".tar.zst")])

    for shard_path in glob.glob(os.path.join(render_dir, "shard_*")):
        if not os.path.isdir(shard_path):
            continue
        stem = os.path.basename(shard_path)
        if stem in completed:
            continue
        plans = _plan_archives(stem, shard_path, includes, exclude_globs, max_part_bytes)
        if plans and all(archive_name in remote_names for archive_name, _ in plans):
            completed.add(stem)
    return completed


def _encode_done_stems(cfg: dict) -> list[str]:
    from encoders import ALL_STAGES
    from run_pipeline import _classify_shards

    stages = [s for s in ALL_STAGES if cfg.get("stages", {}).get(s, True)]
    status = _classify_shards(cfg["paths"]["render_dir"], stages, cfg)
    return sorted(k for k, v in status.items() if v == "encode_done")


def _select_stems(args: argparse.Namespace, cfg: dict) -> list[str]:
    if args.shards:
        return [s.strip() for s in args.shards.split(",") if s.strip()]
    if args.all_verified:
        stems = _encode_done_stems(cfg)
        if args.skip_remote:
            remote = _remote_completed_stems(cfg)
            stems = [s for s in stems if s not in remote]
        return stems
    raise SystemExit("Pass --shards or --all-verified.")


def _terminate_process_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def _run_one(args: argparse.Namespace, stem: str) -> int:
    cmd = [
        sys.executable,
        os.path.join(RENDER_PKG, "upload_hf_encoded_shards.py"),
        "--config",
        os.path.abspath(args.config),
        "--shards",
        stem,
    ]
    if args.force:
        cmd.append("--force")
    if args.keep_local:
        cmd.append("--keep-local")
    if args.upload_only:
        cmd.append("--upload-only")
    if args.no_skip_remote:
        cmd.append("--no-skip-remote")

    timeout_s = args.timeout_minutes * 60.0
    print(
        f"[UPLOAD_WATCHDOG] {stem}: start timeout={args.timeout_minutes:.1f}m",
        flush=True,
    )
    proc = subprocess.Popen(cmd, cwd=RENDER_PKG, preexec_fn=os.setsid)
    try:
        return proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f"[UPLOAD_WATCHDOG] {stem}: timeout; terminating upload", flush=True)
        _terminate_process_group(proc)
        return 124


def main() -> None:
    parser = argparse.ArgumentParser(description="Retry shard uploads one shard at a time")
    parser.add_argument("--config", required=True)
    parser.add_argument("--shards", default=None, help="Comma-separated shard names")
    parser.add_argument("--all-verified", action="store_true")
    parser.add_argument("--skip-remote", action="store_true", help="With --all-verified, skip stems already on Hub")
    parser.add_argument("--force", action="store_true", help="Pass --force to upload_hf_encoded_shards.py")
    parser.add_argument("--upload-only", action="store_true", help="Reuse existing pack_for_upload archives")
    parser.add_argument("--keep-local", action="store_true")
    parser.add_argument("--no-skip-remote", action="store_true")
    parser.add_argument("--timeout-minutes", type=float, default=90.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=60.0)
    args = parser.parse_args()

    from encoders.config import load_config

    cfg = load_config(args.config)
    stems = _select_stems(args, cfg)
    if not stems:
        print("[UPLOAD_WATCHDOG] no shards to upload", flush=True)
        return
    print(f"[UPLOAD_WATCHDOG] selected {len(stems)} shard(s)", flush=True)

    failures: list[str] = []
    for i, stem in enumerate(stems, 1):
        ok = False
        for attempt in range(1, args.retries + 1):
            print(
                f"[UPLOAD_WATCHDOG] [{i}/{len(stems)}] {stem}: attempt {attempt}/{args.retries}",
                flush=True,
            )
            rc = _run_one(args, stem)
            if rc == 0:
                ok = True
                break
            print(f"[UPLOAD_WATCHDOG] {stem}: failed rc={rc}", flush=True)
            if attempt < args.retries:
                time.sleep(args.sleep_seconds)
        if not ok:
            failures.append(stem)

    if failures:
        print(f"[UPLOAD_WATCHDOG] failed shards: {','.join(failures)}", flush=True)
        raise SystemExit(1)
    print("[UPLOAD_WATCHDOG] all selected shards uploaded", flush=True)


if __name__ == "__main__":
    main()

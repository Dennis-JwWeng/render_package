#!/usr/bin/env python3
"""
Refresh upload_record.json from YAML + Hugging Face listing + local encode_done.

Uses hf.upload (repo_id, path_in_repo, repo_type, revision) and hf.download shard
range (expected stems). Writes default:
  <paths.data_root>/github/upload_record.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

RENDER_PKG = os.path.dirname(os.path.abspath(__file__))
if RENDER_PKG not in sys.path:
    sys.path.insert(0, RENDER_PKG)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _remote_stems_from_listing(paths_in_repo: list[str], path_prefix: str) -> set[str]:
    """paths_in_repo: full paths from list_repo_files; prefix e.g. github/render."""
    prefix = path_prefix.strip("/").replace("\\", "/") + "/"
    out: set[str] = set()
    for p in paths_in_repo:
        p = p.replace("\\", "/")
        if not p.startswith(prefix) or not p.endswith(".tar.zst"):
            continue
        base = os.path.basename(p)
        out.add(base[: -len(".tar.zst")])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Write upload_record.json from config + Hub + disk")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="JSON path (default: <data_root>/github/upload_record.json)",
    )
    args = parser.parse_args()

    from encoders import ALL_STAGES
    from encoders.config import hf_hub_token, load_config
    from huggingface_hub import HfApi
    from pipeline_watchdog import expected_stems
    from run_pipeline import _classify_shards

    cfg_path = os.path.abspath(os.path.expanduser(args.config))
    cfg = load_config(cfg_path)
    paths = cfg["paths"]
    data_root = paths["data_root"]
    render_dir = paths["render_dir"]
    out_path = args.output.strip() or os.path.join(data_root, "github", "upload_record.json")

    hf_up = (cfg.get("hf") or {}).get("upload") or {}
    repo_id = hf_up.get("repo_id") or ""
    path_in_repo = (hf_up.get("path_in_repo") or "github/render").strip("/").replace("\\", "/")
    repo_type = hf_up.get("repo_type", "dataset")
    revision = hf_up.get("revision") or "main"

    stems_expected = set(expected_stems(cfg))
    stages = [s for s in ALL_STAGES if cfg.get("stages", {}).get(s, True)]
    classify = _classify_shards(render_dir, stages, cfg)
    encode_done_all = sorted(k for k, v in classify.items() if v == "encode_done")
    encode_done_in_range = sorted(set(encode_done_all) & stems_expected)

    hub_listing_ok = False
    hub_error: str | None = None
    remote_stems: set[str] = set()
    token = hf_hub_token(cfg)

    if not repo_id:
        hub_error = "hf.upload.repo_id empty in config"
    else:
        try:
            api = HfApi(token=token)
            remote_paths = api.list_repo_files(
                repo_id,
                repo_type=repo_type,
                revision=revision,
            )
            remote_stems = _remote_stems_from_listing(remote_paths, path_in_repo)
            hub_listing_ok = True
        except Exception as e:
            hub_error = str(e)

    on_hub_done = sorted(set(encode_done_in_range) & remote_stems)
    pending = sorted(set(encode_done_in_range) - remote_stems)

    record = {
        "updated_utc": _utc_now(),
        "config": cfg_path,
        "hf_upload": {
            "repo_id": repo_id,
            "repo_type": repo_type,
            "revision": revision,
            "path_in_repo": path_in_repo,
            "enabled_in_yaml": bool(hf_up.get("enabled", False)),
        },
        "expected_stems_in_config": len(stems_expected),
        "encode_done_in_range": len(encode_done_in_range),
        "encode_done_stems": encode_done_in_range,
        "hub_listing_ok": hub_listing_ok,
        "hub_error": hub_error,
        "remote_shard_archives_under_prefix": len(remote_stems),
        "encode_done_on_hub": len(on_hub_done),
        "encode_done_on_hub_stems": on_hub_done,
        "encode_done_pending_upload": pending,
        "pending_upload_count": len(pending),
        "notes": {
            "upload_command": (
                f"cd {RENDER_PKG} && python upload_hf_encoded_shards.py "
                f"--config {cfg_path} --all-verified --force"
            ),
            "after_upload": "Re-run this script to refresh this file; or run upload then refresh.",
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"[refresh_upload_record] wrote {out_path}", flush=True)
    print(f"  encode_done (in shard range): {len(encode_done_in_range)}", flush=True)
    print(f"  on Hub (same stems):          {len(on_hub_done)}", flush=True)
    print(f"  pending upload:               {len(pending)}", flush=True)
    if pending:
        print(f"  pending stems: {', '.join(pending[:12])}{' ...' if len(pending) > 12 else ''}", flush=True)
    if hub_error:
        print(f"  hub_error: {hub_error}", flush=True)
    if pending:
        print(
            f"\n  Next: {record['notes']['upload_command']}",
            flush=True,
        )


if __name__ == "__main__":
    main()

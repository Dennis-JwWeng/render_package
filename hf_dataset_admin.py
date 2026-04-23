#!/usr/bin/env python3
"""
Hugging Face dataset utilities: inspect uploads, rename (move) a repo.

Auth: HUGGINGFACE_HUB_TOKEN / HF_TOKEN, or ``stats --config`` to use hf.upload.token from ``<config>.local.yaml`` (via encoders.config).

Examples:
  # Count / size of .tar.zst under a path prefix (default: whole tree)
  python hf_dataset_admin.py stats Dennis0626/trellis500k-github-rendered --repo-type dataset

  # Only under github/render (matches existing layout on that dataset)
  python hf_dataset_admin.py stats Dennis0626/trellis500k-github-rendered \\
      --path-prefix github/render --repo-type dataset

  # Rename dataset (same namespace: rendered -> processed). Requires token + --yes
  python hf_dataset_admin.py rename \\
      Dennis0626/trellis500k-github-processed Dennis0626/trellis500k-github-archives-6-processed \\
      --repo-type dataset --yes
"""
from __future__ import annotations

import argparse
import os
import sys


def _api_token() -> str | None:
    return os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")


def cmd_stats(args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi

    token = _api_token()
    if getattr(args, "config", None):
        RENDER_PKG = os.path.dirname(os.path.abspath(__file__))
        if RENDER_PKG not in sys.path:
            sys.path.insert(0, RENDER_PKG)
        from encoders.config import hf_hub_token, load_config

        token = hf_hub_token(load_config(args.config))

    api = HfApi(token=token)
    prefix = (args.path_prefix or "").strip().strip("/")
    tree_path = prefix if prefix else None

    files = list(
        api.list_repo_tree(
            repo_id=args.repo_id,
            path_in_repo=tree_path,
            recursive=True,
            expand=True,
            repo_type=args.repo_type,
            revision=args.revision,
        )
    )

    tars = [
        f
        for f in files
        if hasattr(f, "path") and f.path.endswith(".tar.zst") and hasattr(f, "size")
    ]
    other = len(files) - len(tars)
    total_bytes = sum(f.size for f in tars)

    print(f"repo:        {args.repo_id}  ({args.repo_type})  @{args.revision}")
    if prefix:
        print(f"path_prefix: {prefix}/")
    print(f".tar.zst:    {len(tars)} files  ({total_bytes / (1024**3):.3f} GiB)")
    print(f"other_nodes: {other} (folders + non-tar files in tree)")
    if tars:
        names = sorted(f.path for f in tars)
        print("archives:")
        for p in names:
            print(f"  {p}")
    elif not files:
        print("(empty tree or path missing — check path_prefix / revision / access)")


def cmd_rename(args: argparse.Namespace) -> None:
    token = _api_token()
    if not token:
        print("Set HUGGINGFACE_HUB_TOKEN in the environment to rename.", file=sys.stderr)
        sys.exit(1)
    if not args.yes:
        print("Refusing to rename without --yes (this affects all Hub URLs and clones).", file=sys.stderr)
        sys.exit(1)

    from huggingface_hub import HfApi

    HfApi().move_repo(
        args.from_id,
        args.to_id,
        repo_type=args.repo_type,
        token=token,
    )
    print(f"Moved {args.from_id} -> {args.to_id}")
    print("Update hf.upload.repo_id in your YAML to the new id.")


def main() -> None:
    p = argparse.ArgumentParser(description="HF dataset stats / rename")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("stats", help="List .tar.zst under repo (optionally under path prefix)")
    s.add_argument("repo_id", type=str)
    s.add_argument("--repo-type", type=str, default="dataset")
    s.add_argument("--revision", type=str, default="main")
    s.add_argument(
        "--path-prefix",
        type=str,
        default="",
        help="e.g. github/render — only list this subtree",
    )
    s.add_argument(
        "--config",
        type=str,
        default="",
        help="YAML (with optional .local.yaml): HF token from hf.upload.token / env via encoders.config",
    )
    s.set_defaults(func=cmd_stats)

    r = sub.add_parser("rename", help="Rename/move repo (HfApi.move_repo)")
    r.add_argument("from_id", type=str, help="namespace/old-name")
    r.add_argument("to_id", type=str, help="namespace/new-name")
    r.add_argument("--repo-type", type=str, default="dataset")
    r.add_argument(
        "--yes",
        action="store_true",
        help="Confirm rename (required)",
    )
    r.set_defaults(func=cmd_rename)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

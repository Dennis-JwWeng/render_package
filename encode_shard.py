#!/usr/bin/env python3
"""
Encode all rendered objects in one shard directory.

Usage:
    python encode_shard.py --config config/default.yaml \
        --shard_dir /path/to/render/shard_016725

    python encode_shard.py --config config/default.yaml \
        --shard_dir /path/to/render/shard_016725 \
        --stages dino_features,slat,ss
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("SPCONV_ALGO", "native")

RENDER_PKG = os.path.dirname(os.path.abspath(__file__))
if RENDER_PKG not in sys.path:
    sys.path.insert(0, RENDER_PKG)


def main():
    parser = argparse.ArgumentParser(description="Encode latents for one shard")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--shard_dir", type=str, required=True,
                        help="Path to rendered shard (contains object subdirs with mesh.ply)")
    parser.add_argument("--stages", type=str, default=None,
                        help="Comma-separated stages: dino_features,unilat,slat,ss (default: all enabled in config)")
    args = parser.parse_args()

    from encoders.config import load_config
    cfg = load_config(args.config)

    stages = None
    if args.stages:
        stages = [s.strip() for s in args.stages.split(",")]

    from encoders import encode_shard
    encode_shard(args.shard_dir, cfg, stages)


if __name__ == "__main__":
    main()

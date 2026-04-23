"""
YAML config loader and path resolver for the encode pipeline.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

RENDER_PKG = Path(__file__).resolve().parent.parent


def _resolve(p: str | None, base: Path = RENDER_PKG) -> str | None:
    if p is None:
        return None
    p = os.path.expanduser(p)
    if os.path.isabs(p):
        return p
    return str((base / p).resolve())


def _detect_num_gpus() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-L"], text=True, stderr=subprocess.DEVNULL
        )
        return max(1, sum(1 for ln in out.splitlines() if ln.strip().startswith("GPU ")))
    except (OSError, subprocess.CalledProcessError, FileNotFoundError):
        return 1


def load_config(yaml_path: str | Path) -> dict[str, Any]:
    """Load YAML config, resolve all paths, fill defaults."""
    yaml_path = Path(yaml_path).resolve()
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    paths = cfg.setdefault("paths", {})
    data_root = _resolve(paths.get("data_root", "."))
    paths["data_root"] = data_root

    defaults = {
        "shard_dir": os.path.join(data_root, "trellis500k-github-archives-5/shards/github"),
        "render_dir": os.path.join(data_root, "github/render"),
        "raw_dir": os.path.join(data_root, "github/raw"),
        "render_tmp": os.path.join(data_root, "github/.render_tmp"),
    }
    for k, v in defaults.items():
        if not paths.get(k):
            paths[k] = v
        else:
            paths[k] = _resolve(paths[k])

    blender = paths.get("blender_bin", "auto")
    if blender == "auto" or not blender:
        paths["blender_bin"] = str(RENDER_PKG / "blender-3.5.1-linux-x64" / "blender")
    else:
        paths["blender_bin"] = _resolve(blender)

    weights = cfg.setdefault("weights", {})
    for k in ("unilat_encoder", "slat_encoder", "ss_encoder", "dinov2", "dinov2_repo"):
        v = weights.get(k)
        if v and v != "torch_hub":
            weights[k] = _resolve(v)

    tp = cfg.setdefault("third_party", {})
    for k in ("trellis", "unilat3d"):
        v = tp.get(k)
        if v:
            tp[k] = _resolve(v)

    render = cfg.setdefault("render", {})

    # Resolve GPU lists
    n_gpus = _detect_num_gpus()
    all_gpu_ids = list(range(n_gpus))
    gpus = cfg.setdefault("gpus", {})
    for key in ("render", "encode"):
        val = gpus.get(key, "auto")
        if val == "auto" or val is None:
            gpus[key] = all_gpu_ids
        elif isinstance(val, list):
            gpus[key] = [int(x) for x in val]
        else:
            gpus[key] = all_gpu_ids

    # Back-compat: render.num_gpus derived from gpus.render
    render["num_gpus"] = len(gpus["render"])

    cfg.setdefault("stages", {})
    cfg.setdefault("encode", {})
    cfg.setdefault("pipeline", {})

    return cfg

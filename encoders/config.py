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


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def hf_hub_token(cfg: dict[str, Any]) -> str | None:
    """HF auth: env HUGGINGFACE_HUB_TOKEN / HF_TOKEN wins; else hf.upload.token from YAML."""
    t = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if t and str(t).strip():
        return str(t).strip()
    up = (cfg.get("hf") or {}).get("upload") or {}
    tok = up.get("token")
    if isinstance(tok, str) and tok.strip():
        return tok.strip()
    return None


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
    """Load YAML config, resolve all paths, fill defaults.

    If ``<name>.local.yaml`` exists beside ``<name>.yaml``, it is deep-merged
    (for secrets: hf.upload.token). Keep *.local.yaml out of version control.

    Sets ``TMPDIR`` / ``TEMP`` / ``TMP`` to ``paths.render_tmp`` (data mount) when
    those env vars are unset, so ``tempfile`` and native libs avoid system ``/tmp``.
    """
    yaml_path = Path(yaml_path).resolve()
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f) or {}
    local_path = yaml_path.parent / f"{yaml_path.stem}.local.yaml"
    if local_path.is_file():
        with open(local_path) as f:
            local = yaml.safe_load(f) or {}
        if isinstance(local, dict):
            _deep_merge(cfg, local)

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

    def _resolve_gpu_val(val: Any) -> list[int]:
        if val == "auto" or val is None:
            return list(all_gpu_ids)
        if isinstance(val, list):
            return [int(x) for x in val]
        return list(all_gpu_ids)

    pool = gpus.get("pool")
    if pool is not None:
        # One list for both stages: watchdog runs render then encode on the same IDs (no 2+2 split needed).
        resolved = _resolve_gpu_val(pool)
        gpus["render"] = resolved
        gpus["encode"] = list(resolved)
    else:
        for key in ("render", "encode"):
            gpus[key] = _resolve_gpu_val(gpus.get(key, "auto"))

    # Back-compat: render.num_gpus derived from gpus.render
    render["num_gpus"] = len(gpus["render"])

    cfg.setdefault("stages", {})
    cfg.setdefault("encode", {})
    cfg.setdefault("pipeline", {})

    _ensure_process_tempdir_on_data_mount(cfg)
    return cfg


def _ensure_process_tempdir_on_data_mount(cfg: dict[str, Any]) -> str | None:
    """If TMPDIR/TEMP/TMP are unset, point them at paths.render_tmp (data mount).

    Keeps tempfile and libraries off system /tmp when it is full or unsuitable.
    """
    paths = cfg.get("paths") or {}
    parent = paths.get("render_tmp")
    if not parent:
        dr = paths.get("data_root") or "."
        parent = os.path.join(str(dr), "github", ".pipeline_tmp")
    parent = os.path.abspath(os.path.expanduser(str(parent)))
    try:
        os.makedirs(parent, exist_ok=True)
    except OSError:
        return None
    for key in ("TMPDIR", "TEMP", "TMP"):
        os.environ.setdefault(key, parent)
    return parent

#!/usr/bin/env python3
"""
Verify that render_package is correctly installed and all dependencies are met.

Usage:
    python verify_install.py [--config config/default.yaml]
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys

RENDER_PKG = os.path.dirname(os.path.abspath(__file__))
if RENDER_PKG not in sys.path:
    sys.path.insert(0, RENDER_PKG)

REQUIRED_PACKAGES = [
    ("torch", "2.5.0"),
    ("torchvision", "0.20.0"),
    ("numpy", "1.24"),
    ("scipy", "1.10"),
    ("PIL", "10.0"),
    ("tqdm", "4.60"),
    ("yaml", "6.0"),
    ("safetensors", "0.4"),
    ("plyfile", "1.0"),
    ("open3d", "0.18"),
]

OPTIONAL_PACKAGES = [
    ("flash_attn", "2.0", "UniLat attention acceleration"),
    ("xformers", "0.0.28", "DINOv2 efficient attention"),
    ("spconv", "2.3", "Sparse convolutions (TRELLIS + UniLat)"),
    ("kaolin", "0.18", "NVIDIA Kaolin (optional)"),
]


def check_package(name: str, min_ver: str | None = None) -> tuple[bool, str]:
    try:
        m = importlib.import_module(name)
        ver = getattr(m, "__version__", "unknown")
        return True, ver
    except ImportError:
        return False, "NOT INSTALLED"


def check_file(path: str) -> bool:
    return os.path.isfile(path)


def check_dir(path: str) -> bool:
    return os.path.isdir(path)


def main():
    parser = argparse.ArgumentParser(description="Verify render_package installation")
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    errors = 0
    warnings = 0

    print("=" * 60)
    print("render_package Installation Verification")
    print("=" * 60)

    # 1. Python packages
    print("\n[1] Python Packages (required)")
    for name, min_ver in REQUIRED_PACKAGES:
        ok, ver = check_package(name)
        status = f"OK ({ver})" if ok else "MISSING"
        marker = "  " if ok else "!!"
        print(f"  {marker} {name:20s} {status}")
        if not ok:
            errors += 1

    print("\n[2] Python Packages (optional)")
    for name, min_ver, desc in OPTIONAL_PACKAGES:
        ok, ver = check_package(name)
        status = f"OK ({ver})" if ok else f"MISSING — {desc}"
        marker = "  " if ok else "??"
        print(f"  {marker} {name:20s} {status}")
        if not ok:
            warnings += 1

    # 2. CUDA
    print("\n[3] CUDA / GPU")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  OK CUDA available: {torch.version.cuda}")
            print(f"  OK GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"     GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("  !! CUDA not available")
            errors += 1
    except Exception as e:
        print(f"  !! Cannot check CUDA: {e}")
        errors += 1

    # 3. Config and paths
    print("\n[4] Configuration")
    config_path = os.path.join(RENDER_PKG, args.config)
    if check_file(config_path):
        print(f"  OK Config: {config_path}")
        try:
            from encoders.config import load_config
            cfg = load_config(config_path)
            print("  OK Config loaded and resolved")
        except Exception as e:
            print(f"  !! Config load failed: {e}")
            errors += 1
            cfg = None
    else:
        print(f"  !! Config not found: {config_path}")
        errors += 1
        cfg = None

    # 4. Weights
    print("\n[5] Model Weights")
    if cfg:
        weight_checks = {
            "UniLat encoder": [
                cfg["weights"]["unilat_encoder"] + ".json",
                cfg["weights"]["unilat_encoder"] + ".safetensors",
            ],
            "TRELLIS SLAT":   [
                cfg["weights"]["slat_encoder"] + ".json",
                cfg["weights"]["slat_encoder"] + ".safetensors",
            ],
            "TRELLIS SS":     [
                cfg["weights"]["ss_encoder"] + ".json",
                cfg["weights"]["ss_encoder"] + ".safetensors",
            ],
            "DINOv2":         [cfg["weights"]["dinov2"]],
        }
        for name, paths in weight_checks.items():
            all_ok = all(check_file(p) for p in paths)
            if all_ok:
                print(f"  OK {name}")
            else:
                for p in paths:
                    if not check_file(p):
                        print(f"  !! {name}: MISSING {p}")
                errors += 1

    # 5. Third-party code
    print("\n[6] Third-Party Model Code")
    if cfg:
        tp_checks = {
            "TRELLIS":  cfg.get("third_party", {}).get("trellis"),
            "UniLat3D": cfg.get("third_party", {}).get("unilat3d"),
            "DINOv2":   cfg.get("weights", {}).get("dinov2_repo"),
        }
        for name, path in tp_checks.items():
            if path and check_dir(path):
                print(f"  OK {name}: {path}")
            else:
                print(f"  !! {name}: MISSING {path}")
                errors += 1

    # 6. Blender
    print("\n[7] Blender")
    if cfg:
        blender = cfg["paths"]["blender_bin"]
        if check_file(blender):
            print(f"  OK Blender: {blender}")
        else:
            print(f"  ?? Blender not found: {blender} (only needed for render stage)")
            warnings += 1

    # 7. spconv algo
    print("\n[8] Environment Variables")
    spconv = os.environ.get("SPCONV_ALGO")
    if spconv == "native":
        print("  OK SPCONV_ALGO=native")
    else:
        print(f"  ?? SPCONV_ALGO={spconv or '(unset)'} — set to 'native' before encoding")
        warnings += 1

    # Summary
    print("\n" + "=" * 60)
    if errors == 0 and warnings == 0:
        print("ALL CHECKS PASSED")
    elif errors == 0:
        print(f"PASSED with {warnings} warning(s)")
    else:
        print(f"FAILED: {errors} error(s), {warnings} warning(s)")
    print("=" * 60)

    sys.exit(1 if errors > 0 else 0)


if __name__ == "__main__":
    main()

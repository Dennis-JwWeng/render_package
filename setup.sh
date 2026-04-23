#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# setup.sh — One-shot environment bootstrap for render_package.
#
# Extracts the bundled conda environment (envs/env.tar.gz) into envs/env/,
# runs conda-unpack to fix path prefixes, and prints activation instructions.
#
# Usage (on the target machine, after extracting the archive):
#   cd render_package
#   bash setup.sh
#
# After setup, activate with:
#   source envs/env/bin/activate
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_TAR="$SCRIPT_DIR/envs/env.tar.gz"
ENV_DIR="$SCRIPT_DIR/envs/env"

if [ -f "$ENV_DIR/bin/python" ]; then
    echo "[setup] Environment already extracted at: $ENV_DIR"
    echo "[setup] To re-extract, remove $ENV_DIR first."
    echo ""
    echo "Activate with:"
    echo "  source $ENV_DIR/bin/activate"
    exit 0
fi

if [ ! -f "$ENV_TAR" ]; then
    echo "[ERROR] Environment archive not found: $ENV_TAR"
    echo "        Did you extract the full render_package archive?"
    exit 1
fi

echo "[setup] Extracting conda environment (~4.5 GiB) ..."
mkdir -p "$ENV_DIR"
tar -xzf "$ENV_TAR" -C "$ENV_DIR"

echo "[setup] Fixing path prefixes (conda-unpack) ..."
set +u
source "$ENV_DIR/bin/activate"
set -u
conda-unpack 2>/dev/null || true

echo "[setup] Verifying Python ..."
"$ENV_DIR/bin/python" -c "import torch; print(f'  PyTorch {torch.__version__}  CUDA: {torch.cuda.is_available()}')"
"$ENV_DIR/bin/python" -c "import open3d; print(f'  Open3D {open3d.__version__}')" 2>/dev/null || true
"$ENV_DIR/bin/python" -c "import spconv; print(f'  spconv ok')" 2>/dev/null || true

echo ""
echo "=== Setup complete ==="
echo ""
echo "Activate the environment:"
echo "  source $ENV_DIR/bin/activate"
echo ""
echo "Then run (examples):"
echo "  # Edit config first"
echo "  vi $SCRIPT_DIR/config/default.yaml"
echo ""
echo "  # Render"
echo "  python render_github.py --num_shards 10"
echo ""
echo "  # Encode (multi-GPU)"
echo "  SPCONV_ALGO=native python encode_all.py --render_root /path/to/renders"
echo ""
echo "  # Full pipeline (render + encode interleaved)"
echo "  SPCONV_ALGO=native python run_pipeline.py --render_gpus 0,1 --encode_gpus 2,3"

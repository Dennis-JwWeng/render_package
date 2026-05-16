#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# pack.sh — Bundle render_package into a fully self-contained tarball.
#
# Includes: Python source, Blender, weights, third-party model code,
# AND the pre-built conda environment (envs/env.tar.gz).
#
# On the target machine, just:
#   tar xzf render_package.tar.gz
#   cd render_package
#   bash setup.sh          # extracts env, fixes paths
#   source envs/env/bin/activate
#   vi config/default.yaml # edit data paths
#
# Usage:
#   bash pack.sh                           # → render_package.tar.gz
#   bash pack.sh /tmp/my_bundle.tar.gz     # custom output path
#
# Requires: rsync
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="${1:-render_package.tar.gz}"
STAGING_ROOT="${PACK_STAGING_DIR:-$(mktemp -d)}"
STAGING="$STAGING_ROOT/render_package"

echo "=== render_package packer ==="
echo "Source:  $SCRIPT_DIR"
echo "Output:  $OUTPUT"
echo "Staging: $STAGING"
echo ""

mkdir -p "$STAGING"

# ── 1. Python source + configs (exclude build artifacts & large blobs) ──
echo "[1/7] Copying Python source & configs..."
rsync -a \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.log' \
    --exclude='*.tar.gz' \
    --exclude='root.code-workspace' \
    --exclude='.git' \
    --exclude='blender-3.5.1-linux-x64' \
    --exclude='weights' \
    --exclude='third_party' \
    --exclude='envs' \
    "$SCRIPT_DIR/" "$STAGING/"

# ── 2. Blender binary bundle ────────────────────────────────────────
echo "[2/7] Copying Blender 3.5.1 bundle (~1.2 GiB)..."
rsync -aL "$SCRIPT_DIR/blender-3.5.1-linux-x64/" "$STAGING/blender-3.5.1-linux-x64/"

# ── 3. Weights (resolve symlinks → real copies) ─────────────────────
echo "[3/7] Copying weights (resolving symlinks)..."
mkdir -p "$STAGING/weights/unilat" "$STAGING/weights/trellis" "$STAGING/weights/dinov2"

cp -L "$SCRIPT_DIR/weights/unilat/encoder.json"        "$STAGING/weights/unilat/"
cp -L "$SCRIPT_DIR/weights/unilat/encoder.safetensors"  "$STAGING/weights/unilat/"

cp -L "$SCRIPT_DIR/weights/trellis/slat_enc.json"           "$STAGING/weights/trellis/"
cp -L "$SCRIPT_DIR/weights/trellis/slat_enc.safetensors"    "$STAGING/weights/trellis/"
cp -L "$SCRIPT_DIR/weights/trellis/ss_enc.json"             "$STAGING/weights/trellis/"
cp -L "$SCRIPT_DIR/weights/trellis/ss_enc.safetensors"      "$STAGING/weights/trellis/"

cp -L "$SCRIPT_DIR/weights/dinov2/dinov2_vitl14_reg4_pretrain.pth" "$STAGING/weights/dinov2/"

# ── 4. Third-party model code (resolve symlinks → real copies) ──────
echo "[4/7] Copying third-party code (resolving symlinks)..."
mkdir -p "$STAGING/third_party"

rsync -aL \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
    "$SCRIPT_DIR/third_party/trellis/" "$STAGING/third_party/trellis/"

rsync -aL \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
    "$SCRIPT_DIR/third_party/unilat3d/" "$STAGING/third_party/unilat3d/"

rsync -aL \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
    "$SCRIPT_DIR/third_party/dinov2/" "$STAGING/third_party/dinov2/"

# ── 5. Conda environment (pre-packed) ───────────────────────────────
echo "[5/7] Copying conda environment (~4.5 GiB)..."
ENV_TAR="$SCRIPT_DIR/envs/env.tar.gz"
if [ ! -f "$ENV_TAR" ]; then
    echo "  WARNING: envs/env.tar.gz not found."
    echo "  Build it first:  conda-pack -n vinedresser3d -o envs/env.tar.gz --ignore-editable-packages"
    echo "  Continuing without the environment (code-only pack)."
else
    mkdir -p "$STAGING/envs"
    cp "$ENV_TAR" "$STAGING/envs/env.tar.gz"
fi

# ── 6. Verify completeness ──────────────────────────────────────────
echo "[6/7] Verifying package..."
ERRORS=0

check_file() {
    if [ ! -f "$STAGING/$1" ]; then
        echo "  MISSING: $1"
        ERRORS=$((ERRORS + 1))
    fi
}

check_dir() {
    if [ ! -d "$STAGING/$1" ]; then
        echo "  MISSING DIR: $1"
        ERRORS=$((ERRORS + 1))
    fi
}

# Core Python
check_file "encoders/__init__.py"
check_file "encoders/config.py"
check_file "encoders/features.py"
check_file "encoders/unilat.py"
check_file "encoders/slat.py"
check_file "encoders/ss.py"
check_file "encoders/utils.py"
check_file "encoders/dinov2_loader.py"
check_file "render_github.py"
check_file "render_only.py"
check_file "encode_shard.py"
check_file "encode_all.py"
check_file "run_pipeline.py"
check_file "verify_install.py"
check_file "setup.sh"
check_file "config/default.yaml"
check_file "config/github_downloads_encode_upload.yaml"
check_file "config/github_downloads_encode_upload.local.yaml.example"

# Envs
check_file "environment.yml"
check_file "environment_encode.yml"
check_file "requirements_encode.txt"
check_file "README.md"

# Blender
check_file "blender-3.5.1-linux-x64/blender"
check_file "dataset_toolkits/blender_script/render.py"

# Weights
check_file "weights/unilat/encoder.json"
check_file "weights/unilat/encoder.safetensors"
check_file "weights/trellis/slat_enc.json"
check_file "weights/trellis/slat_enc.safetensors"
check_file "weights/trellis/ss_enc.json"
check_file "weights/trellis/ss_enc.safetensors"
check_file "weights/dinov2/dinov2_vitl14_reg4_pretrain.pth"

# Third-party
check_dir  "third_party/trellis/models"
check_dir  "third_party/dinov2/dinov2"
check_dir  "third_party/unilat3d/models"

# Bundled env
if [ -f "$ENV_TAR" ]; then
    check_file "envs/env.tar.gz"
fi

# No dangling symlinks
DANGLING=$(find "$STAGING" -type l ! -exec test -e {} \; -print 2>/dev/null | head -10)
if [ -n "$DANGLING" ]; then
    echo "  DANGLING SYMLINKS:"
    echo "$DANGLING" | sed 's/^/    /'
    ERRORS=$((ERRORS + 1))
fi

if [ "$ERRORS" -gt 0 ]; then
    echo ""
    echo "ERROR: $ERRORS verification failures. Fix before packaging."
    rm -rf "$(dirname "$STAGING")"
    exit 1
fi

echo "  All checks passed."

# ── 7. Create tarball ────────────────────────────────────────────────
echo "[7/7] Creating tarball..."
PARENT="$(dirname "$STAGING")"
tar -czf "$OUTPUT" -C "$PARENT" render_package

SIZE=$(du -sh "$OUTPUT" | cut -f1)
echo ""
echo "=== Done ==="
echo "Archive: $OUTPUT ($SIZE)"
echo ""
echo "Deploy on target machine:"
echo "  tar xzf $OUTPUT"
echo "  cd render_package"
echo "  bash setup.sh                 # extract env, verify deps"
echo "  source envs/env/bin/activate  # activate"
echo "  vi config/default.yaml        # set data paths"
echo "  SPCONV_ALGO=native python encode_all.py --render_root /path/to/renders"

# Cleanup staging
rm -rf "$STAGING_ROOT"

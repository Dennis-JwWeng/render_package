# render\_package — 3D Render + Latent Encoding Pipeline

Self-contained toolkit that **renders** 3D assets with Blender and **encodes** them into
multiple latent representations (DINOv2, UniLat, TRELLIS SLAT, TRELLIS SS).

Everything is controlled by a single YAML config (`config/default.yaml`).
To run on a new machine, edit the config paths and create the conda environment.

---

## Directory Layout

```
render_package/
├── config/
│   └── default.yaml            # single config — edit per machine
│
├── encoders/                   # Python package: latent encoding pipeline
│   ├── __init__.py             #   public API: encode_object, encode_shard
│   ├── config.py               #   YAML loader + path resolver
│   ├── dinov2_loader.py        #   DINOv2 ViT-L/14+reg singleton
│   ├── features.py             #   Stage 1: voxelize + DINOv2 aggregation
│   ├── unilat.py               #   Stage 2a: UniLat sparse encoder
│   ├── slat.py                 #   Stage 2b: TRELLIS SLAT encoder
│   ├── ss.py                   #   Stage 2c: TRELLIS SS encoder
│   └── utils.py                #   voxelization, projection, atomic I/O
│
├── dataset_toolkits/           # render helpers
│   ├── blender_script/
│   │   └── render.py           #   Blender-internal render script (bpy)
│   └── utils.py                #   Hammersley view sampling, hashing
│
├── third_party/                # model source code (symlinks or real dirs)
│   ├── trellis/                #   TRELLIS models & sparse modules
│   ├── unilat3d/               #   UniLat3D encoder package (from HF zhaxie/latent)
│   └── dinov2/                 #   facebookresearch/dinov2 hub code
│
├── weights/                    # checkpoint files (symlinks or copies)
│   ├── unilat/
│   │   ├── encoder.json
│   │   └── encoder.safetensors #   ~672 MiB
│   ├── trellis/
│   │   ├── slat_enc.json
│   │   ├── slat_enc.safetensors #  ~165 MiB
│   │   ├── ss_enc.json
│   │   └── ss_enc.safetensors  #   ~114 MiB
│   └── dinov2/
│       └── dinov2_vitl14_reg4_pretrain.pth  # ~1.13 GiB
│
├── blender-3.5.1-linux-x64/   # bundled Blender (~1.2 GiB)
│   └── blender                 #   executable
│
├── render_github.py            # CLI: .tar.zst shards → Blender render
├── render_only.py              # CLI: direct GLB/OBJ → Blender render
├── encode_all.py               # CLI: unified multi-GPU encoder (all datasets)
├── encode_shard.py             # CLI: encode one rendered shard
├── run_pipeline.py             # CLI: interleaved render + encode
├── verify_install.py           # post-deploy dependency check
├── pack.sh                     # build portable tarball (weights + libs + env)
├── setup.sh                    # target-machine bootstrap (extract env, verify)
│
├── envs/
│   └── env.tar.gz              # pre-built conda env (~4.5 GiB, conda-pack)
│
├── environment.yml             # conda env spec (alternative to bundled env)
├── requirements_encode.txt     # pip deps reference
└── environment_encode.yml      # conda env spec (alternative to bundled env)
```

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Input: .tar.zst shard / .glb / .obj / .ply / .fbx / ...       │
└──────────────────────┬──────────────────────────────────────────┘
                       │
            ┌──────────▼──────────┐
            │   Blender Render    │  render_github.py / render_only.py
            │  (CYCLES GPU, 40   │  env: render (conda)
            │   views @ 512²)    │
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │  Per-object output  │  images/  mesh.ply  transforms.json
            └──────────┬──────────┘
                       │
    ┌──────────────────▼──────────────────┐
    │  Stage 1: Voxelize + DINOv2 Agg.   │  encoders/features.py
    │  mesh.ply → 64³ voxels             │  env: vinedresser3d (conda)
    │  40 views × ViT-L/14 → per-voxel  │
    │  features [N, 1024] fp16           │
    └──────────────────┬──────────────────┘
                       │  latents/dino_features.npz
                       │
    ┌──────────┬───────┴───────┐
    ▼          ▼               ▼
┌────────┐ ┌────────┐   ┌──────────┐
│UniLat  │ │ SLAT   │   │   SS     │  (depends on SLAT)
│Stage 2a│ │Stage 2b│──→│ Stage 2c │
└───┬────┘ └───┬────┘   └────┬─────┘
    │          │              │
    ▼          ▼              ▼
unilat.npz  slat.npz       ss.npz        all in latents/
```

### Per-object output (inside a shard)

```
shard_XXXXXX/
└── <object_id>/
    ├── images/                 # rendered PNG views
    │   ├── 000.png ... 039.png
    ├── mesh.ply                # normalized mesh
    ├── transforms.json         # camera params (NeRF convention)
    └── latents/                # encoded features
        ├── dino_features.npz   # keys: features[N,1024] voxel_indices[N,3]
        ├── unilat.npz          # key:  latent[D]
        ├── slat.npz            # keys: feats[M,8]  coords[M,4]
        └── ss.npz              # key:  ss[C,R,R,R]
```

---

## Quick Start

### 1. Deploy (from archive)

The packaged archive is fully self-contained. No conda/pip install needed:

```bash
tar xzf render_package.tar.gz
cd render_package
bash setup.sh                 # extracts bundled env (~4.5 GiB), verifies deps
source envs/env/bin/activate  # single env for both render and encode
```

Requirements on the target machine:
- Linux x86\_64
- NVIDIA GPU with CUDA 12.x drivers (driver >= 525)
- No conda, no pip, no internet needed

**Alternative (build env from spec):** If not using the bundled env, create one
from the spec files:

```bash
conda env create -f environment_encode.yml
conda activate encode
pip install flash-attn==2.7.3 --no-build-isolation
```

### 2. Configuration

Edit `config/default.yaml`:

```yaml
paths:
  data_root: /your/data/root       # where rendered shards live
  blender_bin: auto                # auto = bundled blender-3.5.1

weights:
  unilat_encoder: weights/unilat/encoder
  slat_encoder:   weights/trellis/slat_enc
  ss_encoder:     weights/trellis/ss_enc
  dinov2:         weights/dinov2/dinov2_vitl14_reg4_pretrain.pth
  dinov2_repo:    third_party/dinov2

third_party:
  trellis:  third_party/trellis
  unilat3d: third_party/unilat3d
```

All relative paths are resolved from the `render_package/` root.

### 3. Render

```bash
source envs/env/bin/activate
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Benchmark (test with 1 shard, auto-detect GPUs)
python render_github.py --benchmark

# Full run (100 shards)
python render_github.py --num_shards 100
```

### 4. Encode

```bash
source envs/env/bin/activate

# Encode ALL data sources at once (ABO + HSSD + GitHub + ...)
SPCONV_ALGO=native python encode_all.py \
    --config config/default.yaml \
    --render_root /path/to/TRELLIS-500K/renders \
    --render_root /path/to/TRELLIS-500K/github/render

# Dry run: just count objects per dataset
SPCONV_ALGO=native python encode_all.py \
    --render_root /path/to/TRELLIS-500K/renders \
    --dry_run

# Single dataset only
SPCONV_ALGO=native python encode_all.py \
    --render_root /path/to/renders/HSSD

# Specific shards within a dataset
SPCONV_ALGO=native python encode_all.py \
    --render_root /path/to/renders/ABO \
    --shards 0,1,2

# Specific stages only
SPCONV_ALGO=native python encode_all.py \
    --render_root /path/to/renders \
    --stages slat,ss

# Single shard (low-level CLI)
SPCONV_ALGO=native python encode_shard.py \
    --config config/default.yaml \
    --shard_dir /path/to/render/shard_016725
```

### 5. Multi-GPU Encoding

`encode_all.py` supports multi-GPU via multi-process workers. Each GPU gets
its own process with isolated `CUDA_VISIBLE_DEVICES`. Objects are distributed
via a shared work-stealing queue (faster GPUs process more objects).

```bash
# Auto-detect all GPUs (default)
SPCONV_ALGO=native python encode_all.py \
    --render_root /path/to/renders

# Specify GPU count
SPCONV_ALGO=native python encode_all.py \
    --render_root /path/to/renders \
    --num_gpus 4

# Specify exact GPU IDs
SPCONV_ALGO=native python encode_all.py \
    --render_root /path/to/renders \
    --gpus 0,2,4,6

# Single GPU (legacy behavior)
SPCONV_ALGO=native python encode_all.py \
    --render_root /path/to/renders \
    --num_gpus 1
```

### 6. Interleaved Render + Encode Pipeline

`run_pipeline.py` runs render and encode concurrently on separate GPU
partitions. As each shard finishes rendering, its objects are immediately
fed to the encode worker pool.

```bash
# Full interleaved pipeline (GPUs 0-1 render, GPUs 2-3 encode)
SPCONV_ALGO=native python run_pipeline.py \
    --config config/default.yaml \
    --render_gpus 0,1 \
    --encode_gpus 2,3

# Encode-only (multi-GPU, existing renders)
SPCONV_ALGO=native python run_pipeline.py \
    --config config/default.yaml \
    --encode_only \
    --render_dir /path/to/renders

# Render-only (no encoding)
python run_pipeline.py \
    --config config/default.yaml \
    --render_only
```

**GPU allocation in config:**

```yaml
gpus:
  render: auto       # or [0, 1]
  encode: auto       # or [2, 3]
```

In interleaved mode, render and encode GPUs should not overlap to avoid
memory contention. In encode-only mode, all GPUs are used for encoding.

**Resume:** Both stages are fully resumable. On restart, completed shards are
detected by scanning for output files and skipped automatically.

### 7. Watchdog, upload, and upload record (recommended for GitHub tar shards)

Use one YAML (e.g. `config/trellis_github_archives_6_first100.yaml`) for paths, `hf.download` / `hf.upload`, and GPU pool. Optional secrets: create `config/<same_basename>.local.yaml` with `hf.upload.token` (gitignored); or set `HUGGINGFACE_HUB_TOKEN` / `HF_TOKEN`.

**Start the watchdog (loop every `pipeline.watchdog.interval_seconds`):**

```bash
cd /path/to/render_package
source envs/env/bin/activate
export SPCONV_ALGO=native
# If /tmp is full, point temp at your data mount (matches paths.render_tmp in YAML):
export TMPDIR=/path/to/<data_root>/github/.render_tmp
mkdir -p "$TMPDIR"

python pipeline_watchdog.py --config config/trellis_github_archives_6_first100.yaml
```

**Recommended shard-level production flow:**

Use the config to define an inclusive shard index range:

```yaml
hf:
  download:
    shard_index_start: 0
    shard_index_end: 99
```

For each shard in that range, the full watchdog does:

```text
download shard tar.zst -> render mesh/images/transforms -> encode latents -> pack encoded archive -> upload -> cleanup
```

With `pipeline.watchdog.overlap_render_encode: true`, full watchdog mode runs
that flow as a shard pipeline: when shard A finishes Blender rendering, shard A
is queued for encode/upload and Blender immediately moves on to shard B. This
keeps render moving instead of waiting for shard A's encode pass to finish.
Use non-overlapping `gpus.render` and `gpus.encode` lists for best stability;
the watchdog prints a warning if they overlap.

The uploaded archive is shard-scoped. Small shards keep the legacy
`github/render/<shard_id>.tar.zst` name; large shards can be split on object
boundaries into `github/render/<shard_id>.part_00000.tar.zst`,
`github/render/<shard_id>.part_00001.tar.zst`, and so on. Splitting is
controlled by `hf.upload.archive.max_part_bytes` and keeps every object's
`latents/`, `transforms.json`, and `mesh.ply` in the same part. The archive
contains only the encoded deliverables configured under `hf.upload.archive.include`,
normally:

```yaml
hf:
  upload:
    archive:
      max_part_bytes: 5000000000
      include:
        - latents
        - transforms.json
        - mesh.ply
      exclude_globs:
        - "images/**/*.png"
```

Rendered PNGs and intermediate render folders are not uploaded. After every part
for a shard passes remote size verification, the uploader can reclaim local space
according to config:

```yaml
pipeline:
  delete_source_shard_tar: true
hf:
  upload:
    delete_local_archive_after_success: true
    delete_local_when_remote_exists: true
    delete_render_dir_after_success: true
```

That removes:

- downloaded source shards: `<data_root>/shards/github/<shard_id>.tar.zst`
- temporary upload archives: `<data_root>/github/pack_for_upload/<shard_id>.tar.zst` or `<data_root>/github/pack_for_upload/<shard_id>.part_*.tar.zst`
- rendered shard directories: `<data_root>/github/render/<shard_id>/`

For long runs, start it in `tmux` so closing Cursor or SSH does not stop the job:

```bash
tmux new-session -d -s trellis_watchdog \
  'cd /path/to/render_package && source envs/env/bin/activate && \
   export SPCONV_ALGO=native && \
   export TMPDIR=/path/to/<data_root>/github/.render_tmp && mkdir -p "$TMPDIR" && \
   python pipeline_watchdog.py --config config/trellis_github_archives_6_first100.yaml \
     2>&1 | tee -a /path/to/<data_root>/github/watchdog_full.log'

tmux attach -t trellis_watchdog
```

**Two-phase control (render first, encode later):** use the same `pipeline_state.json`. After all shards have `mesh.ply`, run encode.

```bash
# Phase 1 — download + Blender only; set pipeline.watchdog.delete_source_shard_tar_after_render: true to drop .tar.zst when done
python pipeline_watchdog.py --config config/trellis_github_archives_6_first100.yaml --render-only

# Phase 2 — no download/render; only encode + optional hf.upload
python pipeline_watchdog.py --config config/trellis_github_archives_6_first100.yaml --encode-only
```

**Single cycle then exit (smoke test or cron):**

```bash
cd /path/to/render_package && source envs/env/bin/activate
export SPCONV_ALGO=native
python pipeline_watchdog.py --config config/trellis_github_archives_6_first100.yaml --once
```

**Background + log file on the data disk:**

```bash
cd /path/to/render_package && source envs/env/bin/activate
export SPCONV_ALGO=native
export TMPDIR=/path/to/<data_root>/github/.render_tmp
mkdir -p "$TMPDIR"
nohup python pipeline_watchdog.py --config config/trellis_github_archives_6_first100.yaml \
  >> /path/to/<data_root>/github/watchdog.log 2>&1 &
```

**Pack and upload strict `encode_done` shards to Hugging Face** (`hf.upload.enabled: false` still works with `--force`):

```bash
cd /path/to/render_package && source envs/env/bin/activate
python upload_hf_encoded_shards.py \
  --config config/trellis_github_archives_6_first100.yaml \
  --all-verified --force
```

For production batches with large shards, prefer the upload watchdog wrapper.
It uploads one shard at a time, lets `upload_hf_encoded_shards.py` handle every
part for that shard, skips shards already complete on Hub (legacy single file or
full part set), and retries a shard if the upload process exceeds a timeout:

```bash
tmux new-session -d -s trellis_upload \
  'cd /path/to/render_package && source envs/env/bin/activate && \
   python upload_hf_shards_watchdog.py \
     --config config/trellis_github_archives_6_first100.yaml \
     --all-verified --skip-remote --force \
     --timeout-minutes 90 --retries 3 \
     2>&1 | tee -a /path/to/<data_root>/github/upload_watchdog.log'
```

Some encoded shards can be much larger than their source `.tar.zst`: the source
archive may be ~2 GiB, while encoded `latents/` plus `mesh.ply` and
`transforms.json` can produce around 10 GiB of upload data for shards with many
objects. With `max_part_bytes: 5000000000`, those large outputs are uploaded as
several smaller archives, improving retry behavior without changing shard-level
pipeline status or cleanup.

**Refresh `upload_record.json` (Hub listing vs local encode_done in shard range):**

```bash
cd /path/to/render_package && source envs/env/bin/activate
python refresh_upload_record.py --config config/trellis_github_archives_6_first100.yaml
```

More detail (tmux, state files, retry policy): see **`RUNBOOK.md`**.

---

## Encoding Stages Detail

| Stage | Input | Output | Model | Weights Size |
|---|---|---|---|---|
| `dino_features` | `mesh.ply` + `transforms.json` + `images/` | `dino_features.npz` | DINOv2 ViT-L/14+reg | 1.13 GiB |
| `unilat` | `dino_features.npz` | `unilat.npz` | UniLat3D Sparse Encoder | 672 MiB |
| `slat` | `dino_features.npz` | `slat.npz` | TRELLIS SLAT Encoder | 165 MiB |
| `ss` | `slat.npz` (coords) | `ss.npz` | TRELLIS SS Encoder | 114 MiB |

### Stage dependencies

```
dino_features ──→ unilat
              ──→ slat ──→ ss
```

`ss` requires `slat` to have run first (it reads the voxel coordinates from
`slat.npz`). All other stages only depend on `dino_features`.

### Idempotency

Every stage checks for its output file before running. Re-running the pipeline
skips already-completed objects. This makes the pipeline crash-safe and
resumable.

---

## NPZ Output Format

All latents use NumPy compressed archives (`.npz`):

| File | Keys | Shapes | Description |
|---|---|---|---|
| `dino_features.npz` | `features`, `voxel_indices` | `[N, 1024] fp16`, `[N, 3] int64` | DINOv2 per-voxel features on 64³ grid |
| `unilat.npz` | `latent` | `[D]` float32 | UniLat global latent vector |
| `slat.npz` | `feats`, `coords` | `[M, 8] float32`, `[M, 4] int32` | TRELLIS structured latent (sparse) |
| `ss.npz` | `ss` | `[C, R, R, R]` float32 | TRELLIS sparse structure (dense volume) |

---

## Python API

```python
from encoders.config import load_config
from encoders import encode_object, encode_shard

cfg = load_config("config/default.yaml")

# Encode one object
results = encode_object("/path/to/shard/object_id", cfg)
# returns: {"dino_features": "ok", "unilat": "ok", "slat": "ok", "ss": "ok"}

# Encode all objects in a shard
counts = encode_shard("/path/to/shard", cfg)
# returns: {"dino_features_ok": 100, "dino_features_skip": 50, ...}

# Specific stages only
results = encode_object(scene_dir, cfg, stages=["slat", "ss"])
```

---

## Weights & Third-Party Code

### Weights

The `weights/` directory contains symlinks to model checkpoints. For
packaging/portability, use `pack.sh` to copy real files instead of symlinks.

| Weight | Source | Format |
|---|---|---|
| UniLat encoder | `weights/unilat/encoder.{json,safetensors}` | safetensors |
| TRELLIS SLAT | TRELLIS-image-large ckpt | safetensors |
| TRELLIS SS | TRELLIS-image-large ckpt | safetensors |
| DINOv2 ViT-L/14+reg | PyTorch Hub / facebookresearch | .pth |

### Third-Party Model Code

| Library | Source | Purpose |
|---|---|---|
| `trellis` | [TRELLIS](https://github.com/microsoft/TRELLIS) | SLAT/SS encoder models, sparse modules |
| `unilat3d` | UniLat3D | UniLat encoder, sparse tensor |
| `dinov2` | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | ViT backbone hub code |

---

## Packaging & Deployment

### `pack.sh` — Create Portable Tarball

```bash
bash pack.sh [output_path.tar.gz]
```

Creates a fully self-contained archive (~7 GiB) that includes:
- All Python source code
- Blender 3.5.1 binary bundle (~1.2 GiB)
- Model weights, real copies, no symlinks (~2.1 GiB)
- Third-party model code (~16 MiB)
- Pre-built conda environment via conda-pack (~4.5 GiB)

**No conda/pip/internet needed on the target machine.** Just NVIDIA drivers.

### Deploy on target machine

```bash
tar xzf render_package.tar.gz
cd render_package
bash setup.sh                   # extract env, verify
source envs/env/bin/activate
vi config/default.yaml          # set data_root paths
```

### Rebuild the env tarball (if you modify the environment)

```bash
conda-pack -n vinedresser3d -o envs/env.tar.gz --ignore-editable-packages --force
```

---

## Minimal Encoding Environment

The `environment_encode.yml` defines the minimal conda environment for
encoding. Key dependencies:

| Package | Version | Role |
|---|---|---|
| Python | 3.10 | Runtime |
| PyTorch | 2.5.0+cu124 | Tensor operations, GPU |
| torchvision | 0.20.0+cu124 | Image transforms |
| flash-attn | 2.7.3 | Fast attention (UniLat) |
| spconv-cu124 | 2.3.8 | Sparse convolutions |
| open3d | ≥0.18 | Mesh voxelization |
| xformers | 0.0.28.post2 | Efficient attention (DINOv2) |
| numpy | ≥1.24 | Array operations |
| scipy | ≥1.10 | Sparse matrices |
| safetensors | ≥0.4 | Weight loading |
| plyfile | ≥1.0 | PLY mesh I/O |
| Pillow | ≥10.0 | Image loading |
| tqdm | ≥4.60 | Progress bars |
| PyYAML | ≥6.0 | Config loading |
| utils3d | 1.7 (git) | Camera projection |

### CUDA requirements

- NVIDIA driver ≥ 525 (CUDA 12.x compatible)
- CUDA Toolkit 12.4 (runtime, for spconv and flash-attn)

---

## Troubleshooting

### `SPCONV_ALGO=native`

Always set this environment variable before running encode scripts. Without
it, spconv may attempt to use algorithms not available on all hardware:

```bash
export SPCONV_ALGO=native
```

### Flash Attention installation

`flash-attn` requires building from source or a pre-built wheel matching
your exact PyTorch + CUDA version:

```bash
pip install flash-attn==2.7.3 --no-build-isolation
```

### Blender X11/GL errors

If Blender fails to start with missing library errors, activate the render
conda environment which provides X11/OpenGL system libraries:

```bash
conda activate render
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

### Resuming after crash

The pipeline is fully idempotent. Simply re-run the same command — completed
objects are automatically skipped.

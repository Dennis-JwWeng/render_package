#!/usr/bin/env python3
"""
Trellis500K GitHub archives rendering pipeline.

Decompress .tar.zst shards, extract 3D models from zip, render with Blender CYCLES GPU,
and clean up. Supports benchmark mode (time-limited) and full processing mode.

Usage:
  # Benchmark: test first zst with 4/6/8 workers, 5 min each
  python render_github.py --benchmark

  # Full run: process 100 zst shards
  python render_github.py --num_shards 100
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import shutil
import signal
import sys
import tarfile
import tempfile
import time
import zipfile
from multiprocessing import Process, Value, Event
from subprocess import call, Popen, DEVNULL, TimeoutExpired

import numpy as np
import zstandard

# ── Paths ──
_DATA_ROOT = os.environ.get(
    "TRELLIS_GITHUB_DATA_ROOT",
    "/mnt/pfs/ca41bi/omni3d/TRELLIS-500K",
)
SHARD_DIR = os.path.join(_DATA_ROOT, "trellis500k-github-archives-5/shards/github")
RAW_DIR = os.path.join(_DATA_ROOT, "github/raw")
RENDER_DIR = os.path.join(_DATA_ROOT, "github/render")
# Per-job Blender/temp output (do not use system /tmp; many clusters block or break it)
RENDER_TMP_PARENT = os.environ.get(
    "GITHUB_RENDER_TMP",
    os.path.join(_DATA_ROOT, "github", ".render_tmp"),
)
RENDER_PKG = os.path.dirname(os.path.abspath(__file__))
BLENDER_BIN = os.path.join(RENDER_PKG, "blender-3.5.1-linux-x64", "blender")
BLENDER_SCRIPT = os.path.join(RENDER_PKG, "dataset_toolkits", "blender_script", "render.py")

MODEL_EXTS = {".glb", ".gltf", ".obj", ".fbx", ".stl", ".usd", ".usda", ".dae", ".ply", ".abc"}
NUM_VIEWS = 40
NUM_GPUS = 8


def _detect_num_gpus():
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True, stderr=subprocess.DEVNULL)
        return max(1, len([ln for ln in out.splitlines() if ln.strip().startswith("GPU ")]))
    except (OSError, subprocess.CalledProcessError, FileNotFoundError):
        return NUM_GPUS

sys.path.insert(0, os.path.join(RENDER_PKG, "dataset_toolkits"))
sys.path.insert(0, RENDER_PKG)
from utils import sphere_hammersley_sequence


def build_views(num_views, seed):
    rng = np.random.default_rng(seed)
    offset = (rng.random(), rng.random())
    views = []
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views, offset)
        views.append({"yaw": y, "pitch": p, "radius": 2, "fov": 40 / 180 * np.pi})
    return views


def render_one(blender_bin, model_path, out_dir, gpu_id, num_views=40):
    """Render a single 3D model file. Returns True on success."""
    done_marker = os.path.join(out_dir, "mesh.ply")
    if os.path.isfile(done_marker):
        return "skip"

    stem = os.path.splitext(os.path.basename(model_path))[0]
    view_seed = int(hashlib.md5(stem.encode()).hexdigest()[:8], 16) % (2**31) or 1
    os.makedirs(RENDER_TMP_PARENT, exist_ok=True)
    render_tmp = tempfile.mkdtemp(prefix=f"render_{stem[:16]}_", dir=RENDER_TMP_PARENT)

    try:
        views = build_views(num_views, view_seed)
        args = [
            blender_bin, "-b", "-P", BLENDER_SCRIPT, "--",
            "--views", json.dumps(views),
            "--object", os.path.abspath(model_path),
            "--resolution", "512",
            "--output_folder", render_tmp,
            "--engine", "CYCLES",
            "--save_mesh",
        ]
        if model_path.endswith(".blend"):
            args.insert(1, os.path.abspath(model_path))
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # Keep Blender and libs off /tmp (same parent as mkdtemp above)
        env["TMPDIR"] = RENDER_TMP_PARENT
        env["TEMP"] = RENDER_TMP_PARENT
        env["TMP"] = RENDER_TMP_PARENT
        IDLE_TIMEOUT = 600  # kill if no new output file for 10 min
        proc = Popen(args, stdout=DEVNULL, stderr=DEVNULL, env=env)
        last_activity = time.time()
        last_file_count = 0
        while True:
            try:
                ret = proc.wait(timeout=10)
                break  # process finished
            except TimeoutExpired:
                cur_count = len(os.listdir(render_tmp)) if os.path.isdir(render_tmp) else 0
                if cur_count > last_file_count:
                    last_file_count = cur_count
                    last_activity = time.time()
                if time.time() - last_activity > IDLE_TIMEOUT:
                    proc.kill()
                    proc.wait()
                    print(f"  IDLE TIMEOUT ({IDLE_TIMEOUT}s, no new output) rendering {stem}", flush=True)
                    return "fail"

        tj = os.path.join(render_tmp, "transforms.json")
        mesh = os.path.join(render_tmp, "mesh.ply")
        if ret == 0 and os.path.isfile(tj) and os.path.isfile(mesh):
            images_dir = os.path.join(out_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
            for i in range(num_views):
                src = os.path.join(render_tmp, f"{i:03d}.png")
                dst = os.path.join(images_dir, f"{i:03d}.png")
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
            shutil.copy2(tj, os.path.join(out_dir, "transforms.json"))
            shutil.copy2(mesh, os.path.join(out_dir, "mesh.ply"))
            return "ok"
        else:
            return "fail"
    except Exception as e:
        print(f"  ERROR rendering {stem}: {e}", flush=True)
        return "fail"
    finally:
        shutil.rmtree(render_tmp, ignore_errors=True)


def render_worker(wid, gpu_id, tasks, stop_event, counter, num_views):
    """Worker process that renders assigned tasks."""
    n_ok, n_fail, n_skip = 0, 0, 0
    for gi, (model_path, out_dir) in enumerate(tasks):
        if stop_event.is_set():
            break
        name = os.path.basename(out_dir)
        print(f"[W{wid}:GPU{gpu_id}] ({gi+1}/{len(tasks)}) {name}", flush=True)
        result = render_one(BLENDER_BIN, model_path, out_dir, gpu_id, num_views)
        if result == "ok":
            n_ok += 1
        elif result == "skip":
            n_skip += 1
        else:
            n_fail += 1
        with counter.get_lock():
            counter.value += 1
    print(f"[W{wid}:GPU{gpu_id}] Done: ok={n_ok} fail={n_fail} skip={n_skip}", flush=True)


def decompress_zst(zst_path, raw_dir):
    """Decompress a .tar.zst to raw_dir. Returns list of (model_path, object_name) tuples."""
    shard_name = os.path.basename(zst_path).replace(".tar.zst", "")
    extract_dir = os.path.join(raw_dir, shard_name)
    os.makedirs(extract_dir, exist_ok=True)

    print(f"[DECOMPRESS] {os.path.basename(zst_path)} -> {extract_dir}", flush=True)
    t0 = time.time()

    dctx = zstandard.ZstdDecompressor()
    with open(zst_path, "rb") as f:
        reader = dctx.stream_reader(f)
        with tarfile.open(fileobj=reader, mode="r|") as tar:
            tar.extractall(extract_dir)

    # Find all zip files and extract 3D models
    models = []
    for root, dirs, files in os.walk(extract_dir):
        for fn in files:
            if fn.lower().endswith(".zip"):
                zip_path = os.path.join(root, fn)
                models.extend(_extract_models_from_zip(zip_path, extract_dir))

    elapsed = time.time() - t0
    print(f"[DECOMPRESS] Done in {elapsed:.1f}s, found {len(models)} 3D models", flush=True)
    return models, extract_dir


def _extract_models_from_zip(zip_path, extract_dir):
    """Extract 3D model files from a zip (including nested zips), using sha256 hash as unique name."""
    models = []
    models_dir = os.path.join(extract_dir, "_models")
    os.makedirs(models_dir, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            hash_map = _read_hash_manifest(zf)
            models.extend(_scan_zip_for_models(zf, hash_map, models_dir))

            # Recursively scan nested zip files
            for name in zf.namelist():
                if name.lower().endswith(".zip") and zf.getinfo(name).file_size > 100:
                    try:
                        inner_data = zf.read(name)
                        import io
                        with zipfile.ZipFile(io.BytesIO(inner_data)) as inner_zf:
                            models.extend(_scan_zip_for_models(inner_zf, hash_map, models_dir))
                    except (zipfile.BadZipFile, Exception):
                        pass

    except zipfile.BadZipFile:
        print(f"  WARNING: Bad zip file: {zip_path}", flush=True)
    except Exception as e:
        print(f"  WARNING: Error processing {zip_path}: {e}", flush=True)

    return models


def _read_hash_manifest(zf):
    """Read .objaverse-file-hashes.json from a zipfile, return {rel_path: sha256}."""
    hash_map = {}
    for name in zf.namelist():
        if name.endswith(".objaverse-file-hashes.json") or name == ".objaverse-file-hashes.json":
            try:
                data = json.loads(zf.read(name))
                for entry in data:
                    fid = entry.get("fileIdentifier", "")
                    sha = entry.get("sha256", "")
                    parts = fid.split("/blob/")
                    if len(parts) == 2:
                        rel_path = parts[1].split("/", 1)[-1] if "/" in parts[1] else ""
                        hash_map[rel_path] = sha
            except Exception:
                pass
            break
    return hash_map


def _scan_zip_for_models(zf, hash_map, models_dir):
    """Scan a zipfile for 3D model files and extract them to models_dir."""
    models = []
    for name in zf.namelist():
        ext = os.path.splitext(name)[1].lower()
        if ext not in MODEL_EXTS:
            continue
        if zf.getinfo(name).file_size < 100:
            continue

        sha = None
        for rel_path, h in hash_map.items():
            if name.endswith(rel_path):
                sha = h
                break

        if not sha:
            sha = hashlib.sha256(name.encode()).hexdigest()[:16]

        out_name = sha + ext
        out_path = os.path.join(models_dir, out_name)

        if not os.path.exists(out_path):
            try:
                data = zf.read(name)
                with open(out_path, "wb") as wf:
                    wf.write(data)
            except Exception:
                continue

        models.append((out_path, sha))
    return models


def get_ready_zst_files(max_count=100):
    """Get list of completed (non-downloading) .tar.zst files, sorted."""
    pattern = os.path.join(SHARD_DIR, "*.tar.zst")
    all_zst = sorted(glob.glob(pattern))
    # Filter out files that are still downloading
    ready = [f for f in all_zst if not os.path.exists(f + ".downloading")]
    return ready[:max_count]


def run_render_batch(models, num_workers, num_views, time_limit=None, shard_name=None):
    """Render a batch of models with given number of workers. Returns (completed, elapsed)."""
    tasks = []
    for model_path, obj_name in models:
        if shard_name:
            out_dir = os.path.join(RENDER_DIR, shard_name, obj_name)
        else:
            out_dir = os.path.join(RENDER_DIR, obj_name)
        tasks.append((model_path, out_dir))

    if not tasks:
        return 0, 0.0

    stop_event = Event()
    counter = Value("i", 0)

    n_workers = min(num_workers, len(tasks))
    buckets = [[] for _ in range(n_workers)]
    for i, task in enumerate(tasks):
        buckets[i % n_workers].append(task)

    t0 = time.time()
    procs = []
    for wid in range(n_workers):
        if not buckets[wid]:
            continue
        gpu_id = wid % NUM_GPUS
        p = Process(
            target=render_worker,
            args=(wid, gpu_id, buckets[wid], stop_event, counter, num_views),
            name=f"render-{wid}",
        )
        p.start()
        procs.append(p)

    if time_limit:
        deadline = t0 + time_limit
        while time.time() < deadline:
            if all(not p.is_alive() for p in procs):
                break
            time.sleep(5)
        stop_event.set()
        for p in procs:
            p.join(timeout=120)
            if p.is_alive():
                p.terminate()
    else:
        for p in procs:
            p.join()

    elapsed = time.time() - t0
    completed = counter.value
    return completed, elapsed


def benchmark(args):
    """Benchmark rendering speed with different worker counts on the first zst."""
    zst_files = get_ready_zst_files(1)
    if not zst_files:
        print("[ERROR] No ready .tar.zst files found!", flush=True)
        return

    zst_path = zst_files[0]
    print(f"{'='*60}", flush=True)
    print(f"BENCHMARK: {os.path.basename(zst_path)}", flush=True)
    print(f"{'='*60}", flush=True)

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(RENDER_DIR, exist_ok=True)

    models, extract_dir = decompress_zst(zst_path, RAW_DIR)
    if not models:
        print("[ERROR] No 3D models found in shard!", flush=True)
        return

    print(
        f"\nFound {len(models)} models. Benchmark {args.num_workers} workers, "
        f"time cap {args.bench_time}s.\n",
        flush=True,
    )

    time_limit = args.bench_time
    nw = args.num_workers
    bench_tag = "_benchmark"
    bench_render = os.path.join(RENDER_DIR, bench_tag)
    if os.path.isdir(bench_render):
        shutil.rmtree(bench_render)

    print(f"\n{'-'*40}", flush=True)
    print(f"Testing {nw} workers for {time_limit}s ...", flush=True)
    print(f"{'-'*40}", flush=True)

    completed, elapsed = run_render_batch(
        models, nw, NUM_VIEWS, time_limit=time_limit, shard_name=bench_tag
    )
    rate = completed / elapsed * 3600 if elapsed > 0 else 0
    results = [(nw, completed, elapsed, rate)]
    print(
        f"\n[RESULT] {nw} workers: {completed} models in {elapsed:.0f}s "
        f"({rate:.0f} models/hour)",
        flush=True,
    )

    # Clean up extracted data and benchmark renders
    shutil.rmtree(extract_dir, ignore_errors=True)
    shutil.rmtree(bench_render, ignore_errors=True)

    print(f"\n{'='*60}", flush=True)
    print("BENCHMARK SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Workers':<10}{'Completed':<12}{'Time(s)':<10}{'Rate(models/h)':<15}", flush=True)
    print(f"{'─'*47}", flush=True)
    for nw, comp, elapsed, rate in results:
        print(f"{nw:<10}{comp:<12}{elapsed:<10.0f}{rate:<15.0f}", flush=True)
    print(f"{'='*60}", flush=True)

    best = max(results, key=lambda x: x[3])
    print(f"\nBest: {best[0]} workers ({best[3]:.0f} models/hour)", flush=True)


def full_run(args):
    """Process zst shards one by one: decompress -> render -> delete."""
    num_shards = args.num_shards
    num_workers = args.num_workers
    sleep_mins = args.sleep_minutes

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(RENDER_DIR, exist_ok=True)

    processed = 0
    processed_files = set()

    print(f"[MAIN] Starting full render pipeline", flush=True)
    print(f"[MAIN] Target: {num_shards} shards, {num_workers} workers, {NUM_GPUS} GPUs", flush=True)

    while processed < num_shards:
        zst_files = get_ready_zst_files(num_shards)
        # Filter already processed
        pending = [f for f in zst_files if f not in processed_files]

        if not pending:
            print(f"[MAIN] No new shards available. Sleeping {sleep_mins} minutes...", flush=True)
            time.sleep(sleep_mins * 60)
            continue

        zst_path = pending[0]
        shard_name = os.path.basename(zst_path)
        processed += 1
        processed_files.add(zst_path)

        print(f"\n{'='*60}", flush=True)
        print(f"[MAIN] Shard {processed}/{num_shards}: {shard_name}", flush=True)
        print(f"{'='*60}", flush=True)

        try:
            models, extract_dir = decompress_zst(zst_path, RAW_DIR)
            shard_id = os.path.basename(zst_path).replace(".tar.zst", "")
            if models:
                completed, elapsed = run_render_batch(models, num_workers, NUM_VIEWS, shard_name=shard_id)
                rate = completed / elapsed * 3600 if elapsed > 0 else 0
                print(f"[MAIN] Shard done: {completed} models in {elapsed:.0f}s "
                      f"({rate:.0f} models/hour)", flush=True)
            else:
                print(f"[MAIN] No models found in {shard_name}", flush=True)

            # Clean up extracted raw data
            if os.path.isdir(extract_dir):
                shutil.rmtree(extract_dir, ignore_errors=True)
                print(f"[MAIN] Cleaned raw data: {extract_dir}", flush=True)

            # Delete the zst file (optional; staging copies may want --no-delete-shard)
            if not args.no_delete_shard and os.path.isfile(zst_path):
                os.remove(zst_path)
                print(f"[MAIN] Deleted shard: {zst_path}", flush=True)
            elif args.no_delete_shard and os.path.isfile(zst_path):
                print(f"[MAIN] Kept shard (--no-delete-shard): {zst_path}", flush=True)

        except Exception as e:
            print(f"[MAIN] ERROR processing {shard_name}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    print(f"\n[MAIN] All {processed} shards processed!", flush=True)


def render_shard_from_zst(
    zst_path: str,
    render_dir: str,
    raw_dir: str,
    gpu_ids: list[int],
    num_workers: int = 8,
    num_views: int = 40,
    render_tmp: str | None = None,
    no_delete_shard: bool = True,
) -> list[str]:
    """Render one .tar.zst shard end-to-end. Returns list of rendered object dirs.

    Importable entry point for the pipeline coordinator.
    """
    global NUM_GPUS, RENDER_DIR, RAW_DIR, RENDER_TMP_PARENT, NUM_VIEWS

    NUM_GPUS = len(gpu_ids)
    RENDER_DIR = render_dir
    RAW_DIR = raw_dir
    NUM_VIEWS = num_views
    if render_tmp:
        RENDER_TMP_PARENT = render_tmp
    os.makedirs(RENDER_TMP_PARENT, exist_ok=True)
    os.makedirs(RENDER_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)

    shard_id = os.path.basename(zst_path).replace(".tar.zst", "")

    models, extract_dir = decompress_zst(zst_path, RAW_DIR)
    if not models:
        print(f"[RENDER] No models found in {zst_path}", flush=True)
        return []

    completed, elapsed = run_render_batch(models, num_workers, num_views, shard_name=shard_id)
    rate = completed / elapsed * 3600 if elapsed > 0 else 0
    print(f"[RENDER] Shard {shard_id}: {completed} models in {elapsed:.0f}s ({rate:.0f}/h)", flush=True)

    if os.path.isdir(extract_dir):
        shutil.rmtree(extract_dir, ignore_errors=True)

    if not no_delete_shard and os.path.isfile(zst_path):
        os.remove(zst_path)

    shard_render_dir = os.path.join(RENDER_DIR, shard_id)
    rendered = []
    if os.path.isdir(shard_render_dir):
        for obj_name in sorted(os.listdir(shard_render_dir)):
            obj_dir = os.path.join(shard_render_dir, obj_name)
            if os.path.isdir(obj_dir) and os.path.isfile(os.path.join(obj_dir, "mesh.ply")):
                rendered.append(obj_dir)

    return rendered


def main():
    parser = argparse.ArgumentParser(description="Trellis500K GitHub archives render pipeline")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark on first zst")
    parser.add_argument("--bench_time", type=int, default=300, help="Benchmark time per worker count (seconds)")
    parser.add_argument("--num_shards", type=int, default=100, help="Number of zst shards to process")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of render workers")
    parser.add_argument("--num_views", type=int, default=40)
    parser.add_argument("--sleep_minutes", type=int, default=30, help="Sleep time when waiting for downloads")
    parser.add_argument(
        "--shard_dir",
        type=str,
        default=None,
        help="Directory containing *.tar.zst (default: trellis500k-github-archives-5/shards/github under _DATA_ROOT)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Root directory; sets github/raw and github/render under it (default: TRELLIS_GITHUB_DATA_ROOT or built-in _DATA_ROOT)",
    )
    parser.add_argument(
        "--no-delete-shard",
        action="store_true",
        help="Do not remove the .tar.zst after successful processing (keeps staging/pack copies)",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="GPU count for worker assignment (default: detect via nvidia-smi)",
    )
    parser.add_argument(
        "--render_tmp",
        type=str,
        default=None,
        help="Directory for render temp dirs (default: $GITHUB_RENDER_TMP or <data_root>/github/.render_tmp; never /tmp)",
    )
    args = parser.parse_args()

    global NUM_VIEWS, SHARD_DIR, RAW_DIR, RENDER_DIR, NUM_GPUS, RENDER_TMP_PARENT
    NUM_VIEWS = args.num_views
    if args.data_root:
        dr = os.path.abspath(os.path.expanduser(args.data_root))
        RAW_DIR = os.path.join(dr, "github/raw")
        RENDER_DIR = os.path.join(dr, "github/render")
        RENDER_TMP_PARENT = os.path.join(dr, "github", ".render_tmp")
        print(f"[MAIN] data_root: {dr}", flush=True)
        print(f"[MAIN] raw_dir: {RAW_DIR}", flush=True)
        print(f"[MAIN] render_dir: {RENDER_DIR}", flush=True)
    if args.render_tmp:
        RENDER_TMP_PARENT = os.path.abspath(os.path.expanduser(args.render_tmp))
    os.makedirs(RENDER_TMP_PARENT, exist_ok=True)
    print(f"[MAIN] render_tmp: {RENDER_TMP_PARENT}", flush=True)
    if args.shard_dir:
        SHARD_DIR = os.path.abspath(os.path.expanduser(args.shard_dir))
        print(f"[MAIN] shard_dir: {SHARD_DIR}", flush=True)
    NUM_GPUS = args.num_gpus if args.num_gpus is not None else _detect_num_gpus()
    print(f"[MAIN] NUM_GPUS={NUM_GPUS}", flush=True)

    if not os.path.isfile(BLENDER_BIN):
        print(f"[ERROR] Blender not found: {BLENDER_BIN}")
        sys.exit(1)

    if args.benchmark:
        benchmark(args)
    else:
        full_run(args)


if __name__ == "__main__":
    main()

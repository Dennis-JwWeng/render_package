#!/usr/bin/env python3
"""
Render-only pipeline: GLB → Blender CYCLES GPU render (no DINO).
8 workers, GPU-accelerated.
"""
import glob
import hashlib
import json
import os
import shutil
import sys
import tempfile
from multiprocessing import Process
from subprocess import call

import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "dataset_toolkits"))
sys.path.insert(0, REPO_ROOT)
from utils import sphere_hammersley_sequence

BLENDER_SCRIPT = os.path.join(REPO_ROOT, "dataset_toolkits", "blender_script", "render.py")


def find_blender():
    candidates = [
        os.path.join(REPO_ROOT, "blender-3.5.1-linux-x64", "blender"),
        "/usr/local/bin/blender",
        "/usr/bin/blender",
        shutil.which("blender"),
    ]
    for p in candidates:
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    raise RuntimeError("Blender not found")


def build_views(num_views, seed):
    rng = np.random.default_rng(seed)
    offset = (rng.random(), rng.random())
    views = []
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views, offset)
        views.append({"yaw": y, "pitch": p, "radius": 2, "fov": 40 / 180 * np.pi})
    return views


def render_worker(wid, gpu_id, tasks, blender_bin, num_views, out_base, use_cpu=False):
    n_ok, n_fail, n_skip = 0, 0, 0
    device_tag = "CPU" if use_cpu else f"GPU{gpu_id}"
    for gi, (glb_path, shard, stem) in enumerate(tasks):
        out_dir = os.path.join(out_base, shard, stem)
        done_marker = os.path.join(out_dir, "mesh.ply")
        if os.path.isfile(done_marker):
            n_skip += 1
            continue

        print(f"[R{wid}:{device_tag}] ({gi+1}/{len(tasks)}) {shard}/{stem}", flush=True)
        view_seed = int(hashlib.md5(stem.encode()).hexdigest()[:8], 16) % (2**31) or 1
        render_tmp = tempfile.mkdtemp(prefix=f"render_{stem[:8]}_")
        try:
            views = build_views(num_views, view_seed)
            args = [
                blender_bin, "-b", "-P", BLENDER_SCRIPT, "--",
                "--views", json.dumps(views),
                "--object", os.path.abspath(glb_path),
                "--resolution", "512",
                "--output_folder", render_tmp,
                "--engine", "CYCLES",
                "--save_mesh",
            ]
            if use_cpu:
                args.append("--use_cpu")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "" if use_cpu else str(gpu_id)
            ret = call(args, stdout=open(os.devnull, 'w'), stderr=open(os.devnull, 'w'), env=env)
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
                n_ok += 1
            else:
                print(f"  [R{wid}] FAIL {stem} (exit={ret})", flush=True)
                n_fail += 1
        except Exception as e:
            print(f"  [R{wid}] ERROR {stem}: {e}", flush=True)
            n_fail += 1
        finally:
            shutil.rmtree(render_tmp, ignore_errors=True)
    print(f"[R{wid}:{device_tag}] Done: ok={n_ok} fail={n_fail} skip={n_skip}", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_base", required=True)
    parser.add_argument("--out_base", required=True)
    parser.add_argument("--num_views", type=int, default=40)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--cpu", action="store_true", help="Use CPU rendering instead of GPU")
    args = parser.parse_args()

    blender_bin = find_blender()
    print(f"[INFO] Blender: {blender_bin}", flush=True)

    # Formats supported by dataset_toolkits/blender_script/render.py
    SUPPORTED_EXTS = ("glb", "gltf", "fbx", "obj", "stl", "ply",
                      "usd", "usda", "dae", "abc", "blend")

    shards = sorted([d for d in os.listdir(args.src_base)
                     if os.path.isdir(os.path.join(args.src_base, d))])
    all_items = []
    for shard in shards:
        shard_dir = os.path.join(args.src_base, shard)
        found = []
        for ext in SUPPORTED_EXTS:
            found.extend(glob.glob(os.path.join(shard_dir, f"*.{ext}")))
            found.extend(glob.glob(os.path.join(shard_dir, f"*.{ext.upper()}")))
        for path in sorted(set(found)):
            stem = os.path.splitext(os.path.basename(path))[0]
            all_items.append((path, shard, stem))

    device_str = "CPU" if args.cpu else f"{args.num_gpus} GPUs"
    ext_counts = {}
    for path, _, _ in all_items:
        e = os.path.splitext(path)[1].lower()
        ext_counts[e] = ext_counts.get(e, 0) + 1
    ext_summary = "  ".join(f"{e}×{n}" for e, n in sorted(ext_counts.items()))
    print(f"[INFO] {len(shards)} shards, {len(all_items)} models ({ext_summary})", flush=True)
    print(f"[INFO] {args.num_workers} render workers on {device_str}, {args.num_views} views", flush=True)

    n_rw = args.num_workers
    buckets = [[] for _ in range(n_rw)]
    for i, item in enumerate(all_items):
        buckets[i % n_rw].append(item)

    procs = []
    for wid in range(n_rw):
        if not buckets[wid]:
            continue
        gpu_id = wid % args.num_gpus
        p = Process(target=render_worker,
                    args=(wid, gpu_id, buckets[wid], blender_bin, args.num_views, args.out_base, args.cpu),
                    name=f"render-{wid}")
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
    print("[INFO] All renders done!", flush=True)


if __name__ == "__main__":
    main()

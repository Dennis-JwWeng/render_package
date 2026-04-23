#!/usr/bin/env python3
"""
Download files from Trellis500K HF repos (with manifest/ and shards/ structure) via HF mirror.

Usage:
    python3 download_trellis.py <repo_suffix> <start> <end> [options]

Examples:
    # Download files 0-99 (first 100) from trellis500k-github-archives-5
    python3 download_trellis.py 5 0 99

    # Download files 100-199
    python3 download_trellis.py 5 100 199

    # Only download manifest json files
    python3 download_trellis.py 5 0 99 --manifest-only

    # Only download shard tar.zst files
    python3 download_trellis.py 5 0 99 --shards-only

    # Custom workers and destination
    python3 download_trellis.py 5 0 49 --workers 8 --dest /my/path

    # Use a different repo name pattern
    python3 download_trellis.py 5 0 99 --repo-pattern "datasets/ShadesW/trellis500k-github-archives-{}"
"""
import argparse
import urllib.request
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

MIRROR = "https://hf-mirror.com"
BRANCH = "main"
CHUNK_SIZE = 8 * 1024 * 1024  # 8MB
UA = {'User-Agent': 'Mozilla/5.0'}


def api_list(repo, subdir):
    """List all files under a repo subdirectory via HF API."""
    url = f"{MIRROR}/api/{repo}/tree/{BRANCH}/{subdir}"
    req = urllib.request.Request(url, headers=UA)
    try:
        resp = urllib.request.urlopen(req, timeout=60)
        data = json.loads(resp.read())
    except Exception as e:
        print(f"[WARN] Failed to list {subdir}: {e}", flush=True)
        return []
    files = []
    for item in data:
        if item['type'] == 'file':
            files.append({'path': item['path'], 'size': item.get('size', 0)})
        elif item['type'] == 'directory':
            files.extend(api_list(repo, item['path']))
    return files


def download_file(repo, dest, filepath, expected_size, max_retries):
    url = f"{MIRROR}/{repo}/resolve/{BRANCH}/{filepath}"
    dest_path = os.path.join(dest, filepath)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    if os.path.exists(dest_path):
        local_size = os.path.getsize(dest_path)
        if expected_size <= 0 or local_size == expected_size:
            return filepath, "skipped", expected_size

    is_large = expected_size > 10 * 1024 * 1024  # >10MB uses resume logic
    if is_large:
        return _download_large(url, dest_path, filepath, expected_size, max_retries)
    else:
        return _download_small(url, dest_path, filepath, max_retries)


def _download_small(url, dest_path, filepath, max_retries):
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            resp = urllib.request.urlopen(req, timeout=120)
            data = resp.read()
            with open(dest_path, 'wb') as f:
                f.write(data)
            return filepath, "ok", len(data)
        except Exception as e:
            if attempt == max_retries:
                return filepath, f"FAILED: {e}", 0
            time.sleep(2 * attempt)


def _download_large(url, dest_path, filepath, expected_size, max_retries):
    tmp_path = dest_path + ".downloading"
    local_size = 0
    if os.path.exists(tmp_path):
        local_size = os.path.getsize(tmp_path)

    for attempt in range(1, max_retries + 1):
        try:
            headers = dict(UA)
            if local_size > 0:
                headers['Range'] = f'bytes={local_size}-'

            req = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req, timeout=300)

            status = resp.getcode()
            mode = 'ab' if status == 206 else 'wb'
            if status != 206:
                local_size = 0

            with open(tmp_path, mode) as f:
                while True:
                    chunk = resp.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
                    local_size += len(chunk)

            if expected_size > 0 and local_size != expected_size:
                return filepath, f"INCOMPLETE ({local_size}/{expected_size})", local_size

            os.rename(tmp_path, dest_path)
            return filepath, "ok", local_size

        except Exception as e:
            if attempt == max_retries:
                return filepath, f"FAILED after {max_retries} retries: {e}", local_size
            wait = min(10 * attempt, 60)
            print(f"  [retry {attempt}/{max_retries}] {filepath}: {e}, wait {wait}s...", flush=True)
            time.sleep(wait)


def fmt_size(n):
    for u in ['B', 'KB', 'MB', 'GB', 'TB']:
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024


def expected_shard_stems(repo: str, start: int, end: int) -> list[str]:
    """Sorted shard .tar.zst stems (no extension) for global index range [start, end] inclusive."""
    shard_files = [f for f in api_list(repo, "shards") if f["path"].endswith(".tar.zst")]
    shard_files.sort(key=lambda x: x["path"])
    selected = shard_files[start : end + 1]
    return [os.path.basename(t["path"]).replace(".tar.zst", "") for t in selected]


def build_download_task_list(
    repo: str,
    start: int,
    end: int,
    *,
    manifest_only: bool = False,
    shards_only: bool = False,
    skip_names: set | None = None,
) -> list[dict]:
    """Tasks compatible with run_download_tasks (each has path, size)."""
    all_tasks: list[dict] = []
    if not shards_only:
        manifest_files = api_list(repo, "manifest")
        manifest_files.sort(key=lambda x: x["path"])
        all_tasks.extend(manifest_files[start : end + 1])
    if not manifest_only:
        shard_files = api_list(repo, "shards")
        shard_files.sort(key=lambda x: x["path"])
        all_tasks.extend(shard_files[start : end + 1])
    if skip_names:
        def _keep(task):
            base = task["path"].rsplit("/", 1)[-1]
            stem = base.split(".", 1)[0]
            return base not in skip_names and stem not in skip_names
        all_tasks = [t for t in all_tasks if _keep(t)]
    return all_tasks


def run_download_tasks(
    repo: str,
    dest: str,
    tasks: list[dict],
    *,
    workers: int = 4,
    retries: int = 5,
    quiet: bool = False,
) -> dict:
    """Parallel download. Returns ok/skip/fail counts and downloaded_bytes."""
    if not tasks:
        return {"ok": 0, "skip": 0, "fail": 0, "downloaded_bytes": 0}
    ok_count = skip_count = fail_count = 0
    downloaded_bytes = 0
    start_time = time.time()
    n = len(tasks)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_file, repo, dest, t["path"], t["size"], retries): t
            for t in tasks
        }
        for i, future in enumerate(as_completed(futures), 1):
            filepath, status, size = future.result()
            if status == "ok":
                ok_count += 1
                downloaded_bytes += size
            elif "skipped" in status:
                skip_count += 1
            else:
                fail_count += 1
            if not quiet:
                elapsed = time.time() - start_time
                speed = downloaded_bytes / elapsed if elapsed > 0 else 0
                short_name = os.path.basename(filepath)
                print(
                    f"[{time.strftime('%H:%M:%S')}] [{i}/{n}] {status}: {short_name} "
                    f"({fmt_size(size)}) | Total: {fmt_size(downloaded_bytes)} | Speed: {fmt_size(speed)}/s",
                    flush=True,
                )
    return {
        "ok": ok_count,
        "skip": skip_count,
        "fail": fail_count,
        "downloaded_bytes": downloaded_bytes,
    }


def main():
    global MIRROR
    parser = argparse.ArgumentParser(
        description="Download Trellis500K repo files (manifest + shards) from HF mirror",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples:")[1] if "Examples:" in __doc__ else "",
    )
    parser.add_argument("suffix", help="Repo suffix number, e.g. 5 for trellis500k-github-archives-5")
    parser.add_argument("start", type=int, help="Start index (0-based, inclusive)")
    parser.add_argument("end", type=int, help="End index (inclusive)")
    parser.add_argument("--manifest-only", action="store_true", help="Only download manifest JSON files")
    parser.add_argument("--shards-only", action="store_true", help="Only download shard tar.zst files")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel download threads (default: 4)")
    parser.add_argument("--retries", type=int, default=5, help="Max retries per file (default: 5)")
    parser.add_argument("--dest", type=str, default=None, help="Destination directory (default: auto)")
    parser.add_argument("--repo-pattern", type=str, default="datasets/ShadesW/trellis500k-github-archives-{}",
                        help="Repo path pattern with {} for suffix")
    parser.add_argument("--mirror", type=str, default=MIRROR, help=f"HF mirror URL (default: {MIRROR})")
    parser.add_argument("--skip-file", type=str, default=None,
                        help="Path to a file listing basenames to skip (one per line). "
                             "Tasks whose basename or stem matches any line are excluded. "
                             "Handy for skipping shards already uploaded elsewhere.")
    args = parser.parse_args()

    skip_names = set()
    if args.skip_file:
        try:
            with open(args.skip_file) as fh:
                for ln in fh:
                    ln = ln.strip()
                    if not ln or ln.startswith('#'):
                        continue
                    skip_names.add(ln)
                    # also accept bare stem (e.g. shard_xxx) matching shard_xxx.tar.zst
                    if '.' in ln:
                        skip_names.add(ln.split('.', 1)[0])
        except OSError as e:
            print(f"[WARN] cannot read --skip-file {args.skip_file}: {e}", flush=True)
        if skip_names:
            print(f"[skip-file] loaded {len(skip_names)} skip entries from {args.skip_file}", flush=True)

    repo = args.repo_pattern.format(args.suffix)
    repo_name = repo.split("/")[-1]
    dest = args.dest or f"/data/hdd01/zsn/data/Trellis500K/{repo_name}"
    mirror = args.mirror

    print(f"{'='*60}", flush=True)
    print(f"  Repo:    {mirror}/{repo}", flush=True)
    print(f"  Range:   [{args.start}, {args.end}]", flush=True)
    print(f"  Dest:    {dest}", flush=True)
    print(f"  Workers: {args.workers}, Retries: {args.retries}", flush=True)
    dl_manifest = not args.shards_only
    dl_shards = not args.manifest_only
    types = []
    if dl_manifest:
        types.append("manifest")
    if dl_shards:
        types.append("shards")
    print(f"  Types:   {' + '.join(types)}", flush=True)
    print(f"{'='*60}", flush=True)

    _prev_mirror = MIRROR
    MIRROR = mirror
    try:
        print(f"[{time.strftime('%H:%M:%S')}] Building task list...", flush=True)
        all_tasks = build_download_task_list(
            repo,
            args.start,
            args.end,
            manifest_only=args.manifest_only,
            shards_only=args.shards_only,
            skip_names=skip_names or None,
        )
        if dl_manifest and not args.shards_only:
            mf = api_list(repo, "manifest")
            print(f"  manifest files in repo: {len(mf)}, range slice tasks: {sum(1 for t in all_tasks if 'manifest/' in t['path'])}", flush=True)
        if dl_shards and not args.manifest_only:
            sf = [f for f in api_list(repo, "shards") if f["path"].endswith(".tar.zst")]
            print(f"  shard archives in repo: {len(sf)}, range slice tasks: {sum(1 for t in all_tasks if t['path'].endswith('.tar.zst'))}", flush=True)

        if skip_names:
            print(f"[skip-file] tasks after filter: {len(all_tasks)}", flush=True)

        if not all_tasks:
            print("No files to download.", flush=True)
            return

        total_size = sum(t['size'] for t in all_tasks)
        print(f"\n[{time.strftime('%H:%M:%S')}] Downloading {len(all_tasks)} files, total {fmt_size(total_size)}\n", flush=True)

        t0 = time.time()
        stats = run_download_tasks(repo, dest, all_tasks, workers=args.workers, retries=args.retries, quiet=False)
        elapsed = time.time() - t0
        hours = elapsed / 3600
        print(f"\n{'='*60}")
        print(f"  Finished in {hours:.1f}h")
        print(f"  OK: {stats['ok']}  Skipped: {stats['skip']}  Failed: {stats['fail']}")
        print(f"  Downloaded: {fmt_size(stats['downloaded_bytes'])}")
        print(f"{'='*60}")
    finally:
        MIRROR = _prev_mirror


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Pack verified (fully encoded) render shards and upload .tar.zst to a Hugging Face dataset repo.

Flow: optional local tar.zst integrity read → upload → verify remote size matches → delete local
      pack (YAML: delete_local_archive_after_success). If Hub already has the file, compare
      sizes and optionally delete the duplicate local pack (delete_local_when_remote_exists).

Reads hf.upload (+ paths.pack_staging_dir) from the same YAML as the render/encode pipeline.
Auth: HUGGINGFACE_HUB_TOKEN / HF_TOKEN, or hf.upload.token (e.g. in <config>.local.yaml merged by load_config).

Usage:
  # Dry-run: show what would be packed/uploaded
  python upload_hf_encoded_shards.py --config config/trellis_github_archives_6_first100.yaml --dry-run

  # All shards classified encode_done under render_dir
  python upload_hf_encoded_shards.py --config config/default.yaml --all-verified

  # Named shards only
  python upload_hf_encoded_shards.py --config config/default.yaml --shards shard_2253590,shard_2253592

  # All shard_* under render_dir (e.g. render-only), pack+upload — use config/github_render_pack_upload.yaml
  python upload_hf_encoded_shards.py --config config/github_render_pack_upload.yaml --all-shard-dirs --include-unencoded --force

  # Pack locally only (no HF)
  python upload_hf_encoded_shards.py --config config/default.yaml --all-verified --pack-only
"""
from __future__ import annotations

import argparse
import fnmatch
import glob
import os
import shutil
import sys
import tarfile
import tempfile
RENDER_PKG = os.path.dirname(os.path.abspath(__file__))
if RENDER_PKG not in sys.path:
    sys.path.insert(0, RENDER_PKG)

import zstandard  # noqa: E402


def _verify_local_tar_zst(path: str) -> None:
    """Read full zstd-compressed tar: ensures archive is not corrupt."""
    dctx = zstandard.ZstdDecompressor()
    with open(path, "rb") as raw:
        reader = dctx.stream_reader(raw)
        with tarfile.open(fileobj=reader, mode="r|") as tar:
            for m in tar:
                if not m.isfile():
                    continue
                f = tar.extractfile(m)
                if f is None:
                    continue
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break


def _remote_size(api, repo_id: str, repo_type: str, path_in_repo: str) -> int | None:
    infos = api.get_paths_info(
        repo_id, paths=[path_in_repo], repo_type=repo_type, expand=True
    )
    if not infos:
        return None
    return getattr(infos[0], "size", None)


def _maybe_delete_local(path: str, do_delete: bool) -> None:
    if not do_delete or not os.path.isfile(path):
        return
    os.remove(path)
    print(f"[CLEANUP] removed local archive {path}", flush=True)


def delete_source_shard_tar_if_enabled(cfg: dict, shard_stem: str) -> None:
    """Remove downloaded source shards/github/<stem>.tar.zst when pipeline.delete_source_shard_tar is true."""
    pl = cfg.get("pipeline") or {}
    if not pl.get("delete_source_shard_tar", False):
        return
    shard_dir = cfg["paths"]["shard_dir"]
    zst = os.path.join(shard_dir, f"{shard_stem}.tar.zst")
    if os.path.isfile(zst):
        os.remove(zst)
        print(f"[CLEANUP] removed source shard download {zst}", flush=True)


def delete_render_shard_dir_if_enabled(cfg: dict, shard_stem: str) -> None:
    """Remove rendered shard directory after verified upload when enabled."""
    up = (cfg.get("hf") or {}).get("upload") or {}
    if not up.get("delete_render_dir_after_success", False):
        return
    render_dir = cfg["paths"]["render_dir"]
    shard_path = os.path.join(render_dir, shard_stem)
    if os.path.isdir(shard_path):
        shutil.rmtree(shard_path)
        print(f"[CLEANUP] removed render shard dir {shard_path}", flush=True)


def _iter_pack_members(
    shard_dir: str,
    includes: list[str],
    exclude_globs: list[str],
) -> list[tuple[str, str]]:
    """Return sorted list of (abs_path, arcname) relative to shard_dir."""
    shard_dir = os.path.abspath(shard_dir)
    out: list[tuple[str, str]] = []

    def excluded(rel_posix: str) -> bool:
        for pat in exclude_globs:
            if fnmatch.fnmatch(rel_posix, pat) or fnmatch.fnmatch(os.path.basename(rel_posix), pat):
                return True
        return False

    for obj in sorted(os.listdir(shard_dir)):
        obj_path = os.path.join(shard_dir, obj)
        if not os.path.isdir(obj_path):
            continue
        for inc in includes:
            p = os.path.join(obj_path, inc)
            if os.path.isfile(p):
                rel = f"{obj}/{inc}".replace("\\", "/")
                if not excluded(rel):
                    out.append((p, rel))
            elif os.path.isdir(p):
                for root, _, files in os.walk(p):
                    for f in sorted(files):
                        fp = os.path.join(root, f)
                        rel = os.path.relpath(fp, shard_dir).replace("\\", "/")
                        if not excluded(rel):
                            out.append((fp, rel))

    out.sort(key=lambda x: x[1])
    return out


def _split_members_by_object(
    members: list[tuple[str, str]],
    max_part_bytes: int,
) -> list[list[tuple[str, str]]]:
    """Split archive members on object boundaries using uncompressed size estimates."""
    if max_part_bytes <= 0:
        return [members]

    groups: list[list[tuple[str, str]]] = []
    current_obj: str | None = None
    current_group: list[tuple[str, str]] = []
    for abs_path, arcname in members:
        obj = arcname.split("/", 1)[0]
        if current_obj is None:
            current_obj = obj
        if obj != current_obj:
            groups.append(current_group)
            current_group = []
            current_obj = obj
        current_group.append((abs_path, arcname))
    if current_group:
        groups.append(current_group)

    parts: list[list[tuple[str, str]]] = []
    current_part: list[tuple[str, str]] = []
    current_size = 0
    for group in groups:
        group_size = sum(os.path.getsize(abs_path) for abs_path, _ in group)
        if current_part and current_size + group_size > max_part_bytes:
            parts.append(current_part)
            current_part = []
            current_size = 0
        current_part.extend(group)
        current_size += group_size
    if current_part:
        parts.append(current_part)
    return parts


def _archive_name(shard_name: str, part_idx: int, part_count: int) -> str:
    if part_count == 1:
        return f"{shard_name}.tar.zst"
    return f"{shard_name}.part_{part_idx:05d}.tar.zst"


def _plan_archives(
    shard_name: str,
    shard_path: str,
    includes: list[str],
    exclude_globs: list[str],
    max_part_bytes: int,
) -> list[tuple[str, list[tuple[str, str]]]]:
    members = _iter_pack_members(shard_path, includes, exclude_globs)
    if not members:
        return []
    parts = _split_members_by_object(members, max_part_bytes)
    return [
        (_archive_name(shard_name, i, len(parts)), part_members)
        for i, part_members in enumerate(parts)
    ]


def _existing_local_archives(pack_dir: str, shard_name: str) -> list[str]:
    part_paths = sorted(glob.glob(os.path.join(pack_dir, f"{shard_name}.part_*.tar.zst")))
    if part_paths:
        return [os.path.basename(p) for p in part_paths]
    single = os.path.join(pack_dir, f"{shard_name}.tar.zst")
    if os.path.isfile(single):
        return [os.path.basename(single)]
    return []


def _write_tar_zst(
    members: list[tuple[str, str]],
    out_path: str,
    zstd_level: int,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    cctx = zstandard.ZstdCompressor(level=zstd_level)
    with open(out_path, "wb") as raw:
        with cctx.stream_writer(raw) as compressor:
            with tarfile.open(fileobj=compressor, mode="w|", format=tarfile.GNU_FORMAT) as tar:
                for abs_path, arcname in members:
                    tar.add(abs_path, arcname=arcname, recursive=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack encoded shards to tar.zst and upload to HF")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--shards", type=str, default=None, help="Comma-separated shard dir names under render_dir")
    parser.add_argument("--all-verified", action="store_true", help="Every encode_done shard under render_dir")
    parser.add_argument(
        "--all-shard-dirs",
        action="store_true",
        help="Every top-level shard_* directory under render_dir (pair with --include-unencoded if not yet encode_done)",
    )
    parser.add_argument(
        "--include-unencoded",
        action="store_true",
        help="Do not require encode_done before pack/upload (e.g. render output with images/mesh, no latents yet)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pack-only", action="store_true")
    parser.add_argument("--upload-only", action="store_true", help="Assume .tar.zst already in pack_staging_dir")
    parser.add_argument("--force", action="store_true", help="Ignore hf.upload.enabled: false")
    parser.add_argument("--no-skip-remote", action="store_true", help="Upload even if file already on Hub")
    parser.add_argument(
        "--keep-local",
        action="store_true",
        help="Do not delete local .tar.zst after successful verify+upload (overrides YAML)",
    )
    args = parser.parse_args()

    from encoders.config import hf_hub_token, load_config

    cfg = load_config(args.config)
    up = (cfg.get("hf") or {}).get("upload") or {}

    will_upload = not args.dry_run and not args.pack_only
    if will_upload or args.upload_only:
        if not args.force and not up.get("enabled", False):
            print("[ERROR] hf.upload.enabled is false. Set enabled: true or pass --force.", flush=True)
            sys.exit(1)
        if not up.get("repo_id"):
            print("[ERROR] hf.upload.repo_id is empty in YAML.", flush=True)
            sys.exit(1)

    paths = cfg["paths"]
    render_dir = paths["render_dir"]
    pack_dir = paths.get("pack_staging_dir") or os.path.join(paths["data_root"], "github/pack_for_upload")
    pack_dir = os.path.abspath(os.path.expanduser(pack_dir))
    os.makedirs(pack_dir, exist_ok=True)

    repo_id = up.get("repo_id") or ""
    repo_type = up.get("repo_type", "dataset")
    path_in_repo = up.get("path_in_repo", "encoded/shards").strip("/").replace("\\", "/")
    arch = up.get("archive") or {}
    includes: list[str] = list(arch.get("include") or ["latents", "transforms.json", "mesh.ply"])
    exclude_globs: list[str] = list(arch.get("exclude_globs") or [])
    zstd_level = int(arch.get("zstd_level", 10))
    max_part_bytes = int(arch.get("max_part_bytes") or 0)
    verify_local = bool(up.get("verify_local_archive_before_upload", True))
    verify_remote = bool(up.get("verify_remote_after_upload", True))
    delete_after = bool(up.get("delete_local_archive_after_success", True)) and not args.keep_local
    delete_when_skipped = bool(up.get("delete_local_when_remote_exists", True)) and not args.keep_local

    from encoders import ALL_STAGES
    from huggingface_hub import HfApi
    from run_pipeline import _classify_shards

    stages = [s for s in ALL_STAGES if cfg.get("stages", {}).get(s, True)]

    if args.shards:
        names = [s.strip() for s in args.shards.split(",") if s.strip()]
    elif args.all_verified:
        status_once = _classify_shards(render_dir, stages, cfg)
        names = sorted(k for k, v in status_once.items() if v == "encode_done")
        if not names:
            print("[WARN] No encode_done shards found.", flush=True)
            return
    elif args.all_shard_dirs:
        if not os.path.isdir(render_dir):
            print(f"[ERROR] render_dir is not a directory: {render_dir}", flush=True)
            sys.exit(1)
        names = sorted(
            n
            for n in os.listdir(render_dir)
            if n.startswith("shard_") and os.path.isdir(os.path.join(render_dir, n))
        )
        if not names:
            print("[WARN] No shard_* directories under render_dir.", flush=True)
            return
    else:
        print("[ERROR] Pass --shards, --all-verified, or --all-shard-dirs.", flush=True)
        sys.exit(1)

    status_all = _classify_shards(render_dir, stages, cfg)

    hf_tok = hf_hub_token(cfg)
    api = HfApi(token=hf_tok)
    remote_set: set[str] | None = None
    if will_upload and not args.no_skip_remote:
        print(f"[HF] Listing remote files in {repo_id} ({repo_type}) ...", flush=True)
        remote_set = set(api.list_repo_files(repo_id, repo_type=repo_type))

    for name in names:
        shard_path = os.path.join(render_dir, name)
        if not os.path.isdir(shard_path):
            print(f"[SKIP] missing {shard_path}", flush=True)
            continue
        if not args.include_unencoded and status_all.get(name) != "encode_done":
            print(f"[SKIP] {name} not encode_done (status={status_all.get(name, 'missing')})", flush=True)
            continue

        if args.upload_only:
            archive_plans = [(n, []) for n in _existing_local_archives(pack_dir, name)]
        else:
            archive_plans = _plan_archives(
                name,
                shard_path,
                includes,
                exclude_globs,
                max_part_bytes,
            )
            if not archive_plans:
                print(f"[SKIP] {name}: no files matched include/exclude rules", flush=True)
                continue
        if not archive_plans:
            print(f"[SKIP] no local archive(s) for {name}", flush=True)
            continue

        shard_ok = True
        for part_idx, (archive_name, members) in enumerate(archive_plans, 1):
            local_archive = os.path.join(pack_dir, archive_name)
            remote_rel = f"{path_in_repo}/{archive_name}"

            if args.upload_only:
                if not os.path.isfile(local_archive):
                    print(f"[SKIP] no local archive {local_archive}", flush=True)
                    shard_ok = False
                    break
            else:
                print(
                    f"[PACK] {name} part {part_idx}/{len(archive_plans)} "
                    f"({len(members)} files) -> {local_archive}",
                    flush=True,
                )
                if args.dry_run:
                    for _, arc in members[:8]:
                        print(f"       + {arc}", flush=True)
                    if len(members) > 8:
                        print(f"       ... +{len(members) - 8} more", flush=True)
                    continue
                tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tar.zst", dir=pack_dir)
                os.close(tmp_fd)
                try:
                    _write_tar_zst(members, tmp_path, zstd_level)
                    os.replace(tmp_path, local_archive)
                except Exception:
                    os.unlink(tmp_path)
                    raise

            if args.dry_run or args.pack_only:
                continue

            if verify_local:
                print(f"[VERIFY] local archive {local_archive}", flush=True)
                try:
                    _verify_local_tar_zst(local_archive)
                except Exception as e:
                    print(f"[ERROR] local archive failed integrity read: {e}", flush=True)
                    raise

            local_sz = os.path.getsize(local_archive)

            if remote_set is not None and remote_rel in remote_set:
                print(f"[SKIP] remote exists {remote_rel}", flush=True)
                rsz = _remote_size(api, repo_id, repo_type, remote_rel) if verify_remote else None
                remote_matches = verify_remote and rsz is not None and rsz == local_sz
                if verify_remote and rsz is None:
                    print(f"[WARN] could not stat remote {remote_rel}; keeping local pack and source .tar.zst", flush=True)
                    shard_ok = False
                elif verify_remote and rsz is not None and rsz != local_sz:
                    print(
                        f"[WARN] remote size {rsz} != local {local_sz}; keeping local pack and source .tar.zst",
                        flush=True,
                    )
                    shard_ok = False
                elif delete_when_skipped and remote_matches:
                    _maybe_delete_local(local_archive, True)
                elif delete_when_skipped and not verify_remote:
                    _maybe_delete_local(local_archive, True)
                continue

            print(f"[UPLOAD] {local_archive} -> {repo_id} / {remote_rel}", flush=True)
            api.upload_file(
                path_or_fileobj=local_archive,
                path_in_repo=remote_rel,
                repo_id=repo_id,
                repo_type=repo_type,
                revision=up.get("revision") or "main",
                commit_message=f"Add {archive_name}",
            )

            rsz_uploaded: int | None = None
            if verify_remote:
                rsz_uploaded = _remote_size(api, repo_id, repo_type, remote_rel)
                if rsz_uploaded is None:
                    raise RuntimeError(f"Upload finished but remote path missing: {remote_rel}")
                if rsz_uploaded != local_sz:
                    raise RuntimeError(
                        f"Remote size {rsz_uploaded} != local {local_sz} after upload for {remote_rel}"
                    )
                print(f"[VERIFY] remote OK size={rsz_uploaded}", flush=True)

            _maybe_delete_local(local_archive, delete_after)
            if verify_remote and rsz_uploaded != local_sz:
                shard_ok = False

        if args.dry_run or args.pack_only:
            continue

        if shard_ok:
            delete_source_shard_tar_if_enabled(cfg, name)
            if verify_remote:
                delete_render_shard_dir_if_enabled(cfg, name)
        else:
            print(f"[WARN] {name}: not all archive part(s) verified; keeping source/render", flush=True)


if __name__ == "__main__":
    main()

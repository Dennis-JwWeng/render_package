"""
Shared utilities: voxelization, projection compat shims, atomic I/O.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np


def ensure_utils3d(cfg: dict):
    """Make sure utils3d is importable."""
    try:
        import utils3d
        import utils3d.torch
        return
    except ImportError:
        pass
    unilat_parent = cfg.get("third_party", {}).get("unilat3d")
    if unilat_parent:
        p = str(Path(unilat_parent).resolve().parent)
        if p not in sys.path:
            sys.path.insert(0, p)
    import utils3d
    import utils3d.torch


def intrinsics_from_fov_xy(fov_x, fov_y):
    """Compat shim across utils3d versions."""
    import utils3d.torch
    try:
        return utils3d.torch.intrinsics_from_fov_xy(fov_x, fov_y)
    except AttributeError:
        from utils3d.torch.transforms import intrinsics_from_fov
        return intrinsics_from_fov(fov_x=fov_x, fov_y=fov_y)


def project_cv(points, extrinsics, intrinsics):
    """Compat shim for utils3d.torch.project_cv across versions."""
    import utils3d.torch
    if hasattr(utils3d.torch, "intrinsics_from_fov_xy"):
        return utils3d.torch.project_cv(points, extrinsics, intrinsics)
    from utils3d.torch.transforms import project_cv as _project_cv
    return _project_cv(points, intrinsics, extrinsics)


def voxelize_mesh(mesh_path: str, grid_size: int = 64):
    """Voxelize mesh.ply on a 64^3 grid in [-0.5, 0.5]^3.
    Returns (voxel_indices [N,3] int64, positions [N,3] float64).
    Runs open3d in a subprocess to survive segfaults on corrupt meshes."""
    import subprocess

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".npz")
    os.close(tmp_fd)
    try:
        code = (
            "import open3d as o3d, numpy as np\n"
            f"mesh = o3d.io.read_triangle_mesh({mesh_path!r})\n"
            "if mesh.is_empty() or not mesh.has_triangles():\n"
            f"    raise RuntimeError('Empty or invalid mesh: {mesh_path}')\n"
            "vertices = np.clip(np.asarray(mesh.vertices), -0.5+1e-6, 0.5-1e-6)\n"
            "mesh.vertices = o3d.utility.Vector3dVector(vertices)\n"
            "vg = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(\n"
            f"    mesh, voxel_size=1.0/{grid_size},\n"
            "    min_bound=(-0.5,-0.5,-0.5), max_bound=(0.5,0.5,0.5))\n"
            "voxels = vg.get_voxels()\n"
            "if len(voxels) == 0:\n"
            f"    raise RuntimeError('Voxelization produced zero voxels: {mesh_path}')\n"
            "vox = np.array([v.grid_index for v in voxels], dtype=np.int64)\n"
            "positions = (vox + 0.5) / float({grid}) - 0.5\n"
            f"np.savez_compressed({tmp_path!r}, vox=vox, positions=positions)\n"
        ).replace("{grid}", str(grid_size))

        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, timeout=120, text=True,
        )
        if result.returncode != 0:
            sig = -result.returncode if result.returncode < 0 else result.returncode
            stderr_tail = (result.stderr or "").strip()[-300:]
            raise RuntimeError(
                f"Voxelization subprocess failed (code {sig}): {mesh_path}\n{stderr_tail}"
            )
        data = np.load(tmp_path)
        vox = data["vox"]
        positions = data["positions"]
        return vox, positions
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def atomic_save_npz(path: str, **arrays):
    """Write npz atomically (temp dir + rename) to avoid corruption on crash."""
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(dir=d)
    tmp_npz = os.path.join(tmp_dir, "data.npz")
    try:
        np.savez_compressed(tmp_npz, **arrays)
        os.replace(tmp_npz, path)
    except BaseException:
        if os.path.exists(tmp_npz):
            os.unlink(tmp_npz)
        raise
    finally:
        if os.path.isdir(tmp_dir):
            os.rmdir(tmp_dir)

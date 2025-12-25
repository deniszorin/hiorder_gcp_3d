"""Compare potential evaluation between dict and indexed connectivity."""

from __future__ import annotations

import numpy as np

import geometry
import geometry_idx
import potential
import potential_idx
from viz import build_validation_scenes, generate_reg_mesh, sample_volume_grid


def _to_idx_mesh(mesh: geometry.MeshData) -> geometry_idx.MeshData:
    return geometry_idx.MeshData(V=mesh.V, F=mesh.F)


def _sample_grid(mesh: geometry.MeshData, resolution: int = 30, padding: float = 0.5) -> np.ndarray:
    bounds_min = mesh.V.min(axis=0) - padding
    bounds_max = mesh.V.max(axis=0) + padding
    return sample_volume_grid(bounds_min, bounds_max, resolution=resolution)


def _eval_potential_dict(mesh: geometry.MeshData, q: np.ndarray) -> np.ndarray:
    conn = geometry.build_connectivity(mesh)
    geom = geometry.precompute_face_geometry(mesh)
    comps = potential.smoothed_offset_potential(q, mesh, conn, geom)
    return comps.face + comps.edge + comps.vertex


def _eval_potential_idx(mesh: geometry.MeshData, q: np.ndarray) -> np.ndarray:
    mesh_idx = _to_idx_mesh(mesh)
    conn = geometry_idx.build_connectivity(mesh_idx)
    geom = geometry_idx.precompute_face_geometry(mesh_idx)
    comps = potential_idx.smoothed_offset_potential(q, mesh_idx, conn, geom)
    return comps.face + comps.edge + comps.vertex


def main() -> None:
    meshes = []

    # From run_validation_visualizations (viz.build_validation_scenes)
    for idx, mesh in enumerate(build_validation_scenes()):
        meshes.append((f"validation_scene_{idx}", mesh))

    # From pyvista_isosurfaces.py
    meshes.append(("reg_flat_n2", generate_reg_mesh(lambda u, v: [u, v, 0.0], n=2)))
    meshes.append(
        (
            "reg_paraboloid_n8",
            generate_reg_mesh(lambda u, v: [u, v, (u - 0.501) ** 2 + (v - 0.501) ** 2], n=8),
        )
    )
    meshes.append(("reg_corner_n2", generate_reg_mesh(lambda u, v: [u, v, abs(10.01 * u - 5.01)], n=2)))
    meshes.append(("bunny_lowpoly", geometry.load_obj_mesh("tests/Bunny-LowPoly.obj")))

    for name, mesh in meshes:
        q = _sample_grid(mesh, resolution=30, padding=0.5)
        vals_dict = _eval_potential_dict(mesh, q)
        vals_idx = _eval_potential_idx(mesh, q)
        diff = vals_dict - vals_idx
        l2 = np.linalg.norm(diff)
        linf = np.max(np.abs(diff))
        print(f"{name}: L2={l2:.6e} Linf={linf:.6e}")


if __name__ == "__main__":
    main()

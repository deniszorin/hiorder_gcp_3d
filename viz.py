"""Visualization helpers for potentials."""

from __future__ import annotations

from typing import List, Optional, Sequence
import os

import numpy as np

from geometry import MeshData
from potential import smoothed_offset_potential


ArrayF = np.ndarray


def sample_plane_grid(
    p: ArrayF,
    n: ArrayF,
    extent: float,
    resolution: int,
) -> ArrayF:
    """Sample a grid of points on a plane.

    Returns array of shape (nq, 3).
    """

    p = np.asarray(p, dtype=float)
    n = np.asarray(n, dtype=float)
    n_norm = np.linalg.norm(n)
    if n_norm == 0.0:
        raise ValueError("Plane normal must be non-zero.")
    n = n / n_norm

    if abs(n[2]) < 0.9:
        ref = np.array([0.0, 0.0, 1.0])
    else:
        ref = np.array([1.0, 0.0, 0.0])
    u = np.cross(n, ref)
    u_norm = np.linalg.norm(u)
    if u_norm == 0.0:
        raise ValueError("Failed to build plane basis.")
    u = u / u_norm
    v = np.cross(n, u)

    coords = np.linspace(-extent, extent, resolution)
    uu, vv = np.meshgrid(coords, coords, indexing="xy")
    grid = p[None, None, :] + uu[..., None] * u + vv[..., None] * v
    return grid.reshape(-1, 3)


def visualize_isolines_on_plane(
    q_grid: ArrayF,
    values: ArrayF,
    resolution: int,
    levels: Optional[Sequence[float]] = None,
    output_path: Optional[str] = None,
):
    """Plot isolines on a plane using Plotly."""

    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - visualization dependency
        raise ImportError("plotly is required for visualization.") from exc

    z = values.reshape(resolution, resolution)
    z = np.where(np.isfinite(z), z, np.nan)
    coords = np.arange(resolution)

    fig = go.Figure()
    if levels is None:
        fig.add_trace(
            go.Contour(
                z=z,
                x=coords,
                y=coords,
                contours=dict(showlabels=True),
                colorscale="Viridis",
                contours_coloring="lines",
            )
        )
    else:
        for level in levels:
            fig.add_trace(
                go.Contour(
                    z=z,
                    x=coords,
                    y=coords,
                    contours=dict(type="constraint", operation="=", value=level, showlabels=True),
                    line=dict(width=2),
                    showscale=False,
                )
            )

    if output_path is not None:
        fig.write_html(f"{output_path}.html")
    else:
        fig.show()

    return fig


def sample_volume_grid(
    bounds_min: ArrayF,
    bounds_max: ArrayF,
    resolution: int,
) -> ArrayF:
    """Sample a 3D grid within bounds.

    Returns array of shape (nq, 3).
    """

    bounds_min = np.asarray(bounds_min, dtype=float)
    bounds_max = np.asarray(bounds_max, dtype=float)
    xs = np.linspace(bounds_min[0], bounds_max[0], resolution)
    ys = np.linspace(bounds_min[1], bounds_max[1], resolution)
    zs = np.linspace(bounds_min[2], bounds_max[2], resolution)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="xy")
    grid = np.stack([xx, yy, zz], axis=-1)
    return grid.reshape(-1, 3)


def visualize_isosurface(
    q_grid: ArrayF,
    values: ArrayF,
    resolution: int,
    levels: Optional[Sequence[float]] = None,
    clip_origin: Optional[ArrayF] = None,
    clip_normal: Optional[ArrayF] = None,
    output_path: Optional[str] = None,
):
    """Plot an isosurface using PyVista."""

    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - visualization dependency
        raise ImportError("pyvista is required for visualization.") from exc

    values = values.reshape(resolution, resolution, resolution)
    values = np.where(np.isfinite(values), values, np.nan)
    log_values = np.log10(np.maximum(values, 1e-12))

    grid = pv.ImageData()
    grid.dimensions = (resolution, resolution, resolution)
    bounds_min = q_grid.min(axis=0)
    bounds_max = q_grid.max(axis=0)
    grid.origin = bounds_min
    grid.spacing = (
        (bounds_max[0] - bounds_min[0]) / (resolution - 1),
        (bounds_max[1] - bounds_min[1]) / (resolution - 1),
        (bounds_max[2] - bounds_min[2]) / (resolution - 1),
    )
    grid.point_data["potential"] = values.ravel(order="F")
    grid.point_data["log_potential"] = log_values.ravel(order="F")

    if levels is None:
        levels = [1000.0]
    iso = grid.contour(levels, scalars="potential")
    iso = iso.sample(grid)
    if clip_origin is not None and clip_normal is not None:
        iso = iso.clip(normal=clip_normal, origin=clip_origin)

    p = pv.Plotter()
    p.add_mesh(iso, scalars="log_potential", cmap="viridis", opacity=1.0)

    if output_path is not None:
        html_path = output_path if output_path.endswith(".html") else f"{output_path}.html"
        try:
            p.export_html(html_path)
        except ImportError:
            base, _ = os.path.splitext(html_path)
            iso.save(f"{base}.vtp")
        p.close()
    else:
        p.show()

    return p


def build_validation_scenes() -> List[MeshData]:
    """Construct meshes for validation scenarios."""

    scenes: List[MeshData] = []

    V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    F = np.array([[0, 1, 2]], dtype=int)
    scenes.append(MeshData(V=V, F=F))

    V = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    F = np.array([[0, 1, 2], [1, 0, 3]], dtype=int)
    scenes.append(MeshData(V=V, F=F))

    center = np.array([0.0, 0.0, 0.0])
    angles = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
    ring = np.stack(
        [
            np.cos(angles),
            np.sin(angles),
            np.array([1.0, -1.0, 1.0, -1.0]),
        ],
        axis=1,
    )
    V = np.vstack([center, ring])
    F = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]], dtype=int)
    scenes.append(MeshData(V=V, F=F))

    ring = np.stack([np.cos(angles), np.sin(angles), -np.ones(4)], axis=1)
    V = np.vstack([center, ring])
    F = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]], dtype=int)
    scenes.append(MeshData(V=V, F=F))

    return scenes


def run_validation_visualizations(output_dir: Optional[str] = None) -> None:
    """Generate all described visualizations for validation."""

    from geometry import build_connectivity, precompute_face_geometry

    scenes = build_validation_scenes()
    level_values = [10.0, 100.0, 200.0, 500.0, 1000.0]
    levels_2d = np.log10(level_values)
    levels_3d = level_values

    for idx, mesh in enumerate(scenes):
        connectivity = build_connectivity(mesh)
        geom = precompute_face_geometry(mesh)

        p = mesh.V[0]
        edge_dir = mesh.V[1] - mesh.V[0]
        edge_norm = np.linalg.norm(edge_dir)
        if edge_norm == 0.0:
            n = np.array([1.0, 0.0, 0.0])
        else:
            n = edge_dir / edge_norm
        q_plane = sample_plane_grid(p, n, extent=1.5, resolution=100)
        comps = smoothed_offset_potential(q_plane, mesh, connectivity, geom)
        output_path = None
        if output_dir is not None:
            output_path = f"{output_dir}/scene_{idx}_isolines.png"
        values = comps.face + comps.edge + comps.vertex
        log_values = np.log10(np.maximum(values, 1e-12))
        visualize_isolines_on_plane(
            q_plane,
            log_values,
            resolution=100,
            levels=levels_2d,
            output_path=output_path,
        )

        bounds_min = mesh.V.min(axis=0) - 0.5
        bounds_max = mesh.V.max(axis=0) + 0.5
        q_vol = sample_volume_grid(bounds_min, bounds_max, resolution=200)
        comps = smoothed_offset_potential(q_vol, mesh, connectivity, geom)
        output_path = None
        if output_dir is not None:
            output_path = f"{output_dir}/scene_{idx}_isosurface.html"
        center = 0.5 * (bounds_min + bounds_max)
        clip_normal = geom.normals[0]
        visualize_isosurface(
            q_vol,
            comps.face + comps.edge + comps.vertex,
            resolution=200,
            levels=levels_3d,
            clip_origin=center,
            clip_normal=clip_normal,
            output_path=output_path,
        )

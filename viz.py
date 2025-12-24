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
    log_levels = np.log10(np.maximum(np.asarray(levels, dtype=float), 1e-12))
    iso = grid.contour(log_levels.tolist(), scalars="log_potential")
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


def isosurface_with_clip(
    mesh: MeshData,
    levels: Optional[Sequence[float]] = None,
    res: int = 100,
    parallel=False,
):
    """Interactive PyVista isosurfaces with a clipping plane widget."""

    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - visualization dependency
        raise ImportError("pyvista is required for visualization.") from exc

    from geometry import build_connectivity, precompute_face_geometry

    conn = build_connectivity(mesh)
    geom = precompute_face_geometry(mesh)

    bounds_min = mesh.V.min(axis=0) - 0.5
    bounds_max = mesh.V.max(axis=0) + 0.5

    xs = np.linspace(bounds_min[0], bounds_max[0], res)
    ys = np.linspace(bounds_min[1], bounds_max[1], res)
    zs = np.linspace(bounds_min[2], bounds_max[2], res)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

    comps = smoothed_offset_potential(points, mesh, conn, geom, 
                                      parallel=parallel)
    values = comps.face + comps.edge + comps.vertex
    log_values = np.log10(np.maximum(values, 1e-12))

    grid = pv.ImageData()
    grid.dimensions = (res, res, res)
    grid.origin = bounds_min
    grid.spacing = (
        (bounds_max[0] - bounds_min[0]) / (res - 1),
        (bounds_max[1] - bounds_min[1]) / (res - 1),
        (bounds_max[2] - bounds_min[2]) / (res - 1),
    )
    grid.point_data["potential"] = values.reshape(res, res, res).ravel(order="F")
    grid.point_data["log_potential"] = log_values.reshape(res, res, res).ravel(order="F")

    if levels is None:
        levels = [1000.0]
    log_levels = np.log10(np.maximum(np.asarray(levels, dtype=float), 1e-12))
    iso = grid.contour(log_levels.tolist(), scalars="log_potential")

    p = pv.Plotter()
    p.add_mesh(iso, scalars="log_potential", cmap="viridis", opacity=1.0, name="clip")

    def clip_callback(normal, origin):
        clipped = iso.clip(normal=normal, origin=origin)
        p.add_mesh(
            clipped,
            scalars="log_potential",
            cmap="viridis",
            opacity=1.0,
            name="clip",
            reset_camera=False,
        )

    center = 0.5 * (bounds_min + bounds_max)
    plane_widget = p.add_plane_widget(
        callback=clip_callback,
        origin=center,
        normal=geom.normals[0],
    )

    def toggle_plane(flag):
        plane_widget.SetEnabled(flag)

    p.add_checkbox_button_widget(toggle_plane, value=True, position=(10, 10))
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
    levels_3d = np.log10(level_values)

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

        bounds_min = mesh.V.min(axis=0) - 2.0
        bounds_max = mesh.V.max(axis=0) + 2.0
        q_vol = sample_volume_grid(bounds_min, bounds_max, resolution=100)
        comps = smoothed_offset_potential(q_vol, mesh, connectivity, geom)
        output_path = None
        if output_dir is not None:
            output_path = f"{output_dir}/scene_{idx}_isosurface.html"
        center = 0.5 * (bounds_min + bounds_max)
        clip_normal = geom.normals[0]
        visualize_isosurface(
            q_vol,
            comps.face + comps.edge + comps.vertex,
            resolution=100,
            levels=levels_3d,
            clip_origin=center,
            clip_normal=clip_normal,
            output_path=output_path,
        )


def _lagrange_control_uv() -> np.ndarray:
    """Return equispaced (u, v) pairs for cubic Lagrange triangle control points."""

    return np.array(
        [
            [1.0, 0.0],  # (3,0,0)
            [0.0, 1.0],  # (0,3,0)
            [0.0, 0.0],  # (0,0,3)
            [2.0 / 3.0, 1.0 / 3.0],  # (2,1,0)
            [2.0 / 3.0, 0.0],  # (2,0,1)
            [1.0 / 3.0, 2.0 / 3.0],  # (1,2,0)
            [0.0, 2.0 / 3.0],  # (0,2,1)
            [1.0 / 3.0, 0.0],  # (1,0,2)
            [0.0, 1.0 / 3.0],  # (0,1,2)
            [1.0 / 3.0, 1.0 / 3.0],  # (1,1,1)
        ],
        dtype=float,
    )


def _lagrange_vandermonde_inv() -> np.ndarray:
    """Precompute the inverse Vandermonde for cubic Lagrange nodes."""

    uv = _lagrange_control_uv()
    V = np.zeros((uv.shape[0], 10), dtype=float)
    for i, (u, v) in enumerate(uv):
        w = 1.0 - u - v
        V[i] = [
            u**3,
            v**3,
            w**3,
            u**2 * v,
            u**2 * w,
            u * v**2,
            v**2 * w,
            u * w**2,
            v * w**2,
            u * v * w,
        ]
    return np.linalg.inv(V)


_LAGRANGE_V_INV = _lagrange_vandermonde_inv()


def _lagrange_triangle_eval(u: float, v: float, control_points: ArrayF) -> ArrayF:
    """Evaluate a cubic Lagrange triangle at (u, v) on the unit triangle."""

    w = 1.0 - u - v
    if w < -1e-8:
        raise ValueError("Point (u, v) is outside the unit triangle.")
    u = max(u, 0.0)
    v = max(v, 0.0)
    w = max(w, 0.0)
    monomials = np.array(
        [
            u**3,
            v**3,
            w**3,
            u**2 * v,
            u**2 * w,
            u * v**2,
            v**2 * w,
            u * w**2,
            v * w**2,
            u * v * w,
        ],
        dtype=float,
    )
    coeffs = _LAGRANGE_V_INV @ control_points
    return monomials @ coeffs


def _refine_triangle_uv(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Refine the unit triangle into n^2 triangles in (u, v)."""

    if n < 1:
        raise ValueError("n must be >= 1.")

    verts = []
    index = {}
    for i in range(n + 1):
        for j in range(n + 1 - i):
            u = i / n
            v = j / n
            index[(i, j)] = len(verts)
            verts.append([u, v])

    faces = []
    for i in range(n):
        for j in range(n - i):
            a = index[(i, j)]
            b = index[(i + 1, j)]
            c = index[(i, j + 1)]
            faces.append([a, b, c])
            if j + 1 <= n - i - 1:
                d = index[(i + 1, j + 1)]
                faces.append([b, d, c])

    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=int)


def sample_lagrange_patch(
    control_points: ArrayF,
    n: int,
) -> MeshData:
    """Sample a cubic Lagrange triangle patch into a mesh."""

    control_points = np.asarray(control_points, dtype=float)
    if control_points.shape != (10, 3):
        raise ValueError("control_points must have shape (10, 3).")

    uv, faces = _refine_triangle_uv(n)
    verts = np.zeros((uv.shape[0], 3), dtype=float)
    for idx, (u, v) in enumerate(uv):
        verts[idx] = _lagrange_triangle_eval(u, v, control_points)
    return MeshData(V=verts, F=faces)


def generate_lagrange_patch_samples(
    control_points: ArrayF,
    n: int,
    resolution: int,
    pad: float = 0.5,
):
    """Generate volumetric samples for a cubic Bezier patch."""

    mesh = sample_lagrange_patch(control_points, n)
    from geometry import build_connectivity, precompute_face_geometry

    connectivity = build_connectivity(mesh)
    geom = precompute_face_geometry(mesh)
    bounds_min = mesh.V.min(axis=0) - pad
    bounds_max = mesh.V.max(axis=0) + pad
    q_vol = sample_volume_grid(bounds_min, bounds_max, resolution=resolution)
    comps = smoothed_offset_potential(q_vol, mesh, connectivity, geom)
    values = comps.face + comps.edge + comps.vertex
    return q_vol, values, mesh, geom


def visualize_lagrange_patch_potential(
    control_points: ArrayF,
    n: int,
    resolution: int,
    levels: Optional[Sequence[float]] = None,
    output_path: Optional[str] = None,
):
    """Visualize the potential for a sampled Lagrange patch."""

    q_vol, values, mesh, geom = generate_lagrange_patch_samples(control_points, n, resolution)
    bounds_min = mesh.V.min(axis=0) - 0.5
    bounds_max = mesh.V.max(axis=0) + 0.5
    center = 0.5 * (bounds_min + bounds_max)
    clip_normal = geom.normals[0]
    return visualize_isosurface(
        q_vol,
        values,
        resolution=resolution,
        levels=levels,
        clip_origin=center,
        clip_normal=clip_normal,
        output_path=output_path,
    )


def generate_reg_mesh(f, n: int) -> MeshData:
    """Generate a regular N x N triangulated square mesh from f(u, v)."""

    if n < 1:
        raise ValueError("n must be >= 1.")

    verts = []
    index = {}
    for i in range(n + 1):
        for j in range(n + 1):
            u = i / n
            v = j / n
            index[(i, j)] = len(verts)
            verts.append(f(u, v))

    faces = []
    for i in range(n):
        for j in range(n):
            a = index[(i, j)]
            b = index[(i + 1, j)]
            c = index[(i, j + 1)]
            d = index[(i + 1, j + 1)]
            faces.append([a, b, c])
            faces.append([b, d, c])

    return MeshData(V=np.asarray(verts, dtype=float), F=np.asarray(faces, dtype=int))


def load_obj_mesh(path: str) -> MeshData:
    """Load a triangle mesh OBJ with libigl."""

    try:
        import igl
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("libigl is required to load OBJ meshes here.") from exc

    v, f = igl.read_triangle_mesh(path)
    return MeshData(V=np.asarray(v, dtype=float), F=np.asarray(f, dtype=int))


def sample_potential(
    mesh: MeshData,
    alpha: float = 0.1,
    p: float = 2.0,
    resolution: int = 60,
    xrange: Optional[Sequence[float]] = None,
    yrange: Optional[Sequence[float]] = None,
    zrange: Optional[Sequence[float]] = None,
    default_padding: float = 1.5,
    parallel=False
):
    """Sample potential on a regular grid and return samples plus clip params."""

    from geometry import build_connectivity, precompute_face_geometry

    connectivity = build_connectivity(mesh)
    geom = precompute_face_geometry(mesh)

    bounds_min = mesh.V.min(axis=0)
    bounds_max = mesh.V.max(axis=0)
    extent = np.maximum(bounds_max - bounds_min, 1e-6)
    pad = default_padding * extent
    bounds_min = bounds_min - pad
    bounds_max = bounds_max + pad

    if xrange is not None:
        bounds_min[0], bounds_max[0] = float(xrange[0]), float(xrange[1])
    if yrange is not None:
        bounds_min[1], bounds_max[1] = float(yrange[0]), float(yrange[1])
    if zrange is not None:
        bounds_min[2], bounds_max[2] = float(zrange[0]), float(zrange[1])

    q_vol = sample_volume_grid(bounds_min, bounds_max, resolution=resolution)
    comps = smoothed_offset_potential(q_vol, mesh, connectivity, geom, alpha=alpha, p=p,parallel=parallel)
    values = comps.face + comps.edge + comps.vertex
    center = 0.5 * (bounds_min + bounds_max)
    clip_normal = geom.normals[0]
    return q_vol, values, center, clip_normal


def visualize_square_function_potential(
    fx,
    fy,
    fz,
    alpha: float = 0.1,
    p: float = 2.0,
    n: int = 8,
    resolution: int = 60,
    xrange: Optional[Sequence[float]] = None,
    yrange: Optional[Sequence[float]] = None,
    zrange: Optional[Sequence[float]] = None,
    default_padding: float = 1.5,
    levels: Optional[Sequence[float]] = None,
    output_path: Optional[str] = None,
):
    """Generate mesh, sample potential, and visualize isosurfaces."""

    def f(u, v):
        return np.array([fx(u, v), fy(u, v), fz(u, v)], dtype=float)

    mesh = generate_reg_mesh(f, n)
    q_vol, values, center, clip_normal = sample_potential(
        mesh,
        alpha=alpha,
        p=p,
        resolution=resolution,
        xrange=xrange,
        yrange=yrange,
        zrange=zrange,
        default_padding=default_padding,
    )
    if levels is None:
        levels = [10.0, 100.0, 200.0, 500.0, 1000.0]
    return visualize_isosurface(
        q_vol,
        values,
        resolution=resolution,
        levels=levels,
        clip_origin=center,
        clip_normal=clip_normal,
        output_path=output_path,
    )

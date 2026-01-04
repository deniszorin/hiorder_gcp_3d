"""Visualization helpers for potentials."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence
import os

import numpy as np

from geometry import MeshData
from potential import smoothed_offset_potential


ArrayF = np.ndarray


@dataclass(frozen=True)
class ValidationScene:
    name: str
    mesh: MeshData
    one_sided: bool = False


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


def isolines_with_clip(
    mesh: MeshData,
    levels: Optional[Sequence[float]] = None,
    res: int = 100,
    one_sided: bool = False,
    include_faces: bool = True,
    include_edges: bool = True,
    include_vertices: bool = True,
    alpha: float = 0.1,
    p: float = 2.0,
    localized: bool = False,
    use_numba: bool = False,
    use_cpp: bool = False,
    use_accelerated: bool = False,
    use_simplified: bool = False,
    show_mesh: bool = False,
    use_widget: bool = True,
    clip_origin: Optional[ArrayF] = None,
    clip_normal: Optional[ArrayF] = None,
    output_path: Optional[str] = None,
):
    """Interactive isolines on a plane with sliders for normal and offset."""

    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - visualization dependency
        raise ImportError("pyvista is required for visualization.") from exc

    from geometry import precompute_mesh_geometry

    geom = precompute_mesh_geometry(mesh)

    bounds_min = mesh.V.min(axis=0) - 0.5
    bounds_max = mesh.V.max(axis=0) + 0.5
    extent = 0.5 * np.max(bounds_max - bounds_min)
    center = 0.5 * (bounds_min + bounds_max)

    if clip_origin is None:
        clip_origin = center
    clip_origin = np.asarray(clip_origin, dtype=float)

    if clip_normal is None:
        clip_normal = geom.normals[0]
    clip_normal = np.asarray(clip_normal, dtype=float)
    norm = np.linalg.norm(clip_normal)
    if norm <= 1e-12:
        clip_normal = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        clip_normal = clip_normal / norm

    if levels is None:
        levels = [1000.0]
    levels = np.asarray(levels, dtype=float)

    faces = np.hstack(
        [np.full((mesh.faces.shape[0], 1), 3, dtype=np.int64), mesh.faces.astype(np.int64)]
    ).ravel()
    pv_mesh = pv.PolyData(mesh.V, faces)

    base_origin = clip_origin.copy()
    base_normal = clip_normal.copy()

    state = {
        "offset": 0.0,
        "angle_x": float(np.degrees(np.arccos(np.clip(base_normal[0], -1.0, 1.0)))),
        "angle_y": float(np.degrees(np.arccos(np.clip(base_normal[1], -1.0, 1.0)))),
    }

    plotter = pv.Plotter()
    if show_mesh:
        plotter.add_mesh(pv_mesh, color="white", opacity=0.2, show_edges=True)

    contour_actor = None
    plane_actor = None

    def _current_normal() -> np.ndarray:
        angle_x = np.deg2rad(state["angle_x"])
        angle_y = np.deg2rad(state["angle_y"])

        nx = np.cos(angle_x)
        ny = np.cos(angle_y)
        remaining = 1.0 - nx * nx - ny * ny
        if remaining <= 0.0:
            scale = 1.0 / max(np.linalg.norm([nx, ny]), 1e-12)
            nx *= scale
            ny *= scale
            nz = 0.0
        else:
            nz = np.sqrt(remaining)

        n_pos = np.array([nx, ny, nz], dtype=float)
        n_neg = np.array([nx, ny, -nz], dtype=float)
        if np.dot(n_pos, base_normal) >= np.dot(n_neg, base_normal):
            return n_pos
        return n_neg

    def _update_scene() -> None:
        nonlocal contour_actor, plane_actor
        normal = _current_normal()
        origin = base_origin + state["offset"] * normal

        q_plane = sample_plane_grid(origin, normal, extent=extent, resolution=res)
        if use_simplified:
            from simplified_potential_numba import (
                simplified_smoothed_offset_potential_numba,
                simplified_smoothed_offset_potential_accelerated,
            )

            if use_accelerated:
                values = simplified_smoothed_offset_potential_accelerated(
                    q_plane, mesh, geom,
                    alpha=alpha,
                    p=p,
                    localized=localized,
                    include_faces=include_faces,
                    include_edges=include_edges,
                    include_vertices=include_vertices,
                    one_sided=one_sided,
                )
            else:
                values = simplified_smoothed_offset_potential_numba(
                    q_plane, mesh, geom,
                    alpha=alpha,
                    p=p,
                    localized=localized,
                    include_faces=include_faces,
                    include_edges=include_edges,
                    include_vertices=include_vertices,
                    one_sided=one_sided,
                )
        else:
            values = smoothed_offset_potential(
                q_plane,
                mesh,
                geom,
                alpha=alpha,
                p=p,
                localized=localized,
                include_faces=include_faces,
                include_edges=include_edges,
                include_vertices=include_vertices,
                one_sided=one_sided,
                use_numba=use_numba,
                use_cpp=use_cpp,
                use_accelerated=use_accelerated,
            )

        plane_points = pv.PolyData(q_plane)
        plane_points.point_data["potential"] = values
        surface = plane_points.delaunay_2d()
        isolines = surface.contour(levels.tolist(), scalars="potential")

        if contour_actor is not None:
            plotter.remove_actor(contour_actor)
            contour_actor = None
        if isolines.n_points > 0:
            contour_actor = plotter.add_mesh(isolines, color="#e34a33", line_width=2)

        plane = pv.Plane(
            center=origin,
            direction=normal,
            i_size=2.0 * extent,
            j_size=2.0 * extent,
        )
        if plane_actor is not None:
            plotter.remove_actor(plane_actor)
        plane_actor = plotter.add_mesh(plane, color="#cccccc", opacity=0.15)

        plotter.render()

    _update_scene()

    if use_widget:
        def _update_offset(value: float) -> None:
            state["offset"] = value
            _update_scene()

        def _update_angle_x(value: float) -> None:
            state["angle_x"] = value
            _update_scene()

        def _update_angle_y(value: float) -> None:
            state["angle_y"] = value
            _update_scene()

        plotter.add_slider_widget(
            _update_offset,
            [-float(extent), float(extent)],
            value=float(state["offset"]),
            title="Offset Along Normal",
            pointa=(0.02, 0.1),
            pointb=(0.35, 0.1),
        )
        plotter.add_slider_widget(
            _update_angle_x,
            [0.0, 180.0],
            value=float(state["angle_x"]),
            title="Normal Angle X",
            pointa=(0.02, 0.18),
            pointb=(0.35, 0.18),
        )
        plotter.add_slider_widget(
            _update_angle_y,
            [0.0, 180.0],
            value=float(state["angle_y"]),
            title="Normal Angle Y",
            pointa=(0.02, 0.26),
            pointb=(0.35, 0.26),
        )

    if output_path is not None:
        html_path = output_path if output_path.endswith(".html") else f"{output_path}.html"
        try:
            plotter.export_html(html_path)
        except ImportError:
            base, _ = os.path.splitext(html_path)
            pv_mesh.save(f"{base}.vtp")
        plotter.close()
    else:
        plotter.show()

    return plotter


def visualize_pointed_vertices(
    mesh: MeshData,
    point_size: float = 12.0,
    sphere_radius: Optional[float] = None,
    show_mesh: bool = True,
):
    """Visualize pointed vertices on a mesh using PyVista."""

    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - visualization dependency
        raise ImportError("pyvista is required for visualization.") from exc

    from geometry import precompute_mesh_geometry
    geom = precompute_mesh_geometry(mesh)

    faces = np.hstack(
        [np.full((mesh.faces.shape[0], 1), 3, dtype=np.int64), mesh.faces.astype(np.int64)]
    ).ravel()
    pv_mesh = pv.PolyData(mesh.V, faces)

    pointed = geom.pointed_vertices
    points = mesh.V[pointed]

    plotter = pv.Plotter()
    if show_mesh:
        plotter.add_mesh(pv_mesh, color="white", opacity=0.9, show_edges=True)

    extent = mesh.V.max(axis=0) - mesh.V.min(axis=0)
    normal_length = 0.2 * float(np.max(extent))
    for face in mesh.faces:
        p0, p1, p2 = mesh.V[face[0]], mesh.V[face[1]], mesh.V[face[2]]
        n = np.cross(p1 - p0, p2 - p0)
        n_norm = np.linalg.norm(n)
        if n_norm <= 1e-12:
            continue
        n = n / n_norm
        center = (p0 + p1 + p2) / 3.0
        arrow = pv.Line(center, center + normal_length * n)
        plotter.add_mesh(arrow, color="#1f78b4", line_width=2)

    if points.size:
        cloud = pv.PolyData(points)
        if sphere_radius is not None:
            spheres = cloud.glyph(geom=pv.Sphere(radius=sphere_radius), scale=False)
            plotter.add_mesh(spheres, color="#e34a33")
        else:
            plotter.add_mesh(
                cloud,
                color="#e34a33",
                point_size=point_size,
                render_points_as_spheres=True,
            )

    plotter.show()
    return plotter


def isosurface_with_clip(
    mesh: MeshData,
    levels: Optional[Sequence[float]] = None,
    res: int = 100,
    one_sided: bool = False,
    include_faces: bool = True,
    include_edges: bool = True,
    include_vertices: bool = True,
    alpha: float = 0.1,
    p: float = 2.0,
    localized: bool = False,
    use_numba: bool = False,
    use_cpp: bool = False,
    use_accelerated: bool = False,
    use_simplified: bool = False,
    show_mesh: bool = False,
    use_widget: bool = True,
    clip_origin: Optional[ArrayF] = None,
    clip_normal: Optional[ArrayF] = None,
    output_path: Optional[str] = None,
):
    """Interactive PyVista isosurfaces with a clipping plane widget."""

    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - visualization dependency
        raise ImportError("pyvista is required for visualization.") from exc

    from geometry import precompute_mesh_geometry

    geom = precompute_mesh_geometry(mesh)

    bounds_min = mesh.V.min(axis=0) - 0.5
    bounds_max = mesh.V.max(axis=0) + 0.5

    xs = np.linspace(bounds_min[0], bounds_max[0], res)
    ys = np.linspace(bounds_min[1], bounds_max[1], res)
    zs = np.linspace(bounds_min[2], bounds_max[2], res)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

    if use_simplified:
        from simplified_potential_numba import (
            simplified_smoothed_offset_potential_numba,
            simplified_smoothed_offset_potential_accelerated,
        )

        if use_accelerated:
            values = simplified_smoothed_offset_potential_accelerated(
                points, mesh, geom,
                alpha=alpha,
                p=p,
                localized=localized,
                include_faces=include_faces,
                include_edges=include_edges,
                include_vertices=include_vertices,
                one_sided=one_sided,
            )
        else:
            values = simplified_smoothed_offset_potential_numba(
                points, mesh, geom,
                alpha=alpha,
                p=p,
                localized=localized,
                include_faces=include_faces,
                include_edges=include_edges,
                include_vertices=include_vertices,
                one_sided=one_sided,
            )
    else:
        values = smoothed_offset_potential(
            points,
            mesh,
            geom,
            alpha=alpha,
            p=p,
            localized=localized,
            include_faces=include_faces,
            include_edges=include_edges,
            include_vertices=include_vertices,
            one_sided=one_sided,
            use_numba=use_numba,
            use_cpp=use_cpp,
            use_accelerated=use_accelerated,
        )
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

    if not use_widget and clip_origin is not None and clip_normal is not None:
        iso = iso.clip(normal=clip_normal, origin=clip_origin)

    p = pv.Plotter(off_screen=output_path is not None)
    p.add_mesh(iso, scalars="log_potential", cmap="viridis", opacity=1.0, name="clip")
    if show_mesh:
        faces = np.hstack(
            [np.full((mesh.faces.shape[0], 1), 3, dtype=np.int64), mesh.faces.astype(np.int64)]
        ).ravel()
        tri = pv.PolyData(mesh.V, faces)
        p.add_mesh(tri, color="white", opacity=0.3, show_edges=True)

    if use_widget:
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

    if output_path is not None:
        if output_path.endswith(".html"):
            p.export_html(output_path)
        else:
            p.show(screenshot=output_path)
        p.close()
    return p


def _edge_dir_in_face(face: ArrayF, a: int, b: int) -> int:
    v0, v1, v2 = face.tolist()
    edges = [(v0, v1), (v1, v2), (v2, v0)]
    if (a, b) in edges:
        return 1
    if (b, a) in edges:
        return -1
    return 0


def _orient_faces_consistent(faces: ArrayF) -> ArrayF:
    faces = faces.copy()
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for f_idx, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edge_to_faces.setdefault(key, []).append(f_idx)

    if faces.shape[0] == 0:
        return faces

    visited = np.zeros(faces.shape[0], dtype=bool)
    stack = [0]
    visited[0] = True
    while stack:
        f_idx = stack.pop()
        a, b, c = faces[f_idx]
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            for nbr in edge_to_faces.get(key, []):
                if nbr == f_idx or visited[nbr]:
                    continue
                dir_curr = _edge_dir_in_face(faces[f_idx], u, v)
                dir_nbr = _edge_dir_in_face(faces[nbr], u, v)
                if dir_curr == dir_nbr:
                    f0, f1, f2 = faces[nbr]
                    faces[nbr] = [f0, f2, f1]
                visited[nbr] = True
                stack.append(nbr)
    return faces


def _orient_faces_outward_global(V: ArrayF, faces: ArrayF) -> ArrayF:
    if faces.shape[0] == 0:
        return faces
    center = V.mean(axis=0)
    signs = []
    for a, b, c in faces:
        p0, p1, p2 = V[a], V[b], V[c]
        n = np.cross(p1 - p0, p2 - p0)
        centroid = (p0 + p1 + p2) / 3.0
        signs.append(np.dot(n, centroid - center))
    if np.mean(signs) < 0.0:
        faces = faces.copy()
        faces[:, [1, 2]] = faces[:, [2, 1]]
    return faces


def _angle_from_lengths(b: float, c: float, a: float) -> float:
    denom = 2.0 * b * c
    if denom <= 1e-12:
        return 0.0
    cos_val = (b * b + c * c - a * a) / denom
    return float(np.arccos(np.clip(cos_val, -1.0, 1.0)))


def _vertex_angle_from_origin(p_prev: ArrayF, p: ArrayF, p_next: ArrayF) -> float:
    b = float(np.linalg.norm(p))
    c_prev = float(np.linalg.norm(p_prev - p))
    a_prev = float(np.linalg.norm(p_prev))
    angle_prev = _angle_from_lengths(b, c_prev, a_prev)

    c_next = float(np.linalg.norm(p_next - p))
    a_next = float(np.linalg.norm(p_next))
    angle_next = _angle_from_lengths(b, c_next, a_next)
    return angle_prev + angle_next


def _reflex_delta(target_angle: float, r_reflex: float, radius: float) -> float:
    lo, hi = 1e-3, np.pi - 1e-3

    def angle_for_delta(delta: float) -> float:
        p = np.array([r_reflex, 0.0])
        p_prev = np.array([radius * np.cos(-delta), radius * np.sin(-delta)])
        p_next = np.array([radius * np.cos(delta), radius * np.sin(delta)])
        return _vertex_angle_from_origin(p_prev, p, p_next)

    angle_lo = angle_for_delta(lo)
    angle_hi = angle_for_delta(hi)
    if not (min(angle_lo, angle_hi) <= target_angle <= max(angle_lo, angle_hi)):
        raise ValueError("Requested reflex angle is outside the search range.")

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        angle_mid = angle_for_delta(mid)
        if (angle_mid < target_angle) == (angle_lo < target_angle):
            lo = mid
            angle_lo = angle_mid
        else:
            hi = mid
            angle_hi = angle_mid
    return 0.5 * (lo + hi)


def _nonconvex_polygon(k: int, reflex_angle: float, radius: float) -> ArrayF:
    if k < 4:
        raise ValueError("Need k>=4 for a nonconvex polygon.")
    r_reflex = 0.1 * radius
    delta = _reflex_delta(reflex_angle, r_reflex, radius)
    remaining = 2.0 * np.pi - 2.0 * delta
    step = remaining / (k - 2)
    angles = [0.0]
    angles.extend(delta + i * step for i in range(k - 1))
    radii = np.full(k, radius)
    radii[0] = r_reflex
    xs = radii * np.cos(angles)
    ys = radii * np.sin(angles)
    zs = np.zeros_like(xs)
    points = np.stack([xs, ys, zs], axis=1)
    angle = _vertex_angle_from_origin(points[-1], points[0], points[1])
    if abs(angle - reflex_angle) > 1e-3:
        raise ValueError("Nonconvex angle does not match requested reflex angle.")
    return points


def build_validation_scene_specs(reverse_faces: bool = False) -> List[ValidationScene]:
    """Construct meshes for validation scenarios."""

    scenes: List[ValidationScene] = []

    V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]], dtype=int)
    faces = _orient_faces_consistent(faces)
    if reverse_faces:
        faces = faces[:, [0, 2, 1]]
    scenes.append(ValidationScene(name="triangle", mesh=MeshData(V=V, faces=faces)))

    V = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    faces = np.array([[0, 1, 2], [1, 0, 3]], dtype=int)
    faces = _orient_faces_consistent(faces)
    if reverse_faces:
        faces = faces[:, [0, 2, 1]]
    scenes.append(ValidationScene(name="two_faces", mesh=MeshData(V=V, faces=faces)))

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
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]], dtype=int)
    faces = _orient_faces_consistent(faces)
    if reverse_faces:
        faces = faces[:, [0, 2, 1]]
    scenes.append(ValidationScene(name="ring_up", mesh=MeshData(V=V, faces=faces)))

    ring = np.stack([np.cos(angles), np.sin(angles), -np.ones(4)], axis=1)
    V = np.vstack([center, ring])
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]], dtype=int)
    faces = _orient_faces_consistent(faces)
    if reverse_faces:
        faces = faces[:, [0, 2, 1]]
    scenes.append(ValidationScene(name="ring_down", mesh=MeshData(V=V, faces=faces)))

    V = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    faces = np.array(
        [
            [0, 1, 2],
            [1, 0, 3],
            [3, 0, 2],
            [2, 1, 3],
        ],
        dtype=int,
    )
    faces = _orient_faces_consistent(faces)
    if reverse_faces:
        faces = faces[:, [0, 2, 1]]
#    faces = _orient_faces_outward_global(V, faces)
    scenes.append(
        ValidationScene(name="tetrahedron", mesh=MeshData(V=V, faces=faces), one_sided=False)
    )

    corners = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ]
    )
    corners = corners + np.array([0.02, -0.015, 0.01])
    face_quads = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ]
    centers = []
    for quad in face_quads:
        centers.append(corners[quad].mean(axis=0))
    centers[1] = np.array([0.0, 0.0, 0.0])
    V = np.vstack([corners, np.array(centers)])
    faces = []
    for f_idx, quad in enumerate(face_quads):
        c_idx = 8 + f_idx
        v0, v1, v2, v3 = quad
        faces.extend(
            [
                [c_idx, v0, v1],
                [c_idx, v1, v2],
                [c_idx, v2, v3],
                [c_idx, v3, v0],
            ]
        )
    faces = np.array(faces, dtype=int)
    faces = _orient_faces_consistent(faces)
    if reverse_faces:
        faces = faces[:, [0, 2, 1]]
    scenes.append(
        ValidationScene(name="cube_face_centers", mesh=MeshData(V=V, faces=faces), one_sided=True)
    )

    base = _nonconvex_polygon(k=6, reflex_angle=5.0 * np.pi / 3.0, radius=1.0)
    apex_top = np.array([0.0, 0.0, 1.0])
    apex_bottom = np.array([0.0, 0.0, 0.7])
    V = np.vstack([apex_top, apex_bottom, base])
    faces = []
    base_offset = 2
    for i in range(base.shape[0]):
        j = (i + 1) % base.shape[0]
        faces.append([0, base_offset + i, base_offset + j])
        faces.append([1, base_offset + j, base_offset + i])
    faces = np.array(faces, dtype=int)
    faces = _orient_faces_consistent(faces)
    if reverse_faces:
        faces = faces[:, [0, 2, 1]]
#    faces = _orient_faces_outward_global(V, faces)
    scenes.append(
        ValidationScene(name="double_cone_nonconvex_out", mesh=MeshData(V=V, faces=faces), one_sided=True)
    )
    scenes.append(
        ValidationScene(name="double_cone_nonconvex_both", mesh=MeshData(V=V, faces=faces), one_sided=False)
    )

    return scenes


def build_validation_scenes(reverse_faces=False) -> List[MeshData]:
    """Construct meshes for validation scenarios."""

    return [scene.mesh for scene in build_validation_scene_specs(reverse_faces=reverse_faces)]


def run_validation_visualizations(output_dir: Optional[str] = None,reverse_faces=False) -> None:
    """Generate all described visualizations for validation."""

    from geometry import precompute_mesh_geometry

    scenes = build_validation_scene_specs(reverse_faces=reverse_faces)
    level_values = [10.0, 100.0, 200.0, 500.0, 1000.0]
    levels_2d = np.log10(level_values)

    for idx, scene in enumerate(scenes):
        mesh = scene.mesh
        geom = precompute_mesh_geometry(mesh)

        center = mesh.V.mean(axis=0)
        edge_dir = mesh.V[1] - mesh.V[0]
        edge_norm = np.linalg.norm(edge_dir)
        if edge_norm == 0.0:
            n = geom.normals[0]
        else:
            n = edge_dir / edge_norm
        q_plane = sample_plane_grid(center, n, extent=1.5, resolution=100)
        values = smoothed_offset_potential(
            q_plane, mesh, geom, one_sided=scene.one_sided
        )
        output_path = None
        if output_dir is not None:
            output_path = f"{output_dir}/scene_{idx}_{scene.name}_isolines.png"
        log_values = np.log10(np.maximum(values, 1e-12))
        visualize_isolines_on_plane(
            q_plane,
            log_values,
            resolution=100,
            levels=levels_2d,
            output_path=output_path,
        )

        bounds_min = mesh.V.min(axis=0) - 1.0
        bounds_max = mesh.V.max(axis=0) + 1.0
        q_vol = sample_volume_grid(bounds_min, bounds_max, resolution=100)
        output_path = None
        if output_dir is not None:
            output_path = f"{output_dir}/scene_{idx}_{scene.name}_isosurface.html"
        center = 0.5 * (bounds_min + bounds_max)
        clip_normal = geom.normals[0]
        isosurface_with_clip(
            mesh,
            levels=level_values,
            res=100,
            one_sided=scene.one_sided,
            include_faces=True,
            include_edges=True,
            include_vertices=True,
            show_mesh=False,
            use_widget=False,
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

    return MeshData(V=np.asarray(verts, dtype=float), faces=np.asarray(faces, dtype=int))

"""Extrema visualization helpers using PyVista."""
""" !!! Not properly tested not using predicates do not fully trust this!!!"""

from __future__ import annotations

import math
import os
from typing import Iterable

import numpy as np

from geometry import MeshData
from geometry_connectivity_numba import _build_numba_connectivity


ArrayF = np.ndarray
ArrayI = np.ndarray

REGION_FACE = 0
REGION_EDGE = 1
REGION_VERTEX = 2


def _closest_point_on_segment(
    q: ArrayF,
    p0: ArrayF,
    p1: ArrayF,
) -> tuple[ArrayF, bool, float]:
    d = p1 - p0
    denom = float(np.dot(d, d))
    t = float(np.dot(q - p0, d)) / denom
    if t <= 0.0:
        return p0.copy(), False, t
    if t >= 1.0:
        return p1.copy(), False, t
    return p0 + t * d, True, t


def _closest_point_on_triangle(
    q: ArrayF,
    p0: ArrayF,
    p1: ArrayF,
    p2: ArrayF,
) -> tuple[ArrayF, int]:
    ab = p1 - p0
    ac = p2 - p0
    ap = q - p0
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return p0.copy(), REGION_VERTEX

    bp = q - p1
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return p1.copy(), REGION_VERTEX

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        point = p0 + v * ab
        return point, REGION_EDGE

    cp = q - p2
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return p2.copy(), REGION_VERTEX

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        point = p0 + w * ac
        return point, REGION_EDGE

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        point = p1 + w * (p2 - p1)
        return point, REGION_EDGE

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    point = p0 + ab * v + ac * w
    return point, REGION_FACE


def _points_close(a: ArrayF, b: ArrayF, tol: float) -> bool:
    return float(np.dot(a - b, a - b)) <= tol * tol


def _iter_vertex_edges_ccw(conn, v_idx: int) -> ArrayI:
    start = int(conn.vertex_edge_offsets[v_idx])
    end = int(conn.vertex_edge_offsets[v_idx + 1])
    return conn.vertex_edge_indices[start:end]


def _count_true_runs(flags: Iterable[bool], is_boundary: bool) -> int:
    flags_list = list(flags)
    if not flags_list:
        return 0
    count = 0
    if is_boundary:
        for i, flag in enumerate(flags_list):
            if flag and (i == 0 or not flags_list[i - 1]):
                count += 1
    else:
        for i, flag in enumerate(flags_list):
            if flag and not flags_list[i - 1]:
                count += 1
    return count


def face_point_type(region: int) -> str:
    if region == REGION_FACE:
        return "min"
    return "regular"


def edge_point_type(face_matches: Iterable[bool]) -> str:
    matches = list(face_matches)
    if len(matches) == 2:
        if all(matches):
            return "min"
        if not any(matches):
            return "saddle"
        return "regular"
    if len(matches) == 1:
        return "min" if matches[0] else "regular"
    return "regular"


def vertex_point_type(edge_flags: Iterable[bool], is_boundary: bool) -> str:
    flags = list(edge_flags)
    if not flags:
        return "regular"
    if all(flags):
        return "min"
    if not any(flags):
        return "max"
    k_plus = _count_true_runs(flags, is_boundary)
    if k_plus == 1:
        return "regular"
    return "saddle"


def _mean_face_edge_length(V: ArrayF, face: ArrayI) -> float:
    p0, p1, p2 = V[face[0]], V[face[1]], V[face[2]]
    l0 = float(np.linalg.norm(p1 - p0))
    l1 = float(np.linalg.norm(p2 - p1))
    l2 = float(np.linalg.norm(p0 - p2))
    return (l0 + l1 + l2) / 3.0


def draw_extrema(
    vertices: ArrayF, faces: ArrayI, vertex_idx: int,
    sphere_radius: float | None = None,
    show_mesh: bool = True, mesh_opacity: float = 0.9, show_edges: bool = True,
    mesh_color: str = "white", edge_color: str = "black",
    show_lines: bool = True, line_color: str = "#666666", line_width: float = 1.0,
    output_path: str | None = None,
    show: bool = True,
):
    """Draw minima/maxima/saddles for closest points to a chosen vertex."""

    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - visualization dependency
        raise ImportError("pyvista is required for visualization.") from exc

    V = np.asarray(vertices, dtype=float)
    F = np.asarray(faces, dtype=np.int64)
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError("vertices must have shape (nv, 3).")
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError("faces must have shape (nf, 3).")
    if vertex_idx < 0 or vertex_idx >= V.shape[0]:
        raise IndexError("vertex_idx is out of range.")

    mesh = MeshData(V=V, faces=F)
    conn = _build_numba_connectivity(mesh)

    extent = V.max(axis=0) - V.min(axis=0)
    scale = float(np.linalg.norm(extent))
    if scale <= 0.0:
        scale = 1.0
    tol = 1e-8 * scale
    fallback_radius = 0.015 * scale

    q = V[vertex_idx]
    boundary_edge = (conn.edge_faces[:, 0] < 0) | (conn.edge_faces[:, 1] < 0)

    face_edge_mean = np.empty(conn.faces.shape[0], dtype=float)
    for fidx, face in enumerate(conn.faces):
        face_edge_mean[fidx] = _mean_face_edge_length(V, face)

    edge_lengths = np.empty(conn.edges.shape[0], dtype=float)
    for eidx, edge in enumerate(conn.edges):
        p0, p1 = V[edge[0]], V[edge[1]]
        edge_lengths[eidx] = float(np.linalg.norm(p1 - p0))

    def safe_radius(value: float) -> float:
        if not np.isfinite(value) or value <= 0.0:
            return fallback_radius
        return value

    min_points: list[tuple[ArrayF, float]] = []
    max_points: list[tuple[ArrayF, float]] = []
    saddle_points: list[tuple[ArrayF, float]] = []

    for fidx, face in enumerate(conn.faces):
        if np.any(face == vertex_idx):
            continue
        p0, p1, p2 = V[face[0]], V[face[1]], V[face[2]]
        closest, region = _closest_point_on_triangle(q, p0, p1, p2)
        point_type = face_point_type(region)
        if point_type == "min":
            radius = safe_radius(0.1 * face_edge_mean[fidx])
            min_points.append((closest, radius))

    for eidx, edge in enumerate(conn.edges):
        if edge[0] == vertex_idx or edge[1] == vertex_idx:
            continue
        p0, p1 = V[edge[0]], V[edge[1]]
        closest, inside, _t = _closest_point_on_segment(q, p0, p1)
        if not inside:
            continue

        face_matches = []
        for face_idx in conn.edge_faces[eidx]:
            if face_idx < 0:
                continue
            face = conn.faces[int(face_idx)]
            f0, f1, f2 = V[face[0]], V[face[1]], V[face[2]]
            face_closest, _region = _closest_point_on_triangle(q, f0, f1, f2)
            face_matches.append(_points_close(face_closest, closest, tol))

        face_indices = [int(face_idx) for face_idx in conn.edge_faces[eidx] if face_idx >= 0]
        if face_indices:
            edge_radius = safe_radius(0.2 * float(np.mean(face_edge_mean[face_indices])))
        else:
            edge_radius = fallback_radius
        point_type = edge_point_type(face_matches)
        if point_type == "min":
            min_points.append((closest, edge_radius))
        elif point_type == "saddle":
            saddle_points.append((closest, edge_radius))

    for v_idx in range(V.shape[0]):
        edges_ccw = _iter_vertex_edges_ccw(conn, v_idx)
        if edges_ccw.size == 0:
            continue
        vertex_point = V[v_idx]
        edge_flags = []
        for edge_idx in edges_ccw:
            edge = conn.edges[int(edge_idx)]
            p0, p1 = V[edge[0]], V[edge[1]]
            closest, inside, _t = _closest_point_on_segment(q, p0, p1)
            is_vertex_closest = (not inside) and _points_close(closest, vertex_point, tol)
            edge_flags.append(is_vertex_closest)

        vertex_radius = safe_radius(0.2 * float(np.mean(edge_lengths[edges_ccw])))
        is_boundary = bool(boundary_edge[int(edges_ccw[0])] or boundary_edge[int(edges_ccw[-1])])
        point_type = vertex_point_type(edge_flags, is_boundary)
        if point_type == "min":
            min_points.append((vertex_point, vertex_radius))
        elif point_type == "max":
            max_points.append((vertex_point, vertex_radius))
        elif point_type == "saddle":
            saddle_points.append((vertex_point, vertex_radius))

    faces_pv = np.hstack(
        [np.full((F.shape[0], 1), 3, dtype=np.int64), F.astype(np.int64)]
    ).ravel()
    pv_mesh = pv.PolyData(V, faces_pv)

    plotter = pv.Plotter()
    if show_mesh:
        plotter.add_mesh(
            pv_mesh,
            color=mesh_color,
            opacity=mesh_opacity,
            show_edges=show_edges,
            edge_color=edge_color,
        )

    if show_lines:
        line_targets = [
            point for point, _radius in min_points + max_points + saddle_points
        ]
        for point in line_targets:
            if _points_close(point, q, tol):
                continue
            plotter.add_mesh(
                pv.Line(q, point),
                color=line_color,
                line_width=line_width,
            )

    for points, color in (
        (min_points, "#3182bd"),
        (max_points, "#de2d26"),
        (saddle_points, "#31a354"),
    ):
        if not points:
            continue
        coords = np.asarray([point for point, _radius in points], dtype=float)
        if sphere_radius is None:
            radii = np.asarray([radius for _point, radius in points], dtype=float)
        else:
            radii = np.full(len(points), sphere_radius, dtype=float)
        cloud = pv.PolyData(coords)
        cloud["scale"] = radii
        spheres = cloud.glyph(geom=pv.Sphere(radius=1.0), scale="scale", factor=1.0)
        plotter.add_mesh(spheres, color=color)

    if output_path is not None:
        html_path = output_path if output_path.endswith(".html") else f"{output_path}.html"
        try:
            plotter.export_html(html_path)
        except ImportError:
            base, _ = os.path.splitext(html_path)
            pv_mesh.save(f"{base}.vtp")

    if show:
        plotter.show()
    return plotter


def generate_sphere_mesh(
    n_longitude: int = 32,
    n_latitude: int = 16,
    radius: float = 1.0,
) -> tuple[ArrayF, ArrayI]:
    """Generate a latitude-longitude sphere mesh (triangle fans at caps)."""

    if n_longitude < 3:
        raise ValueError("n_longitude must be at least 3.")
    if n_latitude < 2:
        raise ValueError("n_latitude must be at least 2.")

    vertices: list[list[float]] = []
    vertices.append([0.0, 0.0, radius])

    for j in range(1, n_latitude):
        theta = math.pi * j / n_latitude
        ring_r = radius * math.sin(theta)
        z = radius * math.cos(theta)
        for i in range(n_longitude):
            phi = 2.0 * math.pi * i / n_longitude
            x = ring_r * math.cos(phi)
            y = ring_r * math.sin(phi)
            vertices.append([x, y, z])

    vertices.append([0.0, 0.0, -radius])

    faces: list[list[int]] = []
    north_idx = 0
    south_idx = len(vertices) - 1

    if n_latitude > 1:
        first_ring = 1
        for i in range(n_longitude):
            next_i = (i + 1) % n_longitude
            faces.append([north_idx, first_ring + i, first_ring + next_i])

    ring_count = n_latitude - 1
    for j in range(ring_count - 1):
        ring_a = 1 + j * n_longitude
        ring_b = ring_a + n_longitude
        for i in range(n_longitude):
            next_i = (i + 1) % n_longitude
            v00 = ring_a + i
            v01 = ring_a + next_i
            v10 = ring_b + i
            v11 = ring_b + next_i
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])

    if n_latitude > 1:
        last_ring = 1 + (ring_count - 1) * n_longitude
        for i in range(n_longitude):
            next_i = (i + 1) % n_longitude
            faces.append([south_idx, last_ring + next_i, last_ring + i])

    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64)

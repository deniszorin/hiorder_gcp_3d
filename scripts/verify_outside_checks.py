"""Generate visualization checks for outside_* functions."""

from __future__ import annotations

import math
from pathlib import Path
import os
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

MPL_DIR = ROOT_DIR / ".mplconfig"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT_DIR / ".cache"))
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from geometry import MeshData, precompute_mesh_geometry
from potential import precompute_potential_terms, outside_edge, outside_face, outside_vertex


ArrayF = np.ndarray


def _mesh_to_pv(mesh: MeshData):
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - visualization dependency
        raise ImportError("pyvista is required for visualization.") from exc

    pv.OFF_SCREEN = True

    faces = np.hstack(
        [np.full((mesh.faces.shape[0], 1), 3, dtype=np.int64), mesh.faces.astype(np.int64)]
    ).ravel()
    return pv.PolyData(mesh.V, faces)


def _sample_cube_points(mesh: MeshData, resolution: int = 4, padding: float = 0.1) -> ArrayF:
    vmin = mesh.V.min(axis=0)
    vmax = mesh.V.max(axis=0)
    center = 0.5 * (vmin + vmax)
    side = np.max(vmax - vmin)
    side *= 1.0 + padding
    bounds_min = center - 0.5 * side
    bounds_max = center + 0.5 * side

    xs = np.linspace(bounds_min[0], bounds_max[0], resolution)
    ys = np.linspace(bounds_min[1], bounds_max[1], resolution)
    zs = np.linspace(bounds_min[2], bounds_max[2], resolution)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)


def _cube_frame(mesh: MeshData, padding: float = 0.1) -> tuple[ArrayF, float]:
    vmin = mesh.V.min(axis=0)
    vmax = mesh.V.max(axis=0)
    center = 0.5 * (vmin + vmax)
    side = np.max(vmax - vmin)
    return center, side * (1.0 + padding)


def _plot_outside_result(
    mesh: MeshData,
    q: ArrayF,
    outside: ArrayF,
    closest: ArrayF,
    output_path: Path,
    title: str,
    interactive: bool,
    overlays: list | None = None,
) -> None:
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - visualization dependency
        raise ImportError("pyvista is required for visualization.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotter = pv.Plotter(off_screen=True, window_size=(1400, 900))
    plotter.add_text(title, font_size=12)
    plotter.add_mesh(_mesh_to_pv(mesh), color="white", opacity=1.0, show_edges=True)
    if overlays:
        for overlay in overlays:
            plotter.add_mesh(overlay, color="#cccccc", opacity=0.5)
    # Draw face normals in blue, thicker than the closest-point segments.
    for f in mesh.faces:
        p0, p1, p2 = mesh.V[f[0]], mesh.V[f[1]], mesh.V[f[2]]
        n = np.cross(p1 - p0, p2 - p0)
        n_norm = np.linalg.norm(n)
        if n_norm <= 1e-12:
            continue
        n = n / n_norm
        center = (p0 + p1 + p2) / 3.0
        length = 0.2 * np.max(mesh.V.max(axis=0) - mesh.V.min(axis=0))
        arrow = pv.Line(center, center + length * n)
        plotter.add_mesh(arrow, color="#1f78b4", line_width=3)

    cloud = pv.PolyData(q)
    cloud["outside"] = outside.astype(np.int32)
    plotter.add_mesh(
        cloud,
        scalars="outside",
        cmap=["#2b8cbe", "#e34a33"],
        point_size=12,
        render_points_as_spheres=True,
    )

    npts = q.shape[0]
    points = np.vstack([q, closest])
    lines = np.column_stack(
        [np.full(npts, 2, dtype=np.int64), np.arange(npts), np.arange(npts) + npts]
    ).ravel()
    segments = pv.PolyData(points, lines=lines)
    plotter.add_mesh(segments, color="#555555", opacity=1.0, line_width=1)

    if interactive:
        html_path = output_path.with_suffix(".html")
        plotter.export_html(str(html_path))
    else:
        plotter.show(screenshot=str(output_path))
    plotter.close()


def _edge_index(mesh: MeshData, a: int, b: int) -> int:
    for eidx, (v0, v1) in enumerate(mesh.edges):
        if (v0 == a and v1 == b) or (v0 == b and v1 == a):
            return eidx
    raise ValueError(f"Edge ({a}, {b}) not found.")


def _regular_polygon(k: int, radius: float = 1.0, z: float = 0.0) -> ArrayF:
    angles = np.linspace(0.0, 2.0 * math.pi, k, endpoint=False)
    xs = radius * np.cos(angles)
    ys = radius * np.sin(angles)
    zs = np.full_like(xs, z)
    return np.stack([xs, ys, zs], axis=1)


def _angle_from_lengths(b: float, c: float, a: float) -> float:
    denom = 2.0 * b * c
    if denom <= 1e-12:
        return 0.0
    cos_val = (b * b + c * c - a * a) / denom
    return math.acos(max(-1.0, min(1.0, cos_val)))


def _vertex_angle_from_origin(p_prev: ArrayF, p: ArrayF, p_next: ArrayF) -> float:
    b = float(np.linalg.norm(p))
    c_prev = float(np.linalg.norm(p_prev - p))
    a_prev = float(np.linalg.norm(p_prev))
    angle_prev = _angle_from_lengths(b, c_prev, a_prev)

    c_next = float(np.linalg.norm(p_next - p))
    a_next = float(np.linalg.norm(p_next))
    angle_next = _angle_from_lengths(b, c_next, a_next)

    return angle_prev + angle_next


def _verify_reflex_angle(points: ArrayF, reflex_angle: float, label: str) -> None:
    angle = _vertex_angle_from_origin(points[-1], points[0], points[1])
    if abs(angle - reflex_angle) > 1e-3:
        raise ValueError(
            f"{label} reflex angle mismatch: {angle} vs {reflex_angle}."
        )


def _reflex_delta(target_angle: float, r_reflex: float, radius: float) -> float:
    lo, hi = 1e-3, math.pi - 1e-3

    def angle_for_delta(delta: float) -> float:
        p = np.array([r_reflex, 0.0])
        p_prev = np.array([radius * math.cos(-delta), radius * math.sin(-delta)])
        p_next = np.array([radius * math.cos(delta), radius * math.sin(delta)])
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


def _nonconvex_polygon(k: int, reflex_angle: float, radius: float = 1.0) -> ArrayF:
    if k < 4:
        raise ValueError("Need k>=4 for a nonconvex polygon.")
    r_reflex = 0.1 * radius
    delta = _reflex_delta(reflex_angle, r_reflex, radius)
    remaining = 2.0 * math.pi - 2.0 * delta
    step = remaining / (k - 2)
    angles = [0.0]
    angles.extend(delta + i * step for i in range(k - 1))
    radii = np.full(k, radius)
    radii[0] = r_reflex
    xs = radii * np.cos(angles)
    ys = radii * np.sin(angles)
    zs = np.zeros_like(xs)
    points = np.stack([xs, ys, zs], axis=1)
    _verify_reflex_angle(points, reflex_angle, "Nonconvex polygon")
    return points


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


def _orient_faces_outward_global(V: ArrayF, faces: ArrayF, center: ArrayF) -> ArrayF:
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


def _cone_mesh(vertices: ArrayF, apex: ArrayF) -> MeshData:
    V = np.vstack([apex, vertices])
    k = vertices.shape[0]
    faces = np.array([[0, 1 + i, 1 + (i + 1) % k] for i in range(k)], dtype=int)
    faces = _orient_faces_consistent(faces)
  #  faces = _orient_faces_outward_global(V, faces, V.mean(axis=0))
    return MeshData(V=V, faces=faces)


def _unit_basis_from_normal(n: ArrayF) -> tuple[ArrayF, ArrayF]:
    n = np.asarray(n, dtype=float)
    if abs(n[2]) < 0.9:
        ref = np.array([0.0, 0.0, 1.0])
    else:
        ref = np.array([1.0, 0.0, 0.0])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def _quad_from_axes(center: ArrayF, u: ArrayF, v: ArrayF, size: float):
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - visualization dependency
        raise ImportError("pyvista is required for visualization.") from exc

    half = 0.5 * size
    pts = np.array(
        [
            center - half * u - half * v,
            center + half * u - half * v,
            center + half * u + half * v,
            center - half * u + half * v,
        ]
    )
    faces = np.array([4, 0, 1, 2, 3], dtype=np.int64)
    return pv.PolyData(pts, faces)


def _triangle_from_edges(p0: ArrayF, d0: ArrayF, d1: ArrayF, size: float):
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - visualization dependency
        raise ImportError("pyvista is required for visualization.") from exc

    pts = np.array([p0, p0 + size * d0, p0 + size * d1])
    faces = np.array([3, 0, 1, 2], dtype=np.int64)
    return pv.PolyData(pts, faces)


def build_face_mesh() -> MeshData:
    V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]], dtype=int)
    return MeshData(V=V, faces=faces)


def build_edge_mesh(outward: bool = True) -> MeshData:
    V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    if outward:
        faces = np.array([[0, 2, 1], [0, 1, 3]], dtype=int)
    else:
        faces = np.array([[0, 1, 2], [0, 3, 1]], dtype=int)
    faces = _orient_faces_consistent(faces)
 #   faces = _orient_faces_outward_global(V, faces, V.mean(axis=0))
    return MeshData(V=V, faces=faces)


def build_vertex_cone(k: int = 5, height: float = 1.0, radius: float = 1.0) -> MeshData:
    vertices = _regular_polygon(k, radius=radius, z=0.0)
    apex = np.array([0.0, 0.0, height])
    return _cone_mesh(vertices, apex)


def build_vertex_cone_nonconvex(
    k: int = 5, height: float = 1.0, radius: float = 1.0, reflex_angle: float = 5.0 * math.pi / 3.0
) -> MeshData:
    vertices = _nonconvex_polygon(k, reflex_angle=reflex_angle, radius=radius)
    _verify_reflex_angle(vertices, reflex_angle, "Nonconvex cone")
    apex = np.array([0.0, 0.0, height])
    return _cone_mesh(vertices, apex)


def build_vertex_saddle(
    k: int = 6, amplitude: float = 0.3, radius: float = 1.0
) -> MeshData:
    vertices = _regular_polygon(k, radius=radius, z=0.0)
    phi = np.arctan2(vertices[:, 1], vertices[:, 0])
    vertices[:, 2] = amplitude * np.sin(2.0 * phi)
    apex = np.array([0.0, 0.0, 0.0])
    return _cone_mesh(vertices, apex)


def build_vertex_saddle_nonconvex(
    k: int = 6,
    amplitude: float = 0.3,
    radius: float = 1.0,
    reflex_angle: float = 5.0 * math.pi / 3.0,
) -> MeshData:
    vertices = _nonconvex_polygon(k, reflex_angle=reflex_angle, radius=radius)
    _verify_reflex_angle(vertices, reflex_angle, "Nonconvex saddle")
    phi = np.arctan2(vertices[:, 1], vertices[:, 0])
    vertices[:, 2] = amplitude * np.sin(2.0 * phi)
    apex = np.array([0.0, 0.0, 0.0])
    return _cone_mesh(vertices, apex)

def run_outside_face_test(
    mesh: MeshData,
    face_idx: int,
    output_path: Path,
    resolution: int = 4,
    density_scale: float = 1.0,
    dry_run: bool = False,
    interactive: bool = False,
) -> None:
    geom = precompute_mesh_geometry(mesh)
    res = max(2, int(round(resolution * density_scale)))
    q = _sample_cube_points(mesh, resolution=res)
    _, terms = precompute_potential_terms(q, mesh, geom)
    outside, closest = outside_face(face_idx, terms)
    if dry_run:
        print(f"outside_face: {outside.shape} closest {closest.shape}")
        return
    center, side = _cube_frame(mesh)
    n = geom.normals[face_idx]
    p0 = mesh.V[mesh.faces[face_idx, 0]]
    center_proj = center - np.dot(center - p0, n) * n
    u, v = _unit_basis_from_normal(n)
    square = _quad_from_axes(center_proj, u, v, side)
    _plot_outside_result(
        mesh,
        q,
        outside,
        closest,
        output_path,
        "outside_face",
        interactive,
        overlays=[square],
    )


def run_outside_edge_test(
    mesh: MeshData,
    edge_idx: int,
    output_path: Path,
    resolution: int = 4,
    density_scale: float = 1.0,
    dry_run: bool = False,
    interactive: bool = False,
) -> None:
    geom = precompute_mesh_geometry(mesh)
    res = max(2, int(round(resolution * density_scale)))
    q = _sample_cube_points(mesh, resolution=res)
    _, terms = precompute_potential_terms(q, mesh, geom)
    outside, closest = outside_edge(edge_idx, q, mesh, geom, terms)
    if dry_run:
        print(f"outside_edge: {outside.shape} closest {closest.shape}")
        return
    center, side = _cube_frame(mesh)
    a, b = mesh.edges[edge_idx]
    edge_mid = 0.5 * (mesh.V[a] + mesh.V[b])
    edge_dir = mesh.V[b] - mesh.V[a]
    edge_dir /= np.linalg.norm(edge_dir)
    overlays = []
    for f in mesh.edges_to_faces[edge_idx]:
        n = geom.normals[f]
        axis_v = np.cross(n, edge_dir)
        axis_v /= np.linalg.norm(axis_v)
        overlays.append(_quad_from_axes(edge_mid, edge_dir, axis_v, side))
    _plot_outside_result(
        mesh,
        q,
        outside,
        closest,
        output_path,
        "outside_edge",
        interactive,
        overlays=overlays,
    )


def run_outside_vertex_test(
    mesh: MeshData,
    v_idx: int,
    output_path: Path,
    resolution: int = 4,
    density_scale: float = 1.0,
    dry_run: bool = False,
    interactive: bool = False,
) -> None:
    geom = precompute_mesh_geometry(mesh)
    res = max(2, int(round(resolution * density_scale)))
    q = _sample_cube_points(mesh, resolution=res)
    _, terms = precompute_potential_terms(q, mesh, geom)
    outside, closest = outside_vertex(v_idx, q, mesh, geom, terms)
    if dry_run:
        print(f"outside_vertex: {outside.shape} closest {closest.shape}")
        return
    center, side = _cube_frame(mesh)
    overlays = []
    vpos = mesh.V[v_idx]
    scale = 2.0 * side
    for f in mesh.vertices_to_faces[v_idx]:
        v0, v1, v2 = mesh.faces[f]
        if v_idx == v0:
            a, b = v1, v2
        elif v_idx == v1:
            a, b = v0, v2
        else:
            a, b = v0, v1
        d0 = mesh.V[a] - vpos
        d1 = mesh.V[b] - vpos
        d0 /= np.linalg.norm(d0)
        d1 /= np.linalg.norm(d1)
        overlays.append(_triangle_from_edges(vpos, d0, d1, scale))
    _plot_outside_result(
        mesh,
        q,
        outside,
        closest,
        output_path,
        "outside_vertex",
        interactive,
        overlays=overlays,
    )


def main(
    output_dir: Path | None = None,
    dry_run: bool = False,
    interactive: bool = False,
    density_scale: float = 1.0,
) -> None:
    base_dir = output_dir or Path("visualizations") / "outside_checks"

    face_mesh = build_face_mesh()
    run_outside_face_test(
        face_mesh,
        0,
        base_dir / "face_triangle.png",
        dry_run=dry_run,
        interactive=interactive,
        density_scale=density_scale,
    )

    edge_mesh = build_edge_mesh(outward=True)
    edge_idx = _edge_index(edge_mesh, 0, 1)
    run_outside_edge_test(
        edge_mesh,
        edge_idx,
        base_dir / "edge_tetra_outward.png",
        dry_run=dry_run,
        interactive=interactive,
        density_scale=density_scale,
    )

    edge_mesh_flip = build_edge_mesh(outward=False)
    edge_idx_flip = _edge_index(edge_mesh_flip, 0, 1)
    run_outside_edge_test(
        edge_mesh_flip,
        edge_idx_flip,
        base_dir / "edge_tetra_inward.png",
        dry_run=dry_run,
        interactive=interactive,
        density_scale=density_scale,
    )

    cone = build_vertex_cone()
    run_outside_vertex_test(
        cone,
        0,
        base_dir / "vertex_cone_convex.png",
        dry_run=dry_run,
        interactive=interactive,
        density_scale=density_scale,
    )

    cone_nc = build_vertex_cone_nonconvex()
    run_outside_vertex_test(
        cone_nc,
        0,
        base_dir / "vertex_cone_nonconvex.png",
        dry_run=dry_run,
        interactive=interactive,
        density_scale=density_scale,
    )

    saddle = build_vertex_saddle()
    run_outside_vertex_test(
        saddle,
        0,
        base_dir / "vertex_saddle_convex.png",
        dry_run=dry_run,
        interactive=interactive,
        density_scale=density_scale,
    )

    saddle_nc = build_vertex_saddle_nonconvex()
    run_outside_vertex_test(
        saddle_nc,
        0,
        base_dir / "vertex_saddle_nonconvex.png",
        dry_run=dry_run,
        interactive=interactive,
        density_scale=density_scale,
    )

    try:
        from viz import build_validation_scene_specs
    except ImportError:
        return
    for scene in build_validation_scene_specs():
        if scene.name == "tetrahedron":
            for v_idx in (1, 2, 3):
                run_outside_vertex_test(
                    scene.mesh,
                    v_idx,
                    base_dir / f"vertex_tetrahedron_v{v_idx}.png",
                    dry_run=dry_run,
                    interactive=interactive,
                    density_scale=density_scale,
                )
            break


if __name__ == "__main__":
    density_scale = 1.0
    if "--density" in sys.argv:
        idx = sys.argv.index("--density")
        if idx + 1 < len(sys.argv):
            density_scale = float(sys.argv[idx + 1])
    for arg in sys.argv:
        if arg.startswith("--density="):
            density_scale = float(arg.split("=", 1)[1])

    main(
        dry_run="--dry-run" in sys.argv,
        interactive="--interactive" in sys.argv,
        density_scale=density_scale,
    )

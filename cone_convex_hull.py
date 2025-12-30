"""Convex hull for polygonal cones."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import predicates


ArrayF = np.ndarray

_ORIGIN = np.zeros(3, dtype=float)


def _orient_3d(a: ArrayF, b: ArrayF, c: ArrayF) -> float:
    """Return oriented volume using exact orient3d with origin as reference."""

    return float(predicates.orient3d(_ORIGIN, a, b, c))


def _find_start_index(e: ArrayF, eps: float) -> int | None:
    n = e.shape[0]
    for i in range(n - 2):
        if _orient_3d(e[i], e[i + 1], e[i + 2]) != 0.0:
            return i
    return None


def _are_opposite_collinear(a: ArrayF, b: ArrayF) -> bool:
    cross = np.cross(a, b)
    if np.all(cross == 0.0) and np.dot(a, b) < 0.0:
        return True
    norm_prod = np.linalg.norm(a) * np.linalg.norm(b)
    return np.dot(a, b) == -norm_prod


def _would_create_opposite_pair_tail(
    e_rot: ArrayF, deque_indices: deque[int], new_idx: int
) -> bool:
    if len(deque_indices) < 2:
        return False
    prev_idx = deque_indices[-2]
    return _are_opposite_collinear(e_rot[prev_idx], e_rot[new_idx])


def _would_create_opposite_pair_head(
    e_rot: ArrayF, deque_indices: deque[int], new_idx: int
) -> bool:
    if len(deque_indices) < 2:
        return False
    next_idx = deque_indices[1]
    return _are_opposite_collinear(e_rot[new_idx], e_rot[next_idx])



def cone_convex_hull(
    e: ArrayF,
    eps: float = 1e-12,
    debug: bool = False,
    debug_label: str | None = None,
) -> tuple[np.ndarray, bool, bool]:
    """Return indices of convex hull of cone edge directions in CCW order.

    Returns (indices, coplanar, fullspace). coplanar is True when no non-coplanar
    triple is found. fullspace is True when fewer than 3 edges remain in the
    output. If debug is True, prints raw deque indices and the final index
    sequence. debug_label prefixes those lines when provided.
    """

    e = np.asarray(e, dtype=float)
    if e.ndim != 2 or e.shape[1] != 3:
        raise ValueError("e must be (n, 3).")
    n = e.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 edge directions.")

    start = _find_start_index(e, eps)
    if start is None:
        indices = np.arange(n, dtype=int)
        fullspace = indices.size < 3
        if debug:
            prefix = f"{debug_label}: " if debug_label else ""
            print(f"{prefix}coplanar input, returning all indices")
            print(f"{prefix}raw D (pre-transform, rotated order): {indices.tolist()}")
            print(f"{prefix}final indices: {indices.tolist()}")
        return indices, True, fullspace
    order = np.concatenate([np.arange(start, n), np.arange(0, start)])
    e_rot = e[order]

    D: deque[int] = deque()
    o = _orient_3d(e_rot[0], e_rot[1], e_rot[2])
    if o > 0.0:
        D.appendleft(0)
        D.appendleft(2)
        D.append(1)
        D.append(2)
    else:
        D.appendleft(1)
        D.appendleft(2)
        D.append(0)
        D.append(2)

    for i in range(3, n):
        if (
            _orient_3d(e_rot[D[-2]], e_rot[D[-1]], e_rot[i]) > 0.0
            and _orient_3d(e_rot[i], e_rot[D[0]], e_rot[D[1]]) > 0.0
        ):
            continue

        while len(D) > 1 and _orient_3d(e_rot[D[-2]], e_rot[D[-1]], e_rot[i]) < 0.0:
            if _would_create_opposite_pair_tail(e_rot, D, i):
                break
            D.pop()
        D.append(i)

        while len(D) > 1 and _orient_3d(e_rot[i], e_rot[D[0]], e_rot[D[1]]) < 0.0:
            if _would_create_opposite_pair_head(e_rot, D, i):
                break
            D.popleft()
        D.appendleft(i)

    if len(D) > 1 and D[-1] == D[0]:
        D.pop()
    raw_indices = np.array(list(D), dtype=int)
    indices = order[raw_indices]
    fullspace = indices.size < 3
    if debug:
        prefix = f"{debug_label}: " if debug_label else ""
        print(f"{prefix}raw D (pre-transform, rotated order): {raw_indices.tolist()}")
        print(f"{prefix}final indices: {indices.tolist()}")
    return indices, False, fullspace


def validate_cone_convex_hull(e: ArrayF, D: Iterable[int], eps: float = 1e-12) -> bool:
    """Validate convex hull faces via n[j] dot e[i] <= eps."""

    e = np.asarray(e, dtype=float)
    indices = list(D)
    if not indices:
        return False
    m = len(indices)
    for j in range(m):
        a = e[indices[j]]
        b = e[indices[(j + 1) % m]]
        n = np.cross(a, b)
        dots = e @ n
        if np.any(dots > eps):
            return False
    return True


@dataclass(frozen=True)
class ConeHullResult:
    indices: np.ndarray
    valid: bool
    coplanar: bool
    fullspace: bool


def compute_cone_convex_hull(e: ArrayF, eps: float = 1e-12) -> ConeHullResult:
    """Compute convex hull and validation flag."""

    indices, coplanar, fullspace = cone_convex_hull(e, eps=eps)
    valid = validate_cone_convex_hull(e, indices, eps=eps)
    return ConeHullResult(
        indices=indices, valid=valid, coplanar=coplanar, fullspace=fullspace
    )


def visualize_cone_hull(
    e: ArrayF,
    D: Iterable[int] | None = None,
    output_path: str = "cone_convex_hull.html",
) -> None:
    """Render the cone and its convex hull as HTML using PyVista."""

    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - visualization dependency
        raise ImportError("pyvista is required for visualization.") from exc

    e = np.asarray(e, dtype=float)
    if e.ndim != 2 or e.shape[1] != 3:
        raise ValueError("e must be (n, 3).")
    n = e.shape[0]
    apex = np.zeros(3, dtype=float)
    verts = np.vstack([apex, e])

    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([0, 1 + i, 1 + j])
    faces = np.asarray(faces, dtype=np.int64)
    faces_flat = np.hstack(
        [np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]
    ).ravel()
    cone_mesh = pv.PolyData(verts, faces_flat)

    if D is None:
        D, _, _ = cone_convex_hull(e)
    D = list(D)
    hull_faces = []
    m = len(D)
    for j in range(m):
        a = D[j]
        b = D[(j + 1) % m]
        hull_faces.append([0, 1 + a, 1 + b])
    hull_faces = np.asarray(hull_faces, dtype=np.int64)
    hull_flat = np.hstack(
        [np.full((hull_faces.shape[0], 1), 3, dtype=np.int64), hull_faces]
    ).ravel()
    hull_mesh = pv.PolyData(verts, hull_flat)

    def _face_centers_and_normals(edges: np.ndarray, order: list[int]) -> tuple[np.ndarray, np.ndarray]:
        centers = []
        normals = []
        if len(order) == 0:
            return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float)
        for idx in range(len(order)):
            a = order[idx]
            b = order[(idx + 1) % len(order)]
            va = edges[a]
            vb = edges[b]
            n = np.cross(va, vb)
            norm = np.linalg.norm(n)
            if norm == 0.0:
                continue
            centers.append((va + vb) / 3.0)
            normals.append(n / norm)
        if not centers:
            return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float)
        return np.array(centers, dtype=float), np.array(normals, dtype=float)

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(cone_mesh, color="white", opacity=1.0, show_edges=True)
    plotter.add_mesh(hull_mesh, color="#e34a33", opacity=0.5)

    cone_centers, cone_normals = _face_centers_and_normals(e, list(range(n)))
    if cone_centers.size:
        plotter.add_arrows(cone_centers, cone_normals, mag=0.3, color="#1f78b4")
    hull_centers, hull_normals = _face_centers_and_normals(e, D)
    if hull_centers.size:
        plotter.add_arrows(hull_centers, hull_normals, mag=0.3, color="#33a02c")

    plotter.export_html(output_path)
    plotter.close()

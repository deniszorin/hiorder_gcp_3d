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
    eps: float = 1e-12
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
    if indices.size:
        hull_set = set(indices.tolist())
        indices = np.array([i for i in range(n) if i in hull_set], dtype=int)
    fullspace = indices.size < 3
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


def pointed_vertex(e: ArrayF, eps: float = 1e-12) -> bool:
    """Return True when the convex cone is pointed (polar cone outside)."""

    e = np.asarray(e, dtype=float)
    if e.ndim != 2 or e.shape[1] != 3:
        raise ValueError("e must be (n, 3).")
    if e.shape[0] < 3:
        return False

    result = compute_cone_convex_hull(e, eps=eps)
    if result.coplanar or result.fullspace:
        return False

    indices = result.indices
    m = indices.size
    if m < 3:
        return False

    for i in range(m):
        a = e[indices[i]]
        b = e[indices[(i + 1) % m]]
        n = np.cross(a, b)
        if np.linalg.norm(n) <= eps:
            continue
        for j in range(m):
            if j == i or j == (i + 1) % m:
                continue
            c = e[indices[j]]
            dot = np.dot(n, c)
            if abs(dot) > eps:
                return dot < 0.0
    return False

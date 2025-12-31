import math

import numpy as np
import pytest
from scipy.optimize import nnls

from cone_convex_hull import cone_convex_hull, validate_cone_convex_hull


def _normalize_rows(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1)
    return v / norms[:, None]


def _cone_edges(vertices: np.ndarray, apex: np.ndarray) -> np.ndarray:
    return _normalize_rows(vertices - apex[None, :])


def _ensure_ccw_xy(vertices: np.ndarray) -> np.ndarray:
    xs = vertices[:, 0]
    ys = vertices[:, 1]
    area = np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))
    if area < 0.0:
        return vertices[::-1].copy()
    return vertices


def _regular_polygon(k: int, radius: float = 1.0, z: float = 0.0) -> np.ndarray:
    if k < 6:
        raise ValueError("k must be at least 6.")
    angles = np.linspace(0.0, 2.0 * math.pi, k, endpoint=False)
    xs = radius * np.cos(angles)
    ys = radius * np.sin(angles)
    zs = np.full_like(xs, z)
    vertices = np.stack([xs, ys, zs], axis=1)
    return _ensure_ccw_xy(vertices)


def _star_polygon(k: int, r_outer: float = 1.0, r_inner: float = 0.5) -> np.ndarray:
    if k < 6:
        raise ValueError("k must be at least 6.")
    angles = np.linspace(0.0, 2.0 * math.pi, k, endpoint=False)
    radii = np.where(np.arange(k) % 2 == 0, r_outer, r_inner)
    xs = radii * np.cos(angles)
    ys = radii * np.sin(angles)
    zs = np.zeros_like(xs)
    vertices = np.stack([xs, ys, zs], axis=1)
    return _ensure_ccw_xy(vertices)


def _is_cyclic_equal(a: np.ndarray, b: np.ndarray) -> bool:
    if len(a) != len(b):
        return False
    if len(a) == 0:
        return True
    a_list = list(a)
    b_list = list(b)
    for shift in range(len(b_list)):
        if a_list == b_list[shift:] + b_list[:shift]:
            return True
    return False


def _is_cyclic_equal_or_reverse(a: np.ndarray, b: np.ndarray) -> bool:
    if _is_cyclic_equal(a, b):
        return True
    return _is_cyclic_equal(a, b[::-1])


def _assert_hull_valid(e: np.ndarray, D: np.ndarray) -> None:
    assert validate_cone_convex_hull(e, D, eps=1e-12)


def _perturb_vertices(vertices: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    offsets = np.zeros_like(vertices)
    for i in range(vertices.shape[0]):
        offsets[i] = np.array(
            [(i + 1) * eps, -(i + 2) * eps, (i + 3) * eps], dtype=float
        )
    return vertices + offsets


def _perturb_edges(e: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    offsets = np.zeros_like(e)
    for i in range(e.shape[0]):
        offsets[i] = np.array(
            [(i + 1) * eps, (i + 2) * eps, -(i + 3) * eps], dtype=float
        )
    e = e + offsets
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def _corner_line_points(t_values: np.ndarray, y: float, z: float) -> np.ndarray:
    return np.stack(
        [t_values, np.full_like(t_values, y), np.full_like(t_values, z)], axis=1
    )


_CORNER_T_LINE1 = np.array([-1.8, -0.6, 0.2, 1.1, 1.7], dtype=float)
_CORNER_T_LINE2 = np.array([-1.7, -0.5, 0.3, 1.0, 1.6], dtype=float)


def _build_corner_case(m: int, n: int) -> np.ndarray:
    if not (2 <= m <= 5 and 2 <= n <= 5):
        raise ValueError("m and n must be in 2..5.")
    points = [np.array([-1.0, 0.0, 0.0])]
    if m > 1:
        points.extend(_corner_line_points(_CORNER_T_LINE1[: m - 1], -1.0, -1.0))
    points.append(np.array([1.0, 0.0, 0.0]))
    points.extend(_corner_line_points(_CORNER_T_LINE2[::-1][:n], 1.0, -1.0))
    e = np.array(points, dtype=float)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def _reflex_delta(target_angle: float, r_reflex: float, radius: float) -> float:
    def angle_for_delta(delta: float) -> float:
        p0 = np.array([r_reflex, 0.0])
        p1 = np.array([radius * math.cos(delta), radius * math.sin(delta)])
        p2 = np.array([radius * math.cos(-delta), radius * math.sin(-delta)])
        v1 = p1 - p0
        v2 = p2 - p0
        dot = np.dot(v1, v2)
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        return math.acos(max(-1.0, min(1.0, dot / denom)))

    lo, hi = 0.0, math.pi / 2.0
    angle_lo = angle_for_delta(lo)
    angle_hi = angle_for_delta(hi)
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


def _nonconvex_polygon(
    k: int, reflex_angle: float, radius: float = 1.0
) -> np.ndarray:
    if k < 6:
        raise ValueError("k must be at least 6.")
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
    return _ensure_ccw_xy(points)


def _prune_expected(e: np.ndarray, indices: np.ndarray, eps: float) -> np.ndarray:
    indices = indices.astype(int).tolist()
    changed = True
    while changed and len(indices) > 2:
        changed = False
        for idx in list(indices):
            others = [j for j in indices if j != idx]
            if len(others) < 2:
                continue
            A = e[others].T
            b = e[idx]
            _, rnorm = nnls(A, b)
            if rnorm <= eps:
                indices.remove(idx)
                changed = True
                break
    return np.array(indices, dtype=int)


def test_convex_polygon_inside_apex():
    vertices = _regular_polygon(6)
    apex = np.array([0.0, 0.0, 1.0])
    e = _cone_edges(vertices, apex)
    D, coplanar, fullspace = cone_convex_hull(e)
    assert not coplanar
    assert not fullspace
    assert len(D) == 6
    assert _is_cyclic_equal_or_reverse(D, np.arange(6))
    _assert_hull_valid(e, D)


def test_convex_polygon_outside_apex():
    vertices = _regular_polygon(6)
    apex = np.array([2.0, 0.0, 1.0])
    e = _cone_edges(vertices, apex)
    D, coplanar, fullspace = cone_convex_hull(e)
    assert not coplanar
    assert not fullspace
    assert len(D) == 6
    assert set(D.tolist()) == set(range(6))
    _assert_hull_valid(e, D)


def test_convex_polygon_with_midpoints():
    base = _regular_polygon(6)
    verts = []
    for i in range(base.shape[0]):
        j = (i + 1) % base.shape[0]
        verts.append(base[i])
        verts.append(0.5 * (base[i] + base[j]))
    vertices = _ensure_ccw_xy(np.array(verts))
    apex = np.array([0.0, 0.0, 1.0])
    e = _cone_edges(vertices, apex)
    D, coplanar, fullspace = cone_convex_hull(e)
    assert not coplanar
    assert not fullspace
    expected = np.arange(0, vertices.shape[0], 2)
    assert set(expected.tolist()).issubset(set(D.tolist()))
    _assert_hull_valid(e, D)


def test_star_polygon_alternating():
    vertices = _star_polygon(8)
    apex = np.array([0.0, 0.0, 1.0])
    e = _cone_edges(vertices, apex)
    D, coplanar, fullspace = cone_convex_hull(e)
    assert not coplanar
    assert not fullspace
    expected = np.arange(0, 8, 2)
    assert len(D) == len(expected)
    assert _is_cyclic_equal_or_reverse(D, expected)
    _assert_hull_valid(e, D)


def test_all_coplanar_returns_input():
    angles = np.linspace(0.0, 2.0 * math.pi, 6, endpoint=False)
    e = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1)
    D, coplanar, fullspace = cone_convex_hull(e)
    assert coplanar
    assert not fullspace
    assert np.array_equal(D, np.arange(e.shape[0]))


def test_spiral_polygon():
    N = 3
    d = 1.0
    points = [np.array([0.0, 0.0])]
    dirs = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    for s in range(1, 4 * N):
        length = math.ceil(s / 2) * d
        direction = dirs[(s - 1) % 4]
        points.append(points[-1] + length * direction)
    p = np.array(points)
    q = []
    base = np.array([1.0 / 3.0, 1.0 / 3.0])
    for i in range(4 * N):
        angle = i * math.pi / 2.0
        R = np.array(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        )
        q.append(R @ base)
    q = np.array(q)
    polygon = np.vstack([p, q[::-1]])
    vertices = np.column_stack([polygon, np.zeros(polygon.shape[0])])
    vertices = _ensure_ccw_xy(vertices)
    apex = np.array([0.0, 0.0, 1.0])
    e = _cone_edges(vertices, apex)
    D, coplanar, fullspace = cone_convex_hull(e)
    assert not coplanar
    assert not fullspace
    expected = np.arange(4 * N - 5, 4 * N)
    expected = _prune_expected(e, expected, eps=1e-12)
    assert len(D) == len(expected)
    assert set(D.tolist()) == set(expected.tolist())
    _assert_hull_valid(e, D)


def test_convex_lift_two_vertices():
    vertices = _regular_polygon(6)
    apex = np.array([0.0, 0.0, 1.0])
    vertices[0, 2] = apex[2]
    vertices[3, 2] = apex[2]
    e = _cone_edges(vertices, apex)
    D, coplanar, fullspace = cone_convex_hull(e)
    assert not coplanar
    assert not fullspace
    assert len(D) == 4
    assert {0, 3}.issubset(set(D.tolist()))
    _assert_hull_valid(e, D)


def test_convex_lift_two_vertices_perturbed():
    vertices = _regular_polygon(6)
    apex = np.array([0.0, 0.0, 1.0])
    vertices[0, 2] = apex[2]
    vertices[3, 2] = apex[2]
    vertices = _perturb_vertices(vertices)
    e = _cone_edges(vertices, apex)
    D, coplanar, fullspace = cone_convex_hull(e)
    assert not coplanar
    assert not fullspace
    assert D.tolist() == [0, 2, 3, 4, 5]
    assert not validate_cone_convex_hull(e, D, eps=1e-12)


def test_nonconvex_lift_two_vertices():
    vertices = _nonconvex_polygon(6, reflex_angle=5.0 * math.pi / 3.0)
    apex = np.array([0.0, 0.0, 1.0])
    vertices[0, 2] = apex[2]
    vertices[3, 2] = apex[2]
    e = _cone_edges(vertices, apex)
    D, coplanar, fullspace = cone_convex_hull(e)
    assert not coplanar
    assert not fullspace
    assert len(D) == 4
    assert {0, 3}.issubset(set(D.tolist()))
    _assert_hull_valid(e, D)


def test_nonconvex_lift_two_vertices_perturbed():
    vertices = _nonconvex_polygon(6, reflex_angle=5.0 * math.pi / 3.0)
    apex = np.array([0.0, 0.0, 1.0])
    vertices[0, 2] = apex[2]
    vertices[3, 2] = apex[2]
    vertices = _perturb_vertices(vertices)
    e = _cone_edges(vertices, apex)
    D, coplanar, fullspace = cone_convex_hull(e)
    assert not coplanar
    assert not fullspace
    assert D.tolist() == [0, 1, 3, 5]
    assert not validate_cone_convex_hull(e, D, eps=1e-12)


def test_convex_lift_three_vertices():
    vertices = _regular_polygon(6)
    apex = np.array([0.0, 0.0, 1.0])
    for idx in (0, 2, 4):
        vertices[idx, 2] = apex[2]
    e = _cone_edges(vertices, apex)
    D, coplanar, fullspace = cone_convex_hull(e)
    assert not coplanar
    assert not fullspace
    assert len(D) == 3
    assert set(D.tolist()) == {0, 2, 4}
    _assert_hull_valid(e, D)


def test_nonconvex_lift_three_vertices():
    vertices = _nonconvex_polygon(6, reflex_angle=5.0 * math.pi / 3.0)
    apex = np.array([0.0, 0.0, 1.0])
    for idx in (0, 2, 4):
        vertices[idx, 2] = apex[2]
    e = _cone_edges(vertices, apex)
    D, coplanar, fullspace = cone_convex_hull(e)
    assert not coplanar
    assert not fullspace
    assert len(D) == 3
    assert set(D.tolist()) == {0, 2, 4}
    _assert_hull_valid(e, D)


_CORNER_CASES = [
    ("case_1", 2, 2, [0, 1, 2, 3, 4], True, False, [0, 1, 2, 3, 4], True),
    ("case_2", 2, 3, [0, 1, 2, 3, 4, 5], True, False, [0, 1, 2, 3, 4, 5], True),
    ("case_3", 2, 4, [0, 1, 2, 3, 4, 5, 6], True, False, [0, 1, 2, 3, 4, 5, 6], True),
    ("case_4", 2, 5, [0, 1, 2, 3, 4, 5, 6, 7], True, False, [0, 1, 2, 3, 4, 5, 6, 7], True),
    ("case_5", 3, 2, [0, 1, 2, 3, 4, 5], True, False, [0, 1, 2, 3, 4, 5], True),
    ("case_6", 3, 3, [0, 1, 2, 3, 4, 5, 6], True, False, [0, 1, 2, 3, 4, 5, 6], True),
    ("case_7", 3, 4, [0, 1, 2, 3, 4, 5, 6, 7], True, False, [0, 1, 2, 3, 4, 5, 6, 7], True),
    ("case_8", 4, 2, [0, 1, 2, 3, 4, 5, 6], True, False, [0, 1, 2, 3, 4, 5, 6], True),
    ("case_9", 4, 3, [0, 1, 2, 3, 4, 5, 6, 7], True, False, [0, 1, 2, 3, 4, 5, 6, 7], True),
    ("case_10", 5, 2, [0, 1, 2, 3, 4, 5, 6, 7], True, False, [0, 1, 2, 3, 4, 5, 6, 7], True),
]


@pytest.mark.parametrize(
    "name,m,n,expected,expected_valid,expected_coplanar,expected_perturbed,expected_perturbed_valid",
    _CORNER_CASES,
)
def test_corner_case_exact(
    name: str,
    m: int,
    n: int,
    expected: list[int],
    expected_valid: bool,
    expected_coplanar: bool,
    expected_perturbed: list[int],
    expected_perturbed_valid: bool,
) -> None:
    e = _build_corner_case(m, n)
    D, coplanar, fullspace = cone_convex_hull(e)
    assert coplanar is expected_coplanar
    assert not fullspace
    assert np.array_equal(D, np.array(expected, dtype=int))
    assert validate_cone_convex_hull(e, D, eps=1e-12) is expected_valid


@pytest.mark.parametrize(
    "name,m,n,expected,expected_valid,expected_coplanar,expected_perturbed,expected_perturbed_valid",
    _CORNER_CASES,
)
def test_corner_case_perturbed(
    name: str,
    m: int,
    n: int,
    expected: list[int],
    expected_valid: bool,
    expected_coplanar: bool,
    expected_perturbed: list[int],
    expected_perturbed_valid: bool,
) -> None:
    e = _build_corner_case(m, n)
    e = _perturb_edges(e)
    D, coplanar, fullspace = cone_convex_hull(e)
    assert not coplanar
    assert not fullspace
    assert np.array_equal(D, np.array(expected_perturbed, dtype=int))
    assert validate_cone_convex_hull(e, D, eps=1e-12) is expected_perturbed_valid

from __future__ import annotations

import argparse
import math
from pathlib import Path

import os
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, os.fspath(PROJECT_ROOT))

from cone_convex_hull import cone_convex_hull, visualize_cone_hull


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


def _spiral_polygon(N: int, d: float = 1.0) -> np.ndarray:
    points = [np.array([0.0, 0.0])]
    dirs = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    for s in range(1, 4 * N):
        length = math.ceil(s / 2) * d
        direction = dirs[(s - 1) % 4]
        points.append(points[-1] + length * direction)
    p = np.array(points)

    q = []
    base = np.array([1.0 / 10.0, 1.0 / 10.0])
    for i in range(4 * N):
        angle = i * math.pi / 2.0
        R = np.array(
            [[math.cos(angle), -math.sin(angle)], [math.cos(angle), math.sin(angle)]]
        )
        q.append(p[i] + R @ base)
    q = np.array(q)

    polygon = np.vstack([p, q[::-1]])
    vertices = np.column_stack([polygon, np.zeros(polygon.shape[0])])
    return _ensure_ccw_xy(vertices)


def _write_case(
    output_dir: Path,
    name: str,
    vertices: np.ndarray,
    apex: np.ndarray,
) -> None:
    e = _cone_edges(vertices, apex)
    _write_edge_case(output_dir, name, e)
    _write_edge_case(output_dir, f"{name}_reversed", e[::-1].copy())


def _write_edge_case(output_dir: Path, name: str, e: np.ndarray) -> None:
    e = _normalize_rows(e)
    D, _, _ = cone_convex_hull(e)
    output_path = output_dir / f"{name}.html"
    visualize_cone_hull(e, D=D, output_path=str(output_path))
    print(f"{name}: wrote {output_path}")


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
    return _normalize_rows(e + offsets)


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
    return np.array(points, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="visualizations/cone_convex_hull",
        help="Directory to write HTML outputs.",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    apex = np.array([0.0, 0.0, 1.0])
    convex = _regular_polygon(6)
    _write_case(output_dir, "convex_inside", convex, apex)

    apex_out = np.array([2.0, 0.0, 1.0])
    _write_case(output_dir, "convex_outside", convex, apex_out)

    midpoints = []
    for i in range(convex.shape[0]):
        j = (i + 1) % convex.shape[0]
        midpoints.append(convex[i])
        midpoints.append(0.5 * (convex[i] + convex[j]))
    _write_case(
        output_dir, "convex_midpoints", _ensure_ccw_xy(np.array(midpoints)), apex
    )

    star = _star_polygon(8)
    _write_case(output_dir, "star_polygon", star, apex)

    spiral = _spiral_polygon(3)
    _write_case(output_dir, "spiral_polygon", spiral, apex)

    convex_two = convex.copy()
    convex_two[0, 2] = apex[2]
    convex_two[3, 2] = apex[2]
    _write_case(output_dir, "convex_two_lifted", convex_two, apex)

    star_two = star.copy()
    star_two[0, 2] = apex[2]
    star_two[4, 2] = apex[2]
    _write_case(output_dir, "star_two_lifted", star_two, apex)

    convex_three = convex.copy()
    for idx in (0, 2, 4):
        convex_three[idx, 2] = apex[2]
    _write_case(output_dir, "convex_three_lifted", convex_three, apex)

    star_three = star.copy()
    for idx in (0, 2, 4):
        star_three[idx, 2] = apex[2]
    _write_case(output_dir, "star_three_lifted", star_three, apex)

    convex_two_pert = _perturb_vertices(convex_two)
    _write_case(output_dir, "convex_two_lifted_perturbed", convex_two_pert, apex)

    star_two_pert = _perturb_vertices(star_two)
    _write_case(output_dir, "star_two_lifted_perturbed", star_two_pert, apex)

    corner_cases = [
        ("case_1", 2, 2),
        ("case_2", 2, 3),
        ("case_3", 2, 4),
        ("case_4", 2, 5),
        ("case_5", 3, 2),
        ("case_6", 3, 3),
        ("case_7", 3, 4),
        ("case_8", 4, 2),
        ("case_9", 4, 3),
        ("case_10", 5, 2),
    ]
    for name, m, n in corner_cases:
        e = _build_corner_case(m, n)
        _write_edge_case(output_dir, f"corner_{name}_exact", e)
        _write_edge_case(output_dir, f"corner_{name}_exact_reversed", e[::-1].copy())
        _write_edge_case(
            output_dir, f"corner_{name}_perturbed", _perturb_edges(e)
        )
        _write_edge_case(
            output_dir,
            f"corner_{name}_perturbed_reversed",
            _perturb_edges(e)[::-1].copy(),
        )


if __name__ == "__main__":
    main()

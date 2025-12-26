from pathlib import Path

import numpy as np

from geometry import MeshData, precompute_face_geometry
from potential import smoothed_offset_potential
from viz import sample_volume_grid


def _load_regression():
    data_path = Path(__file__).resolve().parent / "data" / "potential_regression.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing regression data: {data_path}")
    return np.load(data_path)


def _sample_triangle():
    V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]], dtype=int)
    mesh = MeshData(V=V, faces=faces)
    geom = precompute_face_geometry(mesh)
    q = sample_volume_grid(V.min(axis=0) - 0.5, V.max(axis=0) + 0.5, resolution=50)
    return smoothed_offset_potential(q, mesh, geom)


def _sample_tetrahedron_missing_face():
    V = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1]], dtype=int)
    mesh = MeshData(V=V, faces=faces)
    geom = precompute_face_geometry(mesh)
    q = sample_volume_grid(V.min(axis=0) - 0.5, V.max(axis=0) + 0.5, resolution=50)
    return smoothed_offset_potential(q, mesh, geom)


def test_potential_regression_triangle():
    data = _load_regression()
    current = _sample_triangle()
    np.testing.assert_allclose(current, data["tri"], rtol=1e-7, atol=1e-7)


def test_potential_regression_tetrahedron_missing_face():
    data = _load_regression()
    current = _sample_tetrahedron_missing_face()
    np.testing.assert_allclose(current, data["tet_missing_face"], rtol=1e-7, atol=1e-7)

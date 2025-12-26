import numpy as np

import geometry
import potential


def _mesh_single_triangle():
    V = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2]], dtype=int)
    return geometry.MeshData(V=V, faces=faces)


def test_H_piecewise():
    z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    out = potential.H(z)
    assert out[0] == 0.0
    assert out[-1] == 1.0
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)


def test_smoothed_offset_potential_runs():
    mesh = _mesh_single_triangle()
    geom = geometry.precompute_face_geometry(mesh)

    q = np.array([[0.2, 0.2, 0.5], [0.3, 0.1, -0.2]])
    values = potential.smoothed_offset_potential(q, mesh, geom)

    assert values.shape == (2,)
    assert np.all(np.isfinite(values))


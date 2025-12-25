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
    comps = potential.smoothed_offset_potential(q, mesh, geom)

    assert comps.face.shape == (2,)
    assert comps.edge.shape == (2,)
    assert comps.vertex.shape == (2,)
    assert np.all(np.isfinite(comps.face))
    assert np.all(np.isfinite(comps.edge))
    assert np.all(np.isfinite(comps.vertex))


def test_smoothed_offset_potential_flags():
    mesh = _mesh_single_triangle()
    geom = geometry.precompute_face_geometry(mesh)

    q = np.array([[0.2, 0.2, 0.5]])
    comps = potential.smoothed_offset_potential(
        q,
        mesh,
        geom,
        include_faces=False,
        include_edges=False,
        include_vertices=True,
    )

    assert comps.face.shape == (1,)
    assert comps.edge.shape == (1,)
    assert comps.vertex.shape == (1,)
    assert np.all(comps.face == 0.0)
    assert np.all(comps.edge == 0.0)

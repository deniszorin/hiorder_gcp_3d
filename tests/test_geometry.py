import math
import tempfile
from pathlib import Path

import numpy as np
import textwrap

import geometry
from geometry import load_obj_mesh


def _mesh_single_triangle():
    V = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    F = np.array([[0, 1, 2]], dtype=int)
    return geometry.MeshData(V=V, F=F)


def _mesh_tetrahedron():
    V = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    F = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 1],
            [3, 2, 1],
        ],
        dtype=int,
    )
    return geometry.MeshData(V=V, F=F)


def _mesh_tetrahedron_missing_face():
    mesh = _mesh_tetrahedron()
    F = mesh.F[:-1]
    return geometry.MeshData(V=mesh.V, F=F)


def test_build_connectivity_single_triangle():
    mesh = _mesh_single_triangle()
    conn = geometry.build_connectivity(mesh)
    assert len(conn.faces_per_vertex) == 3
    assert all(len(faces) == 1 for faces in conn.faces_per_vertex)
    assert len(conn.faces_per_edge) == 3
    assert all(len(faces) == 1 for faces in conn.faces_per_edge)
    assert len(conn.edges_per_vertex) == 3
    assert all(len(edges) == 2 for edges in conn.edges_per_vertex)


def test_build_connectivity_tetrahedron():
    mesh = _mesh_tetrahedron()
    conn = geometry.build_connectivity(mesh)
    assert all(len(faces) == 3 for faces in conn.faces_per_vertex)
    assert all(len(faces) == 2 for faces in conn.faces_per_edge)
    assert all(len(edges) == 3 for edges in conn.edges_per_vertex)


def test_build_connectivity_tetrahedron_missing_face():
    mesh = _mesh_tetrahedron_missing_face()
    conn = geometry.build_connectivity(mesh)
    face_counts = [len(faces) for faces in conn.faces_per_vertex]
    assert face_counts.count(3) == 1
    assert face_counts.count(2) == 3
    edge_counts = [len(faces) for faces in conn.faces_per_edge]
    assert edge_counts.count(1) == 3
    assert edge_counts.count(2) == 3


def test_precompute_face_geometry_single_triangle():
    mesh = _mesh_single_triangle()
    geom = geometry.precompute_face_geometry(mesh)
    assert geom.normals.shape == (1, 3)
    assert geom.edge_dirs.shape == (1, 3, 3)
    assert geom.edge_inward.shape == (1, 3, 3)

    n = geom.normals[0]
    assert math.isclose(np.linalg.norm(n), 1.0, rel_tol=1e-7)
    # Expected normal is +Z for CCW triangle in XY plane.
    assert n[2] > 0.0

    for i in range(3):
        d = geom.edge_dirs[0, i]
        assert math.isclose(np.linalg.norm(d), 1.0, rel_tol=1e-7)
        inward = geom.edge_inward[0, i]
        # Inward edge normal should be perpendicular to normal.
        assert math.isclose(float(np.dot(n, inward)), 0.0, abs_tol=1e-7)


def test_load_obj_mesh_libigl_triangle(tmp_path: Path):
    pytest = __import__("pytest")
    pytest.importorskip("igl")
    obj_text = textwrap.dedent(
        """
        v 0 0 0
        v 1 0 0
        v 0 1 0
        f 1 2 3
        """
    ).strip()
    obj_path = tmp_path / "tri.obj"
    obj_path.write_text(obj_text, encoding="utf-8")

    mesh = load_obj_mesh(str(obj_path))
    assert mesh.V.shape == (3, 3)
    assert mesh.F.shape == (1, 3)
    assert np.allclose(mesh.V[1], [1.0, 0.0, 0.0])
    assert np.all(mesh.F[0] == np.array([0, 1, 2]))


def test_load_obj_mesh_libigl():
    pytest = __import__("pytest")
    pytest.importorskip("igl")
    obj_path = Path(__file__).resolve().parent / "Bunny-LowPoly.obj"
    mesh = load_obj_mesh(str(obj_path))
    assert mesh.V.shape[0] > 0
    assert mesh.F.shape[0] > 0
    assert mesh.F.shape[1] == 3

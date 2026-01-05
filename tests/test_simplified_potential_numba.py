import numpy as np

import geometry
import potential
import potential_cpp
import simplified_potential_numba
import viz


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


def _mesh_tetrahedron():
    V = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    faces = np.array(
        [
            [0, 1, 2],
            [1, 0, 3],
            [3, 0, 2],
            [2, 1, 3],
        ],
        dtype=int,
    )
    return geometry.MeshData(V=V, faces=faces)


def _mesh_two_faces():
    V = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    faces = np.array([[0, 1, 2], [1, 0, 3]], dtype=int)
    return geometry.MeshData(V=V, faces=faces)


def _sample_mesh_grid(mesh, resolution):
    bounds_min = mesh.V.min(axis=0) - 0.5
    bounds_max = mesh.V.max(axis=0) + 0.5
    q = viz.sample_volume_grid(bounds_min, bounds_max, resolution=resolution)
    return q + np.array([1e-6, 2e-6, 3e-6])


def _assert_match(ref, simplified, rtol=1e-6, atol=1e-6):
    nan_ref = np.isnan(ref)
    nan_simplified = np.isnan(simplified)
    assert np.array_equal(nan_ref, nan_simplified)
    mask = ~(nan_ref | nan_simplified)
    np.testing.assert_allclose(simplified[mask], ref[mask], rtol=rtol, atol=atol)


def _project_point_to_segment(q, p0, p1):
    d = p1 - p0
    denom = float(np.dot(d, d))
    if denom <= 1e-12:
        proj = p0
        return proj, float(np.linalg.norm(q - proj))
    t = float(np.dot(q - p0, d)) / denom
    t = float(np.clip(t, 0.0, 1.0))
    proj = p0 + t * d
    return proj, float(np.linalg.norm(q - proj))


def _triangle_distance(q, p0, p1, p2):
    ab = p1 - p0
    ac = p2 - p0
    ap = q - p0
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return float(np.linalg.norm(ap))

    bp = q - p1
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return float(np.linalg.norm(bp))

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        proj = p0 + v * ab
        return float(np.linalg.norm(q - proj))

    cp = q - p2
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return float(np.linalg.norm(cp))

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        proj = p0 + w * ac
        return float(np.linalg.norm(q - proj))

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        proj = p1 + w * (p2 - p1)
        return float(np.linalg.norm(q - proj))

    n = np.cross(ab, ac)
    n_norm = float(np.linalg.norm(n))
    if n_norm <= 1e-12:
        return float(np.linalg.norm(ap))
    return abs(float(np.dot(n, ap))) / n_norm


def _expected_simplified_potential(q_points, mesh, p=2.0):
    edge_valence = np.array(
        [len(face_list) for face_list in mesh.edges_to_faces],
        dtype=int,
    )
    vertex_internal = np.ones(mesh.V.shape[0], dtype=bool)
    for edge_idx, (v0, v1) in enumerate(mesh.edges):
        if edge_valence[edge_idx] < 2:
            vertex_internal[v0] = False
            vertex_internal[v1] = False

    faces = mesh.faces
    edges = mesh.edges
    V = mesh.V

    out = np.zeros(q_points.shape[0], dtype=float)
    for i, q in enumerate(q_points):
        face_sum = 0.0
        for f in faces:
            p0, p1, p2 = V[f[0]], V[f[1]], V[f[2]]
            r_f = _triangle_distance(q, p0, p1, p2)
            face_sum += 1.0 / (r_f ** p)

        edge_sum = 0.0
        for edge_idx, (v0, v1) in enumerate(edges):
            weight = edge_valence[edge_idx] - 1
            if weight == 0:
                continue
            _, r_e = _project_point_to_segment(q, V[v0], V[v1])
            edge_sum += weight * (1.0 / (r_e ** p))

        vertex_sum = 0.0
        for v_idx, p_v in enumerate(V):
            if vertex_internal[v_idx]:
                r_v = float(np.linalg.norm(q - p_v))
                vertex_sum += 1.0 / (r_v ** p)

        out[i] = face_sum - edge_sum + vertex_sum
    return out


def _reference_smoothed_alpha_zero(q, mesh, geom):
    with np.errstate(divide="ignore", invalid="ignore"):
        return potential.smoothed_offset_potential(
            q, mesh, geom,
            alpha=0.0,
            localized=False,
            one_sided=False,
        )


def test_simplified_potential_numba_triangle_matches_formula():
    mesh = _mesh_single_triangle()
    geom = geometry.precompute_mesh_geometry(mesh)
    q = _sample_mesh_grid(mesh, resolution=6)

    ref = _expected_simplified_potential(q, mesh)
    simplified = simplified_potential_numba.simplified_smoothed_offset_potential_numba(
        q, mesh, geom,
        alpha=0.0,
        localized=False,
        one_sided=False,
    )

    _assert_match(ref, simplified)


def test_simplified_potential_numba_triangle_matches_smoothed_alpha_zero():
    mesh = _mesh_single_triangle()
    geom = geometry.precompute_mesh_geometry(mesh)
    q = _sample_mesh_grid(mesh, resolution=6)

    ref = _reference_smoothed_alpha_zero(q, mesh, geom)
    simplified = simplified_potential_numba.simplified_smoothed_offset_potential_numba(
        q, mesh, geom,
        alpha=0.0,
        localized=False,
        one_sided=False,
    )

    _assert_match(ref, simplified)


def test_simplified_potential_numba_two_faces_matches_smoothed_alpha_zero():
    mesh = _mesh_two_faces()
    geom = geometry.precompute_mesh_geometry(mesh)
    q = _sample_mesh_grid(mesh, resolution=6)

    ref = _reference_smoothed_alpha_zero(q, mesh, geom)
    simplified = simplified_potential_numba.simplified_smoothed_offset_potential_numba(
        q, mesh, geom,
        alpha=0.0,
        localized=False,
        one_sided=False,
    )

    _assert_match(ref, simplified)


def test_simplified_potential_numba_tetrahedron_matches_smoothed_alpha_zero():
    mesh = _mesh_tetrahedron()
    geom = geometry.precompute_mesh_geometry(mesh)
    q = _sample_mesh_grid(mesh, resolution=6)

    ref = _reference_smoothed_alpha_zero(q, mesh, geom)
    simplified = simplified_potential_numba.simplified_smoothed_offset_potential_numba(
        q, mesh, geom,
        alpha=0.0,
        localized=False,
        one_sided=False,
    )

    _assert_match(ref, simplified)


def test_simplified_potential_numba_validation_scenes_match_smoothed_alpha_zero():
    scenes = viz.build_validation_scene_specs()
    for scene in scenes:
        mesh = scene.mesh
        geom = geometry.precompute_mesh_geometry(mesh)
        q = _sample_mesh_grid(mesh, resolution=6)

        ref = _reference_smoothed_alpha_zero(q, mesh, geom)
        simplified = simplified_potential_numba.simplified_smoothed_offset_potential_numba(
            q, mesh, geom,
            alpha=0.0,
            localized=False,
            one_sided=False,
        )

        _assert_match(ref, simplified)


def test_simplified_potential_numba_tetrahedron_matches_formula():
    mesh = _mesh_tetrahedron()
    geom = geometry.precompute_mesh_geometry(mesh)
    q = _sample_mesh_grid(mesh, resolution=6)

    ref = _expected_simplified_potential(q, mesh)
    simplified = simplified_potential_numba.simplified_smoothed_offset_potential_numba(
        q, mesh, geom,
        alpha=0.0,
        localized=False,
        one_sided=False,
    )

    _assert_match(ref, simplified)


def _simplified_cpp_values(q, mesh):
    return potential_cpp.simplified_smoothed_offset_potential_cpp(
        q, mesh.V, mesh.faces,
        alpha=0.0,
        localized=False,
        one_sided=False,
    )


def _simplified_cpp_tinyad_values(q, mesh):
    values = np.zeros(q.shape[0], dtype=float)
    for i in range(q.shape[0]):
        value, _grad = potential_cpp.simplified_smoothed_offset_potential_cpp_tinyad(
            q[i], mesh.V, mesh.faces,
            alpha=0.0,
            localized=False,
            one_sided=False,
        )
        values[i] = value
    return values


def test_simplified_potential_cpp_matches_numba():
    meshes = [_mesh_single_triangle(), _mesh_two_faces(), _mesh_tetrahedron()]
    meshes.extend(scene.mesh for scene in viz.build_validation_scene_specs())
    for mesh in meshes:
        geom = geometry.precompute_mesh_geometry(mesh)
        q = _sample_mesh_grid(mesh, resolution=6)

        ref = simplified_potential_numba.simplified_smoothed_offset_potential_numba(
            q, mesh, geom,
            alpha=0.0,
            localized=False,
            one_sided=False,
        )
        cpp_vals = _simplified_cpp_values(q, mesh)

        _assert_match(ref, cpp_vals)


def test_simplified_potential_cpp_tinyad_matches_numba():
    meshes = [_mesh_single_triangle(), _mesh_two_faces(), _mesh_tetrahedron()]
    meshes.extend(scene.mesh for scene in viz.build_validation_scene_specs())
    for mesh in meshes:
        geom = geometry.precompute_mesh_geometry(mesh)
        q = _sample_mesh_grid(mesh, resolution=6)

        ref = simplified_potential_numba.simplified_smoothed_offset_potential_numba(
            q, mesh, geom,
            alpha=0.0,
            localized=False,
            one_sided=False,
        )
        cpp_vals = _simplified_cpp_tinyad_values(q, mesh)

        _assert_match(ref, cpp_vals)

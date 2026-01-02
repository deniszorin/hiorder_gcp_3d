import numpy as np

import potential_cpp
import potential_numba


def _finite_diff_grad(func, q, eps=1e-6):
    grad = np.zeros(3)
    for i in range(3):
        dq = np.zeros(3)
        dq[i] = eps
        grad[i] = (func(q + dq) - func(q - dq)) / (2.0 * eps)
    return grad


def _finite_diff_grad_full(func, x, eps=1e-6):
    grad = np.zeros_like(x)
    for i in range(x.size):
        dx = np.zeros_like(x)
        dx[i] = eps
        grad[i] = (func(x + dx) - func(x - dx)) / (2.0 * eps)
    return grad


def _finite_diff_hess_full(func, x, eps=1e-5):
    hess = np.zeros((x.size, x.size))
    for i in range(x.size):
        dx = np.zeros_like(x)
        dx[i] = eps
        grad_plus = _finite_diff_grad_full(func, x + dx, eps)
        grad_minus = _finite_diff_grad_full(func, x - dx, eps)
        hess[:, i] = (grad_plus - grad_minus) / (2.0 * eps)
    return hess


def _pack_face_vars(q, face_points):
    return np.concatenate([q.ravel(), face_points.reshape(-1)])


def _unpack_face_vars(x):
    q = x[:3]
    face_points = x[3:].reshape(3, 3)
    return q, face_points


def _pack_edge_vars(q, edge_points):
    return np.concatenate([q.ravel(), edge_points.reshape(-1)])


def _unpack_edge_vars(x):
    q = x[:3]
    edge_points = x[3:].reshape(4, 3)
    return q, edge_points


def _pack_vertex_vars(q, p_v, neighbor_points):
    return np.concatenate([q.ravel(), p_v.ravel(), neighbor_points.reshape(-1)])


def _unpack_vertex_vars(x, k):
    q = x[:3]
    p_v = x[3:6]
    neighbor_points = x[6:].reshape(k, 3)
    return q, p_v, neighbor_points


def _edge_points_for_numba(face0, local0, face1=None):
    if local0 == 0:
        edge_p0, edge_p1, other0 = face0[0], face0[1], face0[2]
    elif local0 == 1:
        edge_p0, edge_p1, other0 = face0[1], face0[2], face0[0]
    else:
        edge_p0, edge_p1, other0 = face0[2], face0[0], face0[1]

    edge_points = np.zeros((4, 3))
    edge_points[0] = edge_p0
    edge_points[1] = edge_p1
    edge_points[2] = other0

    if face1 is not None:
        for p in face1:
            if not np.allclose(p, edge_p0) and not np.allclose(p, edge_p1):
                edge_points[3] = p
                break

    return edge_points


def test_potential_face_cpp_matches_numba():
    q = np.array([0.3, 0.2, 0.7])
    face_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.2, 0.8, 0.1],
        ]
    )
    params = potential_numba.PotentialParameters(
        alpha=0.2,
        p=2.0,
        epsilon=0.3,
        localized=False,
        one_sided=False,
    )
    ref = potential_numba.potential_face(q, face_points, params)
    params_cpp = potential_cpp.potential_parameters(
        alpha=params.alpha,
        p=params.p,
        epsilon=params.epsilon,
        localized=params.localized,
        one_sided=params.one_sided,
    )
    val = potential_cpp.potential_face(
        q,
        face_points,
        params=params_cpp,
    )
    np.testing.assert_allclose(val, ref, rtol=1e-6, atol=1e-6)


def test_potential_edge_cpp_matches_numba():
    q = np.array([0.1, 0.4, 0.3])
    face0 = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    face1 = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    local0 = 2
    has_f1 = True
    params = potential_numba.PotentialParameters(
        alpha=0.2,
        p=2.0,
        epsilon=0.3,
        localized=False,
        one_sided=False,
    )
    edge_points = _edge_points_for_numba(face0, local0, face1)
    ref = potential_numba.potential_edge(q, edge_points, has_f1, params)
    params_cpp = potential_cpp.potential_parameters(
        alpha=params.alpha,
        p=params.p,
        epsilon=params.epsilon,
        localized=params.localized,
        one_sided=params.one_sided,
    )
    val = potential_cpp.potential_edge(
        q,
        edge_points,
        has_f1,
        params=params_cpp,
    )
    np.testing.assert_allclose(val, ref, rtol=1e-6, atol=1e-6)


def test_potential_vertex_cpp_matches_numba():
    q = np.array([0.2, -0.1, 0.4])
    p_v = np.array([0.0, 0.0, 0.0])
    neighbor_points = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.1],
            [-1.0, 0.0, 0.2],
            [0.0, -1.0, 0.3],
        ]
    )
    is_boundary = False
    pointed_vertex = False
    params = potential_numba.PotentialParameters(
        alpha=0.2,
        p=2.0,
        epsilon=0.3,
        localized=False,
        one_sided=False,
    )
    ref = potential_numba.potential_vertex(
        q,
        p_v,
        neighbor_points,
        is_boundary,
        pointed_vertex,
        params,
    )
    params_cpp = potential_cpp.potential_parameters(
        alpha=params.alpha,
        p=params.p,
        epsilon=params.epsilon,
        localized=params.localized,
        one_sided=params.one_sided,
    )
    val = potential_cpp.potential_vertex(
        q,
        p_v,
        neighbor_points,
        is_boundary,
        pointed_vertex,
        params=params_cpp,
    )
    np.testing.assert_allclose(val, ref, rtol=1e-6, atol=1e-6)


def test_potential_face_tinyad_gradient_matches_finite_difference():
    q = np.array([0.3, 0.2, 0.7])
    face_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.2, 0.8, 0.1],
        ]
    )
    params_cpp = potential_cpp.potential_parameters(
        alpha=0.2,
        p=2.0,
        epsilon=0.3,
        localized=False,
        one_sided=False,
    )

    def func(x):
        return potential_cpp.potential_face(
            x,
            face_points,
            params=params_cpp,
        )

    value_ad, grad_ad = potential_cpp.potential_face_cpp_tinyad(
        q,
        face_points,
        params=params_cpp,
    )
    value_cpp = func(q)
    grad_fd = _finite_diff_grad(func, q)
    np.testing.assert_allclose(value_ad, value_cpp, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(grad_ad, grad_fd, rtol=1e-5, atol=1e-6)


def test_potential_edge_tinyad_gradient_matches_finite_difference():
    q = np.array([0.1, 0.4, 0.3])
    face0 = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    face1 = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    local0 = 2
    local1 = 0
    has_f1 = True
    params_cpp = potential_cpp.potential_parameters(
        alpha=0.2,
        p=2.0,
        epsilon=0.3,
        localized=False,
        one_sided=False,
    )
    edge_points = _edge_points_for_numba(face0, local0, face1)

    def func(x):
        return potential_cpp.potential_edge(
            x,
            edge_points,
            has_f1,
            params=params_cpp,
        )

    value_ad, grad_ad = potential_cpp.potential_edge_cpp_tinyad(
        q,
        edge_points,
        has_f1,
        params=params_cpp,
    )
    value_cpp = func(q)
    grad_fd = _finite_diff_grad(func, q)
    np.testing.assert_allclose(value_ad, value_cpp, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(grad_ad, grad_fd, rtol=1e-5, atol=1e-6)


def test_potential_vertex_tinyad_gradient_matches_finite_difference():
    q = np.array([0.2, -0.1, 0.4])
    p_v = np.array([0.0, 0.0, 0.0])
    neighbor_points = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.1],
            [-1.0, 0.0, 0.2],
            [0.0, -1.0, 0.3],
        ]
    )
    is_boundary = False
    pointed_vertex = False
    params_cpp = potential_cpp.potential_parameters(
        alpha=0.2,
        p=2.0,
        epsilon=0.3,
        localized=False,
        one_sided=False,
    )

    def func(x):
        return potential_cpp.potential_vertex(
            x,
            p_v,
            neighbor_points,
            is_boundary,
            pointed_vertex,
            params=params_cpp,
        )

    value_ad, grad_ad = potential_cpp.potential_vertex_cpp_tinyad(
        q,
        p_v,
        neighbor_points,
        is_boundary,
        pointed_vertex,
        params=params_cpp,
    )
    value_cpp = func(q)
    grad_fd = _finite_diff_grad(func, q)
    np.testing.assert_allclose(value_ad, value_cpp, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(grad_ad, grad_fd, rtol=1e-5, atol=1e-6)


def test_potential_face_grad_hess_matches_finite_difference():
    q = np.array([0.3, 0.2, 0.7])
    face_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.2, 0.8, 0.1],
        ]
    )
    params_cpp = potential_cpp.potential_parameters(
        alpha=0.2,
        p=2.0,
        epsilon=0.3,
        localized=False,
        one_sided=False,
    )
    x0 = _pack_face_vars(q, face_points)

    def func(x):
        q_i, face_i = _unpack_face_vars(x)
        return potential_cpp.potential_face(q_i, face_i, params=params_cpp)

    grad_ad, hess_ad = potential_cpp.potential_face_grad_hess(
        q, face_points, params=params_cpp
    )
    grad_fd = _finite_diff_grad_full(func, x0)
    hess_fd = _finite_diff_hess_full(func, x0)
    np.testing.assert_allclose(grad_ad, grad_fd, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(hess_ad, hess_fd, rtol=1e-5, atol=1e-5)


def test_potential_edge_grad_hess_matches_finite_difference():
    q = np.array([0.1, 0.4, 0.3])
    face0 = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    face1 = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    local0 = 2
    has_f1 = True
    edge_points = _edge_points_for_numba(face0, local0, face1)
    params_cpp = potential_cpp.potential_parameters(
        alpha=0.2,
        p=2.0,
        epsilon=0.3,
        localized=False,
        one_sided=False,
    )
    x0 = _pack_edge_vars(q, edge_points)

    def func(x):
        q_i, edge_i = _unpack_edge_vars(x)
        return potential_cpp.potential_edge(q_i, edge_i, has_f1, params=params_cpp)

    grad_ad, hess_ad = potential_cpp.potential_edge_grad_hess(
        q, edge_points, has_f1, params=params_cpp
    )
    grad_fd = _finite_diff_grad_full(func, x0)
    hess_fd = _finite_diff_hess_full(func, x0)
    np.testing.assert_allclose(grad_ad, grad_fd, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(hess_ad, hess_fd, rtol=1e-5, atol=1e-6)


def test_potential_vertex_grad_hess_matches_finite_difference():
    q = np.array([0.2, -0.1, 0.4])
    p_v = np.array([0.0, 0.0, 0.0])
    neighbor_points = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.1],
            [-1.0, 0.0, 0.2],
            [0.0, -1.0, 0.3],
        ]
    )
    is_boundary = False
    pointed_vertex = False
    params_cpp = potential_cpp.potential_parameters(
        alpha=0.2,
        p=2.0,
        epsilon=0.3,
        localized=False,
        one_sided=False,
    )
    x0 = _pack_vertex_vars(q, p_v, neighbor_points)

    def func(x):
        q_i, p_v_i, neighbors_i = _unpack_vertex_vars(x, neighbor_points.shape[0])
        return potential_cpp.potential_vertex(
            q_i, p_v_i, neighbors_i, is_boundary, pointed_vertex, params=params_cpp
        )

    grad_ad, hess_ad = potential_cpp.potential_vertex_grad_hess(
        q, p_v, neighbor_points, is_boundary, pointed_vertex, params=params_cpp
    )
    grad_fd = _finite_diff_grad_full(func, x0)
    hess_fd = _finite_diff_hess_full(func, x0)
    np.testing.assert_allclose(grad_ad, grad_fd, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(hess_ad, hess_fd, rtol=1e-5, atol=1e-5)

import numpy as np

import potential_cpp


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
    edge_points = x[3:].reshape(2, 3)
    return q, edge_points


def _pack_vertex_vars(q, p_v):
    return np.concatenate([q.ravel(), p_v.ravel()])


def _unpack_vertex_vars(x):
    q = x[:3]
    p_v = x[3:]
    return q, p_v


def test_simplified_potential_face_grad_hess_matches_finite_difference():
    q = np.array([0.25, 0.25, 0.5])
    face_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
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
        return potential_cpp.simplified_potential_face(
            q_i,
            face_i,
            params=params_cpp,
        )

    grad_ad, hess_ad = potential_cpp.simplified_potential_face_grad_hess(
        q, face_points, params=params_cpp
    )
    grad_fd = _finite_diff_grad_full(func, x0)
    hess_fd = _finite_diff_hess_full(func, x0)
    np.testing.assert_allclose(grad_ad, grad_fd, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(hess_ad, hess_fd, rtol=1e-5, atol=1e-5)


def test_simplified_potential_edge_grad_hess_matches_finite_difference():
    q = np.array([0.4, 0.2, 0.1])
    edge_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
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
        return potential_cpp.simplified_potential_edge(
            q_i,
            edge_i,
            params=params_cpp,
        )

    grad_ad, hess_ad = potential_cpp.simplified_potential_edge_grad_hess(
        q, edge_points, params=params_cpp
    )
    grad_fd = _finite_diff_grad_full(func, x0)
    hess_fd = _finite_diff_hess_full(func, x0)
    np.testing.assert_allclose(grad_ad, grad_fd, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(hess_ad, hess_fd, rtol=1e-5, atol=1e-6)


def test_simplified_potential_vertex_grad_hess_matches_finite_difference():
    q = np.array([0.2, -0.1, 0.4])
    p_v = np.array([0.0, 0.0, 0.0])
    params_cpp = potential_cpp.potential_parameters(
        alpha=0.2,
        p=2.0,
        epsilon=0.3,
        localized=False,
        one_sided=False,
    )
    x0 = _pack_vertex_vars(q, p_v)

    def func(x):
        q_i, p_v_i = _unpack_vertex_vars(x)
        return potential_cpp.simplified_potential_vertex(
            q_i,
            p_v_i,
            params=params_cpp,
        )

    grad_ad, hess_ad = potential_cpp.simplified_potential_vertex_grad_hess(
        q, p_v, params=params_cpp
    )
    grad_fd = _finite_diff_grad_full(func, x0)
    hess_fd = _finite_diff_hess_full(func, x0)
    np.testing.assert_allclose(grad_ad, grad_fd, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(hess_ad, hess_fd, rtol=1e-5, atol=1e-5)

"""Numba-friendly smoothed offset potential kernels."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numba import njit


ArrayF = np.ndarray
ArrayI = np.ndarray

singular_value = 1e12
eps = 1e-12


class NumbaConnectivity(NamedTuple):
    """Numba-friendly connectivity arrays."""

    V: ArrayF       # vertices  nv x 3
    faces: ArrayI   #faces   nf x 3 vertex indices CCW
    edges: ArrayI   # edges  ne x 2 vertex indices
    face_edges: ArrayI  # nf x 3 edge indices per face
    edge_faces: ArrayI  # ne x 3 face indices per edge, -1 if boundary
    vertex_face_offsets: ArrayI   # nv x 1  offsets in vertex_face_indices where the list of incident faces starts
    vertex_face_indices: ArrayI   #  contactenated array of incident faces indices for all vertices
    vertex_edge_offsets: ArrayI   # same for edges
    vertex_edge_indices: ArrayI


class PotentialParameters(NamedTuple):
    alpha: float
    p: float
    epsilon: float
    localized: bool
    one_sided: bool


# ****************************************************************************
# Basic geometry functions for numba to avoid function calls
# spelling out componentwise to avoide the same

@njit(cache=True)
def dot3(a: ArrayF, b: ArrayF) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@njit(cache=True)
def norm3(v: ArrayF) -> float:
    return np.sqrt(dot3(v, v))


@njit(cache=True)
def safe_norm(v: ArrayF) -> float:
    n = norm3(v)
    if n < eps:
        return eps
    return n


@njit(cache=True)
def unit_vec(v: ArrayF) -> ArrayF:
    n = safe_norm(v)
    out = np.empty(3, dtype=np.float64)
    out[0] = v[0] / n
    out[1] = v[1] / n
    out[2] = v[2] / n
    return out


@njit(cache=True)
def unit_dir(p0: ArrayF, p1: ArrayF) -> ArrayF:
    return unit_vec(p1 - p0)


@njit(cache=True)
def cross3(a: ArrayF, b: ArrayF) -> ArrayF:
    out = np.empty(3, dtype=np.float64)
    out[0] = a[1] * b[2] - a[2] * b[1]
    out[1] = a[2] * b[0] - a[0] * b[2]
    out[2] = a[0] * b[1] - a[1] * b[0]
    return out


@njit(cache=True)
def face_normal(p0: ArrayF, p1: ArrayF, p2: ArrayF) -> ArrayF:
    n = cross3(p1 - p0, p2 - p0)
    return n / safe_norm(n)


@njit(cache=True)
def face_edge_endpoints(p0: ArrayF, p1: ArrayF, p2: ArrayF, local_edge: int,
) -> tuple[ArrayF, ArrayF]:
    """
    Given  face vertices (p0,p1,p2) and local_edge index 0..2, return endpoints
    """
    if local_edge == 0:
        return p0, p1
    if local_edge == 1:
        return p1, p2
    return p2, p0


@njit(cache=True)
def face_edge_inward(
    n: ArrayF,
    p0: ArrayF, p1: ArrayF, p2: ArrayF,
    local_edge: int,
) -> ArrayF:
    """
    Given a face and local edge index 0..2, return a vector perp to the edge pointing inside.
    """
    edge_p0, edge_p1 = face_edge_endpoints(p0, p1, p2, local_edge)
    d_e = unit_vec(edge_p1 - edge_p0)
    return cross3(n, d_e)


@njit(cache=True)
def set_face_points(face_points: ArrayF, idx: int, p0: ArrayF, p1: ArrayF, p2: ArrayF) -> None:
    face_points[idx, 0] = p0
    face_points[idx, 1] = p1
    face_points[idx, 2] = p2



@njit(cache=True)
def edge_projection(
    q: ArrayF, p0: ArrayF, d_unit: ArrayF
) -> tuple[ArrayF, float, ArrayF]:
    """
    q: point, (p0, d_unit): edge anchor and unit direction
    returns: q's projection position P_e on the edge line, distance r_e, unit vector from P_e to q
    """
    # the choice of the direction on the edge does not affect
    # projected position P_e can use any
    t = dot3(q - p0, d_unit)
    P_e = p0 + t * d_unit
    diff = q - P_e
    r_e = norm3(diff)
    # P_e: projection of q to each edge line.
    # r_e: distance from q to each edge line.
    unit_Pe_to_q = diff / safe_norm(diff)
    return P_e, r_e, unit_Pe_to_q

# ****************************************************************************
# Potential blending/localization functions

@njit(cache=True)
def H_scalar(z: float) -> float:
    if z < -1.0:
        return 0.0
    if z > 1.0:
        return 1.0
    return ((2.0 - z) * (z + 1.0) ** 2) / 4.0


@njit(cache=True)
def H_alpha_scalar(t: float, alpha: float) -> float:
    return H_scalar(t / alpha)


@njit(cache=True)
def h_local_scalar(z: float) -> float:
    return (2.0 * z + 1.0) * (z - 1.0) ** 2


@njit(cache=True)
def h_epsilon_scalar(z: float, epsilon: float) -> float:
    return h_local_scalar(z / epsilon)

# ****************************************************************************
# Helpers for local to global and back vertex/edge index conversion 


@njit(cache=True)
def find_local_edge(conn: NumbaConnectivity, fidx: int, edge_idx: int) -> int:
    for i in range(3):
        if conn.face_edges[fidx, i] == edge_idx:
            return i
    return -1

# ****************************************************************************
# Mesh connectivity and geometry conversion to numba-friendly format


def _build_csr(adj_lists: list[list[int]]) -> tuple[ArrayI, ArrayI]:
    offsets = np.zeros(len(adj_lists) + 1, dtype=np.int64)
    total = 0
    for i, items in enumerate(adj_lists):
        offsets[i] = total
        total += len(items)
    offsets[len(adj_lists)] = total
    indices = np.empty(total, dtype=np.int64)
    k = 0
    for items in adj_lists:
        for item in items:
            indices[k] = item
            k += 1
    return offsets, indices


def _build_numba_connectivity(mesh) -> NumbaConnectivity:
    V = np.asarray(mesh.V, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    edges = np.asarray(mesh.edges, dtype=np.int64)

    edge_index = {}
    for eidx in range(edges.shape[0]):
        a = int(edges[eidx, 0])
        b = int(edges[eidx, 1])
        key = (a, b) if a < b else (b, a)
        edge_index[key] = eidx

    face_edges = np.zeros((faces.shape[0], 3), dtype=np.int64)
    for f in range(faces.shape[0]):
        v0 = int(faces[f, 0])
        v1 = int(faces[f, 1])
        v2 = int(faces[f, 2])
        pairs = ((v0, v1), (v1, v2), (v2, v0))
        for i, (a, b) in enumerate(pairs):
            key = (a, b) if a < b else (b, a)
            face_edges[f, i] = edge_index[key]

    edge_faces = np.full((edges.shape[0], 2), -1, dtype=np.int64)
    for eidx, faces_list in enumerate(mesh.edges_to_faces):
        if len(faces_list) > 0:
            edge_faces[eidx, 0] = faces_list[0]
        if len(faces_list) > 1:
            edge_faces[eidx, 1] = faces_list[1]

    vertex_face_offsets, vertex_face_indices = _build_csr(mesh.vertices_to_faces)
    vertex_edge_offsets, vertex_edge_indices = _build_csr(mesh.vertices_to_edges)

    return NumbaConnectivity(
        V=V,
        faces=faces,
        edges=edges,
        face_edges=face_edges,
        edge_faces=edge_faces,
        vertex_face_offsets=vertex_face_offsets,
        vertex_face_indices=vertex_face_indices,
        vertex_edge_offsets=vertex_edge_offsets,
        vertex_edge_indices=vertex_edge_indices,
    )


# ****************************************************************************
#  Potential directional terms Phi^{e,f}, Phi^{v,e} 

@njit(cache=True)
def phi_ef(
    q: ArrayF,
    n: ArrayF, p0: ArrayF, p1: ArrayF, p2: ArrayF,
    local_edge: int,
) -> float:
    edge_p0, edge_p1 = face_edge_endpoints(p0, p1, p2, local_edge)
    d_unit = unit_dir(edge_p0, edge_p1)
    _P_e, _r_e, unit_Pe_to_q = edge_projection(q, edge_p0, d_unit)
    # Phi^{e,f} := (q - P_e)_+ dot (n x d_e[i]) per face
    edge_inward = face_edge_inward(n, p0, p1, p2, local_edge)
    return dot3(unit_Pe_to_q, edge_inward)


@njit(cache=True)
def phi_ve(q: ArrayF, p0: ArrayF, p1: ArrayF, d_unit: ArrayF) -> tuple[float, float]:
    """Phi^{0,e} and Phi^{1,-e} for each edge, used by edge/vertex terms.
    """
    unit0 = unit_dir(p0, q)
    unit1 = unit_dir(p1, q)
    # Phi^{i,e}, i=0,1 factors per edge
    phi0 = dot3(unit0, d_unit)
    phi1 = dot3(unit1, -d_unit)
    return phi0, phi1

# ****************************************************************************
#  Functions to check if a point is outside a given face, edge, vertex

@njit(cache=True)
def outside_face_scalar(r_f: float) -> bool:
    """
    Assumes that signed distance to the face is given (computed elsewhere) and returns if it is positive
    """
    return r_f > 0.0


@njit(cache=True)
def edge_normal_from_faces(face_points: ArrayF, has_f1: bool) -> ArrayF:
    f0_p0, f0_p1, f0_p2 = face_points[0]
    n_sum = face_normal(f0_p0, f0_p1, f0_p2)
    if has_f1:
        f1_p0, f1_p1, f1_p2 = face_points[1]
        n_sum += face_normal(f1_p0, f1_p1, f1_p2)
    n_norm = norm3(n_sum)
    if n_norm > eps:
        n_sum = n_sum / n_norm
    return n_sum


@njit(cache=True)
def outside_edge_scalar(
    q: ArrayF,
    r_e: float, P_e: ArrayF, edge_normal: ArrayF,
    r_f0: float, phi_ef0: float, has_f1: bool, r_f1: float, phi_ef1: float,
) -> bool:
    """
    Assumes that that the distance to the edge r_e, projection P_e, 
    as well as  signed distances to faces f0, f1, and directional factors Phi^{e,fi}, i=0,1 
    are give, implements the logic of the local outside test: 
    determine if one of the halfplanes of f0, f1, or the edge iself is closest, 
    use the test for the closest element: signed distance for face, and dot product with 
    the edge normal (this works because the volume where the edge is closest is within pi/2 of the normal)  
    """
    # initialize r_min with the distance to the edge
    r_min = r_e
    # check where the distance to f0 is less than edge
    # if the projection is within the halfplane (Phi^{e,f0} > 0)
    C_EDGE = 2 
    C_FACE_0 = 0 
    C_FACE_1 = 1
    closest_elt = C_EDGE
    if phi_ef0 > 0.0:
        r0 = abs(r_f0)
        if r0 < r_min:
            r_min = r0
            closest_elt = C_FACE_0
    # same for f1
    if has_f1 and phi_ef1 > 0.0:
        r1 = abs(r_f1)
        if r1 < r_min:
            r_min = r1
            closest_elt = C_FACE_1
    # if f0 is closest, check if  signed distance to it  r_{f_0} is positive
    if  closest_elt == C_FACE_0:
        return r_f0 > 0.0
    # if f1 is closest, check if  signed distance to it  r_{f_1} is positive
    if closest_elt == C_FACE_1:
        return r_f1 > 0.0
    # if the edge itself is closest,  check dot product with the average normal
    # it always points outside, and the sector where edge is closest is within pi/2
    # of the average normal.
    return dot3(q - P_e, edge_normal) > 0.0


@njit(cache=True)
def outside_vertex_scalar(
    q: ArrayF,
    r_v: float,
    r_f_min_signed: float,
    face_min: int,
    r_e_min: float,
    edge_min: int,
    P_e_min: ArrayF,
    edge_normal_min: ArrayF,
    pointed_vertex: bool,
) -> bool:
    """
    Assumes distance to the vertex, 
    signed distance to the closest face,
    distance and projection to the closest edge are given. 
    Determines which element (closest face, closest edge or vertex) is closest, and 
    then does the outside check based on the element.
    """
    # get closest face sector and edge ray if any
    r_f_min_abs = 1e30
    if face_min >= 0:
        r_f_min_abs = abs(r_f_min_signed)
    r_min_fe = r_f_min_abs
    if r_e_min < r_min_fe:
        r_min_fe = r_e_min

    use_vertex = r_v < r_min_fe
    use_face = (face_min >= 0) and (r_f_min_abs <= r_e_min) and (not use_vertex)
    use_edge = (edge_min >= 0) and (not use_face) and (not use_vertex)

    if use_face:
        return r_f_min_signed > 0.0
    if use_edge:
        return dot3(q - P_e_min, edge_normal_min) > 0.0
    # if any points left unassigned after a pass over all edges and faces, use
    # the pointed-vertex flag for those vertex-closest queries.
    # the reason for this is that if the vertex is closest, this means q 
    # is in the polar cone and the pointed-vertex flag indicates if this 
    # cones is inside or outside (the whole cone has to be on one side)
    return pointed_vertex

# ****************************************************************************
# Potential evaluation, face, edge, vertex components

@njit(cache=True)
def potential_face(
    q: ArrayF, face_points: ArrayF,
    params: PotentialParameters,
) -> float:
    """
    Evaluate the potential from a face, at point q
    Parameters are the same as for the top-level potential function
    returns the value of the potential
    """
    p0, p1, p2 = face_points
    n = face_normal(p0, p1, p2)
    # signed distance to the face plane.
    r_f = dot3(q - p0, n)

    B = 1.0
    for local_edge in range(3):
        phi_ef_val = phi_ef(q, n, p0, p1, p2, local_edge)
        B *= H_alpha_scalar(phi_ef_val, params.alpha)

    denom = abs(r_f) ** params.p
    if denom <= eps:
        I_f = singular_value
    else:
        I_f = B / denom

    if params.one_sided and not outside_face_scalar(r_f):
        I_f = 0.0
    if params.localized:
        I_f *= h_epsilon_scalar(abs(r_f), params.epsilon)
    return I_f


@njit(cache=True)
def potential_edge(
    q: ArrayF, edge_points: ArrayF,
    has_f1: bool,
    params: PotentialParameters,
) -> float:
    """
    Evaluate the potential from an edge, at point q.
    edge_points[0:2] are the edge endpoints, edge_points[2] is f0's non-edge point,
    edge_points[3] is f1's non-edge point when has_f1 is True.
    """

    edge_p0,edge_p1  = edge_points[0],edge_points[1]
    f0_p0, f0_p1, f0_p2 = edge_p0, edge_p1, edge_points[2]

    d_unit = unit_dir(edge_p0, edge_p1)
    P_e, r_e, unit_Pe_to_q = edge_projection(q, edge_p0, d_unit)
    phi0, phi1 = phi_ve(q, edge_p0, edge_p1, d_unit)

    phi_ef0 = r_f0 = h_face_0 = 0.0
    n0            = face_normal(f0_p0, f0_p1, f0_p2)
    edge_inward_0 = face_edge_inward(n0, f0_p0, f0_p1, f0_p2, 0)
    phi_ef0       = dot3(unit_Pe_to_q, edge_inward_0)
    h_face_0      = H_alpha_scalar(phi_ef0, params.alpha)
    r_f0          = dot3(q - f0_p0, n0)
    if params.one_sided:
        h_face_0 *= outside_face_scalar(r_f0)

    phi_ef1 = r_f1 = h_face_1 = 0.0
    if has_f1:
        f1_p0, f1_p1, f1_p2 = edge_p1, edge_p0, edge_points[3]
        n1            = face_normal(f1_p0, f1_p1, f1_p2)
        edge_inward_1 = face_edge_inward(n1, f1_p0, f1_p1, f1_p2, 0)
        phi_ef1       = dot3(unit_Pe_to_q, edge_inward_1)
        h_face_1      = H_alpha_scalar(phi_ef1, params.alpha)
        r_f1          = dot3(q - f1_p0, n1)
        if params.one_sided:
            h_face_1 *= outside_face_scalar(r_f1)

    B_edge = (1.0 - h_face_0 - h_face_1) * H_alpha_scalar(phi0, params.alpha) * H_alpha_scalar(phi1, params.alpha)

    # distance to edge r_e already computed per edge
    denom = r_e ** params.p
    if denom <= eps:
        I_e = singular_value
    else:
        I_e = B_edge / denom

    if params.one_sided:
        face_points = np.empty((2, 3, 3), dtype=np.float64)
        set_face_points(face_points, 0, f0_p0, f0_p1, f0_p2)
        if has_f1:
            set_face_points(face_points, 1, f1_p0, f1_p1, f1_p2)
        edge_n = edge_normal_from_faces(face_points, has_f1)
        if not outside_edge_scalar(
            q,
            r_e, P_e, edge_n,
            r_f0, phi_ef0, has_f1, r_f1, phi_ef1,
        ):
            I_e = 0.0
    if params.localized:
        I_e *= h_epsilon_scalar(r_e, params.epsilon)
    return I_e


@njit(cache=True)
def vertex_face_term(
    q: ArrayF, p_v: ArrayF, neighbor_points: ArrayF, is_boundary: bool,
    alpha: float, one_sided: bool,
) -> tuple[float, float, int]:
    """
    Computes directional term for a vertex potential, corresponding to the sum over faces
    (see the writeup)
    Along the way, computes closest face to q and signed distance to this face, needed
    for outside_vertex.
    """

    # get closest face sector and edge ray if any
    face_term = 0.0
    # initialize to inf
    r_f_min_signed = 0.0
    # local index of closest face
    face_min = -1
    r_f_min = 1e30

    k = neighbor_points.shape[0]
    if k < 2:
        return 0.0, 0.0, -1

    limit = k - 1 if is_boundary else k
    for i in range(limit):
        p_prev = neighbor_points[i]
        p_next = neighbor_points[(i + 1) % k]
        p0, p1, p2 = p_next, p_v, p_prev
        n = face_normal(p0, p1, p2)

        # face directional factor affecting the vertex (incident edges only)
        phi0 = phi_ef(q, n, p0, p1, p2, 0)
        phi1 = phi_ef(q, n, p0, p1, p2, 1)
        h0   = H_alpha_scalar(phi0, alpha)
        h1   = H_alpha_scalar(phi1, alpha)

        r_f = dot3(q - p0, n)
        if one_sided:
            outside_face = outside_face_scalar(r_f)
            h0 *= outside_face
            h1 *= outside_face

        face_term += h0 * h1
        # is the projection inside the face, determined by Phi^{e_i,f} signs, i= 0,1
        if phi0 > 0.0 and phi1 > 0.0:
            # if it is, then compare to the current min distance, and replace if less
            if abs(r_f) < r_f_min:
                r_f_min = abs(r_f)
                face_min = i
                r_f_min_signed = r_f

    return face_term, r_f_min_signed, face_min


@njit(cache=True)
def vertex_edge_term(
    q: ArrayF, p_v: ArrayF, neighbor_points: ArrayF, is_boundary: bool,
    alpha: float, one_sided: bool,
) -> tuple[float, float, int, ArrayF, ArrayF]:
    """
    Computes directional term for a vertex potential, corresponding to the sum over edges
    (see the writeup)
    Along the way, computes closest edge to q, needed for outside_vertex, the distance to the edge, 
    and the projection of q to the line of the edge.
    """
    edge_term = 0.0
    r_e_min = 1e30
    edge_min = -1
    P_e_min = np.zeros(3, dtype=np.float64)

    edge_normal_min = np.zeros(3, dtype=np.float64)
    k = neighbor_points.shape[0]
    if k == 0:
        return edge_term, r_e_min, edge_min, P_e_min, edge_normal_min

    for i in range(k):
        p_i = neighbor_points[i]
        d_unit = unit_dir(p_v, p_i)
        P_e, r_e, unit_Pe_to_q = edge_projection(q, p_v, d_unit)

        #  Phi^{v,e} terms
        phi_v, _phi_other = phi_ve(q, p_v, p_i, d_unit)
        h_v = H_alpha_scalar(phi_v, alpha)

        has_prev = (i > 0) or (not is_boundary)
        has_next = (i < k - 1) or (not is_boundary)
        prev_idx = i - 1 if i > 0 else k - 1
        next_idx = i + 1 if i < k - 1 else 0

        face_points = np.zeros((2, 3, 3), dtype=np.float64)
        local_edges = np.zeros(2, dtype=np.int64)
        face_count = 0
        if has_prev:
            p_prev = neighbor_points[prev_idx]
            set_face_points(face_points, face_count, p_i, p_v, p_prev)
            local_edges[face_count] = 0
            face_count += 1
        if has_next:
            p_next = neighbor_points[next_idx]
            set_face_points(face_points, face_count, p_next, p_v, p_i)
            local_edges[face_count] = 1
            face_count += 1

        #  Phi^{e,f} terms
        phi_ef = np.zeros(2, dtype=np.float64)
        h_face = np.zeros(2, dtype=np.float64)
        r_f = np.zeros(2, dtype=np.float64)
        n_face = np.zeros((2, 3), dtype=np.float64)
        for j in range(face_count):
            p0, p1, p2 = face_points[j]
            n =           face_normal(p0, p1, p2)
            edge_inward = face_edge_inward(n, p0, p1, p2, local_edges[j])
            phi_ef[j] =   dot3(unit_Pe_to_q, edge_inward)
            h_face[j] =   H_alpha_scalar(phi_ef[j], alpha)
            r_f[j] =      dot3(q - p0, n)
            if one_sided:
                h_face[j] *= outside_face_scalar(r_f[j])
            n_face[j] = n

        phi_ef0 = phi_ef[0]
        h_face_0 = h_face[0]
        r_f0 = r_f[0]
        phi_ef1 = h_face_1 = r_f1 = 0.0
        has_f1 = False
        if face_count > 1:
            phi_ef1 = phi_ef[1]
            h_face_1 = h_face[1]
            r_f1 = r_f[1]
            has_f1 = True

        if one_sided:
            n_sum = np.copy(n_face[0])
            if has_f1:
                n_sum += n_face[1]
            n_norm = norm3(n_sum)
            if n_norm > eps:
                n_sum = n_sum / n_norm
            if not outside_edge_scalar(
                q,
                r_e, P_e, n_sum,
                r_f0, phi_ef0, has_f1, r_f1, phi_ef1,
            ):
                h_v = 0.0

        # complete part of the edge directional factor to be used for the vertex
        # it is different from the complete factor as it only uses h_v = H^alpha(Phi^{v,e})
        # for this vertex, not both
        edge_term += (1.0 - h_face_0 - h_face_1) * h_v

        # is the projection of q to the ray starting at vertex along the edge inside the ray
        if phi_v > 0.0 and r_e < r_e_min:
            # replace the distance if projection is inside and the distance is less
            r_e_min = r_e
            edge_min = i
            P_e_min = P_e
            if one_sided:
                edge_normal_min = n_sum

    return edge_term, r_e_min, edge_min, P_e_min, edge_normal_min


@njit(cache=True)
def potential_vertex(
    q: ArrayF, p_v: ArrayF, neighbor_points: ArrayF, is_boundary: bool,
    pointed_vertex: bool,
    params: PotentialParameters,
) -> float:
    "potential due to a vertex at point q"
    r_v = norm3(q - p_v)

    # denominator of the potential has a sum over directional terms over faces and edges computed here
    # these are also needed to determine local sidedeness

    face_term, r_f_min_signed, face_min = vertex_face_term(
        q, p_v, neighbor_points, is_boundary,
        params.alpha, params.one_sided,
    )

    edge_term, r_e_min, edge_min, P_e_min, edge_normal_min = vertex_edge_term(
        q, p_v, neighbor_points, is_boundary,
        params.alpha, params.one_sided,
    )

    if params.one_sided:
        if not outside_vertex_scalar(
            q,
            r_v,
            r_f_min_signed, face_min,
            r_e_min, edge_min, P_e_min, edge_normal_min,
            pointed_vertex,
        ):
            return 0.0

    denom = r_v ** params.p
    if denom <= eps:
        I_v = singular_value
    else:
        I_v = (1.0 - face_term - edge_term) / denom

    if params.localized:
        I_v *= h_epsilon_scalar(r_v, params.epsilon)
    return I_v

# ****************************************************************************
# Helper to extract unique edge and vertex lists from a face list

@njit(cache=True)
def get_vertices_and_edges(
    face_indices: ArrayI, conn: NumbaConnectivity,
) -> tuple[ArrayI, ArrayI]:
    """
    Go over the list of faces, extract edges and faces, 
    place them in lists of unique edges and faces; not using sets to keep this numba-compatible
    """
    ne = conn.edges.shape[0]
    nv = conn.V.shape[0]
    edge_mark = np.zeros(ne, dtype=np.uint8)
    vertex_mark = np.zeros(nv, dtype=np.uint8)
    edge_count = 0
    vertex_count = 0

    for idx in range(face_indices.size):
        fidx = face_indices[idx]
        for i in range(3):
            edge_idx = conn.face_edges[fidx, i]
            if edge_mark[edge_idx] == 0:
                edge_mark[edge_idx] = 1
                edge_count += 1
        v0, v1, v2 = conn.faces[fidx]
        if vertex_mark[v0] == 0:
            vertex_mark[v0] = 1
            vertex_count += 1
        if vertex_mark[v1] == 0:
            vertex_mark[v1] = 1
            vertex_count += 1
        if vertex_mark[v2] == 0:
            vertex_mark[v2] = 1
            vertex_count += 1

    edge_list = np.empty(edge_count, dtype=np.int64)
    vertex_list = np.empty(vertex_count, dtype=np.int64)
    edge_count = 0
    vertex_count = 0
    edge_mark[:] = 0
    vertex_mark[:] = 0

    for idx in range(face_indices.size):
        fidx = face_indices[idx]
        for i in range(3):
            edge_idx = conn.face_edges[fidx, i]
            if edge_mark[edge_idx] == 0:
                edge_mark[edge_idx] = 1
                edge_list[edge_count] = edge_idx
                edge_count += 1
        v0, v1, v2 = conn.faces[fidx]
        if vertex_mark[v0] == 0:
            vertex_mark[v0] = 1
            vertex_list[vertex_count] = v0
            vertex_count += 1
        if vertex_mark[v1] == 0:
            vertex_mark[v1] = 1
            vertex_list[vertex_count] = v1
            vertex_count += 1
        if vertex_mark[v2] == 0:
            vertex_mark[v2] = 1
            vertex_list[vertex_count] = v2
            vertex_count += 1

    return edge_list, vertex_list


# ****************************************************************************
#  Main potential calls


@njit(cache=True)
def smoothed_offset_potential_point(
    q: ArrayF, face_indices: ArrayI,
    conn: NumbaConnectivity, pointed_vertices: ArrayF,
    params: PotentialParameters,
    include_faces: bool, include_edges: bool, include_vertices: bool,
) -> float:
    """
    Compute potential from faces,edges and vertices given by the face list face_indices at point q.
    See smoothed_offset_potential for arguments.
    """
    if not (include_faces or include_edges or include_vertices):
        return 0.0

    face_sum = 0.0
    edge_sum = 0.0
    vertex_sum = 0.0

    edge_list, vertex_list = get_vertices_and_edges(face_indices, conn)
    if include_faces:
        for i in range(face_indices.size):
            fidx = face_indices[i]
            face_points = conn.V[conn.faces[fidx, :]]
            face_sum += potential_face(q, face_points, params)

    if include_edges:
        for i in range(edge_list.size):
            edge_idx = edge_list[i]
            f0, f1 = conn.edge_faces[edge_idx]
            if f0 < 0 and f1 >= 0:
                f0, f1 = f1, -1
            has_f1 = f1 >= 0
            local0 = find_local_edge(conn, f0, edge_idx)
            edge_v0 =  conn.faces[f0, local0]
            edge_v1 =  conn.faces[f0, (local0 + 1) % 3]
            other_v0 = conn.faces[f0, (local0 + 2) % 3]

            edge_points = np.empty((4, 3), dtype=np.float64)
            edge_points[0] = conn.V[edge_v0]
            edge_points[1] = conn.V[edge_v1]
            edge_points[2] = conn.V[other_v0]
            if has_f1:
                local1 = find_local_edge(conn, f1, edge_idx)
                other_v1 = conn.faces[f1, (local1 + 2) % 3]
                edge_points[3] = conn.V[other_v1]
            edge_sum += potential_edge(
                q, edge_points,
                has_f1,
                params,
            )

    if include_vertices:
        for i in range(vertex_list.size):
            v_idx = vertex_list[i]
            p_v = conn.V[v_idx]
            start_e = conn.vertex_edge_offsets[v_idx]
            end_e = conn.vertex_edge_offsets[v_idx + 1]
            k = end_e - start_e
            neighbor_points = np.empty((k, 3), dtype=np.float64)
            boundary_count = 0
            for j in range(k):
                edge_idx = conn.vertex_edge_indices[start_e + j]
                a, b = conn.edges[edge_idx]
                neighbor_idx = b if a == v_idx else a
                neighbor_points[j] = conn.V[neighbor_idx]
                if conn.edge_faces[edge_idx, 0] < 0 or conn.edge_faces[edge_idx, 1] < 0:
                    boundary_count += 1
            is_boundary = boundary_count == 2
            pointed_vertex = pointed_vertices[v_idx]
            vertex_sum += potential_vertex(
                q, p_v, neighbor_points, is_boundary,
                pointed_vertex, params,
            )

    return face_sum + edge_sum + vertex_sum


def smoothed_offset_potential_numba(
    q: ArrayF,
    mesh, geom,
    alpha: float = 0.1, p: float = 2.0, epsilon: float = 0.1,
    include_faces: bool = True, include_edges: bool = True, include_vertices: bool = True,
    localized: bool = False, one_sided: bool = False,
) -> ArrayF:
    """
    Compute potential from a mesh at points in the array q.
    conn, geom: connectivity and mesh geometry, see geometry.py
    alpha, p, epsilon: potential parameters, see potential_description.tex
    include_faces, include_edges, include_vertices: turn on/off the components of the potential.
    localized: enable h^epsilon factor.
    one_sided: enable localized sidedness checks in the potential.
    """

    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        q = q[None, :]
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError("q must have shape (3,) or (nq, 3).")

    conn = _build_numba_connectivity(mesh)
    pointed_vertices = np.asarray(geom.pointed_vertices, dtype=np.bool_)
    face_indices = np.arange(conn.faces.shape[0], dtype=np.int64)
    params = PotentialParameters(alpha, p, epsilon, localized, one_sided)

    nq = q.shape[0]
    out = np.zeros(nq, dtype=np.float64)
    for i in range(nq):
        out[i] = smoothed_offset_potential_point(
            q[i], face_indices,
            conn, pointed_vertices,
            params,
            include_faces, include_edges, include_vertices,
        )
    return out

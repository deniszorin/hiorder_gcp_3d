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


class NumbaGeometry(NamedTuple):
    """Numba-friendly geometry arrays."""

    normals: ArrayF  # nf x 3 
    edge_inward: ArrayF  # nf x 3 x 3 inward pointing unit vectors for each edge 
    edge_normals: ArrayF  # ne x 3  average of face normals per edge
    pointed_vertices: ArrayF  # nv boolean pointed vertex flags


# ****************************************************************************
# Basic geometry functions for numba to avoid function calls

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
def project_point_to_line(q: ArrayF, p0: ArrayF, p1: ArrayF) -> tuple[ArrayF, float, ArrayF]:
    d = p1 - p0
    d_norm = safe_norm(d)
    d_unit = d / d_norm
    # the choice of the direction on the edge does not affect
    # projected position P_e can use any
    t = dot3(q - p0, d_unit)
    P_e = p0 + t * d_unit
    diff = q - P_e
    r_e = norm3(diff)
    return P_e, r_e, d_unit


@njit(cache=True)
def edge_projection(
    q: ArrayF, p0: ArrayF, p1: ArrayF
) -> tuple[ArrayF, float, ArrayF, ArrayF]:
    P_e, r_e, d_unit = project_point_to_line(q, p0, p1)
    # P_e: projection of q to each edge line.
    # r_e: distance from q to each edge line.
    diff = q - P_e
    unit_edge = diff / safe_norm(diff)
    return P_e, r_e, d_unit, unit_edge

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
def face_edge_vertices(conn: NumbaConnectivity, fidx: int, local_edge: int) -> tuple[int, int]:
    v0 = conn.faces[fidx, 0]
    v1 = conn.faces[fidx, 1]
    v2 = conn.faces[fidx, 2]
    if local_edge == 0:
        return v0, v1
    if local_edge == 1:
        return v1, v2
    return v2, v0


@njit(cache=True)
def find_local_edge(conn: NumbaConnectivity, fidx: int, edge_idx: int) -> int:
    for i in range(3):
        if conn.face_edges[fidx, i] == edge_idx:
            return i
    return -1


# ****************************************************************************
#  Potential compoents 

@njit(cache=True)
def phi_ef(
    q: ArrayF,
    conn: NumbaConnectivity,
    geom: NumbaGeometry,
    fidx: int,
    local_edge: int,
) -> float:
    a, b = face_edge_vertices(conn, fidx, local_edge)
    p0 = conn.V[a]
    p1 = conn.V[b]
    _P_e, _r_e, _d_unit, unit_edge = edge_projection(q, p0, p1)
    # Phi^{e,f} := (q - P_e)_+ dot (n x d_e[i]) per face
    return dot3(unit_edge, geom.edge_inward[fidx, local_edge])


@njit(cache=True)
def phi_ve(q: ArrayF, p0: ArrayF, p1: ArrayF, d_unit: ArrayF) -> tuple[float, float]:
    # Phi^{0,e} and Phi^{1,-e} for each edge, used by edge/vertex terms.
    vec0 = q - p0
    vec1 = q - p1
    unit0 = unit_vec(vec0)
    unit1 = unit_vec(vec1)
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
    q: ArrayF, v_idx: int,
    conn: NumbaConnectivity, geom: NumbaGeometry,
    r_v: float,
    r_f_min_signed: float, face_min: int,
    r_e_min: float, edge_min: int, P_e_min: ArrayF,
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
        return dot3(q - P_e_min, geom.edge_normals[edge_min]) > 0.0
    # if any points left unassigned after a pass over all edges, use
    # the pointed-vertex flag for those vertex-closest queries.
    # the reason for this is that if the vertex is closest, this means q 
    # is in the polar cone and the pointed-vertex flag indicates if this 
    # cones is inside or outside (the whole cone has to be on one side)
    return geom.pointed_vertices[v_idx]


@njit(cache=True)
def potential_face(
    q: ArrayF, fidx: int,
    conn: NumbaConnectivity, geom: NumbaGeometry,
    alpha: float, p: float, epsilon: float,
    localized: bool, one_sided: bool,
) -> float:
    """
    """
    v0 = conn.faces[fidx, 0]
    p0 = conn.V[v0]
    n = geom.normals[fidx]
    # signed distance to the face plane.
    r_f = dot3(q - p0, n)
    r_f_abs = abs(r_f)

    B = 1.0
    for local_edge in range(3):
        phi_ef_val = phi_ef(q, conn, geom, fidx, local_edge)
        B *= H_alpha_scalar(phi_ef_val, alpha)

    denom = r_f_abs ** p
    if denom <= eps:
        I_f = singular_value
    else:
        I_f = B / denom

    if one_sided and not outside_face_scalar(r_f):
        I_f = 0.0
    if localized:
        I_f *= h_epsilon_scalar(r_f_abs, epsilon)
    return I_f


@njit(cache=True)
def potential_edge(
    q: ArrayF, edge_idx: int,
    conn: NumbaConnectivity, geom: NumbaGeometry,
    alpha: float, p: float, epsilon: float,
    localized: bool, one_sided: bool,
) -> float:
    a = conn.edges[edge_idx, 0]
    b = conn.edges[edge_idx, 1]
    p0 = conn.V[a]
    p1 = conn.V[b]
    P_e, r_e, d_unit, unit_edge = edge_projection(q, p0, p1)
    phi0, phi1 = phi_ve(q, p0, p1, d_unit)

    f0 = conn.edge_faces[edge_idx, 0]
    f1 = conn.edge_faces[edge_idx, 1]
    has_f1 = f1 >= 0

    phi_ef0 = 0.0
    r_f0 = 0.0
    h_face_0 = 0.0
    if f0 >= 0:
        local0 = find_local_edge(conn, f0, edge_idx)
        phi_ef0 = dot3(unit_edge, geom.edge_inward[f0, local0])
        h_face_0 = H_alpha_scalar(phi_ef0, alpha)
        v0 = conn.faces[f0, 0]
        r_f0 = dot3(q - conn.V[v0], geom.normals[f0])
        if one_sided:
            h_face_0 *= outside_face_scalar(r_f0)

    phi_ef1 = 0.0
    r_f1 = 0.0
    h_face_1 = 0.0
    if has_f1:
        local1 = find_local_edge(conn, f1, edge_idx)
        phi_ef1 = dot3(unit_edge, geom.edge_inward[f1, local1])
        h_face_1 = H_alpha_scalar(phi_ef1, alpha)
        v1 = conn.faces[f1, 0]
        r_f1 = dot3(q - conn.V[v1], geom.normals[f1])
        if one_sided:
            h_face_1 *= outside_face_scalar(r_f1)

    B_edge = (1.0 - h_face_0 - h_face_1) * H_alpha_scalar(phi0, alpha) * H_alpha_scalar(phi1, alpha)

    # distance to edge r_e already computed per edge
    denom = r_e ** p
    if denom <= eps:
        I_e = singular_value
    else:
        I_e = B_edge / denom

    if one_sided:
        if not outside_edge_scalar(
            q,
            r_e, P_e, geom.edge_normals[edge_idx],
            r_f0, phi_ef0, has_f1, r_f1, phi_ef1,
        ):
            I_e = 0.0
    if localized:
        I_e *= h_epsilon_scalar(r_e, epsilon)
    return I_e


@njit(cache=True)
def vertex_face_term(
    q: ArrayF, v_idx: int,
    conn: NumbaConnectivity, geom: NumbaGeometry,
    alpha: float, one_sided: bool,
) -> tuple[float, float, int]:
    # get closest face sector and edge ray if any
    face_term = 0.0
    # initialize to inf
    r_f_min_signed = 0.0
    # ids of closest faces
    face_min = -1
    r_f_min = 1e30

    start_f = conn.vertex_face_offsets[v_idx]
    end_f = conn.vertex_face_offsets[v_idx + 1]
    for idx in range(start_f, end_f):
        f = conn.vertex_face_indices[idx]
        v0 = conn.faces[f, 0]
        v1 = conn.faces[f, 1]
        v2 = conn.faces[f, 2]
        # figure out which edges are incident at the vertex
        if v_idx == v0:
            e0 = 0
            e1 = 2
        elif v_idx == v1:
            e0 = 0
            e1 = 1
        elif v_idx == v2:
            e0 = 1
            e1 = 2
        else:
            continue

        # face directional factor affecting the vertex (incident edges only)
        phi0 = phi_ef(q, conn, geom, f, e0)
        phi1 = phi_ef(q, conn, geom, f, e1)
        h0 = H_alpha_scalar(phi0, alpha)
        h1 = H_alpha_scalar(phi1, alpha)

        v_face = conn.faces[f, 0]
        r_f = dot3(q - conn.V[v_face], geom.normals[f])
        if one_sided:
            outside_face = outside_face_scalar(r_f)
            h0 *= outside_face
            h1 *= outside_face

        face_term += h0 * h1
        # is the projection inside the face, determined by Phi^{e_i,f} signs, i= 0,1
        if phi0 > 0.0 and phi1 > 0.0:
            r_abs = abs(r_f)
            # if it is, then compare to the current min distance, and replace if less
            if r_abs < r_f_min:
                r_f_min = r_abs
                face_min = f
                r_f_min_signed = r_f

    return face_term, r_f_min_signed, face_min


@njit(cache=True)
def vertex_edge_term(
    q: ArrayF, v_idx: int,
    conn: NumbaConnectivity, geom: NumbaGeometry,
    alpha: float, one_sided: bool,
) -> tuple[float, float, int, ArrayF]:
    """This function does two things at once: computes the sum of edge directional factors for the vertex, 
    and along the way computes the closest edge distance and edge  r_e_min, edge_min,  and projection on closest edge
    """ 

    edge_term = 0.0
    r_e_min = 1e30
    edge_min = -1
    P_e_min = np.zeros(3, dtype=np.float64)

    start_e = conn.vertex_edge_offsets[v_idx]
    end_e = conn.vertex_edge_offsets[v_idx + 1]
    # for eidx in m_vertices_to_edges
    for idx in range(start_e, end_e):
        edge_idx = conn.vertex_edge_indices[idx]
        a = conn.edges[edge_idx, 0]
        b = conn.edges[edge_idx, 1]
        p0 = conn.V[a]
        p1 = conn.V[b]
        P_e, r_e, d_unit, unit_edge = edge_projection(q, p0, p1)

        #  Phi^{v,e} terms
        if v_idx == a:
            phi_v, _phi_other = phi_ve(q, p0, p1, d_unit)
        else:
            _phi_other, phi_v = phi_ve(q, p0, p1, d_unit)
        h_v = H_alpha_scalar(phi_v, alpha)

        f0 = conn.edge_faces[edge_idx, 0]
        f1 = conn.edge_faces[edge_idx, 1]
        has_f1 = f1 >= 0

        #  Phi^{e,f} terms
        phi_ef0 = 0.0
        r_f0 = 0.0
        h_face_0 = 0.0
        if f0 >= 0:
            local0 = find_local_edge(conn, f0, edge_idx)
            phi_ef0 = dot3(unit_edge, geom.edge_inward[f0, local0])
            h_face_0 = H_alpha_scalar(phi_ef0, alpha)
            v0 = conn.faces[f0, 0]
            r_f0 = dot3(q - conn.V[v0], geom.normals[f0])
            if one_sided:
                h_face_0 *= outside_face_scalar(r_f0)

        phi_ef1 = 0.0
        r_f1 = 0.0
        h_face_1 = 0.0
        if has_f1:
            local1 = find_local_edge(conn, f1, edge_idx)
            phi_ef1 = dot3(unit_edge, geom.edge_inward[f1, local1])
            h_face_1 = H_alpha_scalar(phi_ef1, alpha)
            v1 = conn.faces[f1, 0]
            r_f1 = dot3(q - conn.V[v1], geom.normals[f1])
            if one_sided:
                h_face_1 *= outside_face_scalar(r_f1)

        if one_sided:
            if not outside_edge_scalar(
                q,
                r_e, P_e, geom.edge_normals[edge_idx],
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
            edge_min = edge_idx
            P_e_min = P_e

    return edge_term, r_e_min, edge_min, P_e_min


@njit(cache=True)
def potential_vertex(
    q: ArrayF, v_idx: int,
    conn: NumbaConnectivity, geom: NumbaGeometry,
    alpha: float, p: float, epsilon: float,
    localized: bool, one_sided: bool,
) -> float:
    "potential due to vertex v_idx at point q"
    r_v = norm3(q - conn.V[v_idx])

    # denominator of the potential has a sum over directional terms over faces and edges computed here
    # these are also needed to determine local sidedeness

    face_term, r_f_min_signed, face_min = vertex_face_term(q, v_idx, conn, geom, alpha, one_sided)

    edge_term, r_e_min, edge_min, P_e_min = vertex_edge_term(
        q, v_idx,
        conn, geom,
        alpha, one_sided,
    )

    if one_sided:
        if not outside_vertex_scalar(
            q, v_idx,
            conn, geom,
            r_v,
            r_f_min_signed, face_min,
            r_e_min, edge_min, P_e_min,
        ):
            return 0.0

    denom = r_v ** p
    if denom <= eps:
        I_v = singular_value
    else:
        I_v = (1.0 - face_term - edge_term) / denom

    if localized:
        I_v *= h_epsilon_scalar(r_v, epsilon)
    return I_v


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
        v0 = conn.faces[fidx, 0]
        v1 = conn.faces[fidx, 1]
        v2 = conn.faces[fidx, 2]
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
        v0 = conn.faces[fidx, 0]
        v1 = conn.faces[fidx, 1]
        v2 = conn.faces[fidx, 2]
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


@njit(cache=True)
def smoothed_offset_potential_point(
    q: ArrayF, face_indices: ArrayI,
    conn: NumbaConnectivity, geom: NumbaGeometry,
    alpha: float, p: float, epsilon: float,
    include_faces: bool, include_edges: bool, include_vertices: bool,
    localized: bool, one_sided: bool,
) -> float:
    if not (include_faces or include_edges or include_vertices):
        return 0.0

    face_sum = 0.0
    edge_sum = 0.0
    vertex_sum = 0.0

    edge_list, vertex_list = get_vertices_and_edges(face_indices, conn)
    if include_faces:
        for i in range(face_indices.size):
            face_sum += potential_face( q, face_indices[i], conn, geom, alpha, p, epsilon, localized, one_sided)

    if include_edges:
        for i in range(edge_list.size):
            edge_sum += potential_edge(q, edge_list[i], conn, geom, alpha, p, epsilon, localized, one_sided)

    if include_vertices:
        for i in range(vertex_list.size):
            vertex_sum += potential_vertex(q, vertex_list[i], conn, geom, alpha, p, epsilon, localized, one_sided)

    return face_sum + edge_sum + vertex_sum


@njit(cache=True)
def _smoothed_offset_potential_numba_impl(
    q: ArrayF, face_indices: ArrayI,
    conn: NumbaConnectivity, geom: NumbaGeometry,
    alpha: float, p: float, epsilon: float,
    include_faces: bool, include_edges: bool, include_vertices: bool,
    localized: bool, one_sided: bool,
) -> ArrayF:
    nq = q.shape[0]
    out = np.zeros(nq, dtype=np.float64)
    for i in range(nq):
        out[i] = smoothed_offset_potential_point(
            q[i], face_indices,
            conn, geom,
            alpha, p, epsilon,
            include_faces, include_edges, include_vertices,
            localized, one_sided,
        )
    return out


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


def _build_numba_geometry(geom) -> NumbaGeometry:
    return NumbaGeometry(
        normals=np.asarray(geom.normals, dtype=np.float64),
        edge_inward=np.asarray(geom.edge_inward, dtype=np.float64),
        edge_normals=np.asarray(geom.edge_normals, dtype=np.float64),
        pointed_vertices=np.asarray(geom.pointed_vertices, dtype=np.bool_),
    )


def smoothed_offset_potential_numba(
    q: ArrayF,
    mesh, geom,
    alpha: float = 0.1, p: float = 2.0, epsilon: float = 0.1,
    include_faces: bool = True, include_edges: bool = True, include_vertices: bool = True,
    localized: bool = False, one_sided: bool = False,
) -> ArrayF:
    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        q = q[None, :]
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError("q must have shape (3,) or (nq, 3).")

    conn = _build_numba_connectivity(mesh)
    geom_nb = _build_numba_geometry(geom)
    face_indices = np.arange(conn.faces.shape[0], dtype=np.int64)

    return _smoothed_offset_potential_numba_impl(
        q, face_indices,
        conn,geom_nb,
        alpha, p, epsilon,
        include_faces, include_edges, include_vertices,
        localized, one_sided,
    )

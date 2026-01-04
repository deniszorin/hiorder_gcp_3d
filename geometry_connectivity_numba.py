"""Shared numba helpers for potential kernels."""

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


@njit(cache=True)
def project_point_to_segment(
    q: ArrayF, p0: ArrayF, p1: ArrayF
) -> tuple[ArrayF, float, bool]:
    d = p1 - p0
    denom = dot3(d, d)
    t = dot3(q - p0, d) / denom
    if t <= 0.0:
        proj = p0
        inside = False
    elif t >= 1.0:
        proj = p1
        inside = False
    else:
        proj = p0 + t * d
        inside = True
    r = norm3(q - proj)
    return proj, r, inside


@njit(cache=True)
def distance_to_edge_line(q_minus_p: ArrayF, edge: ArrayF) -> float:
    t = dot3(q_minus_p, edge) / dot3(edge, edge)
    return norm3(q_minus_p - t * edge)


@njit(cache=True)
def triangle_distance(q: ArrayF, p0: ArrayF, p1: ArrayF, p2: ArrayF) -> float:
    e0, e1, e2 = p1 - p0, p2 - p1, p0 - p2
    n = cross3(e0, -e2)
    n_norm = norm3(n)
    if n_norm <= eps:
        return min(norm3(q - p0), norm3(q - p1), norm3(q - p2))
    else:
        signed_dist = dot3(n, q - p0)
        q_proj = q - (signed_dist / (n_norm * n_norm)) * n
        r_f = abs(signed_dist) / n_norm
        qp0,qp1,qp2 = q_proj - p0, q_proj - p1,q_proj - p2

        #  side tests for edge halfplanes, > 0 = inside the triangle
        s0 = dot3(qp0, cross3(n, e0))
        s1 = dot3(qp1, cross3(n, e1))
        s2 = dot3(qp2, cross3(n, e2))

        # "slab" tests for each edge, is a point on the same side as the edge
        #  with respect to perpendicular trough the edge endpoints
        t0_0 = dot3(qp0, e0)
        t0_1 = dot3(qp1, e1)
        t0_2 = dot3(qp2, e2)
        t1_0 = dot3(qp1, -e0)
        t1_1 = dot3(qp2, -e1)
        t1_2 = dot3(qp0, -e2)

        # 7 regions, triangle interior, closest to one of edges, closest to one of vertices
        if s0 >= -eps and s1 >= -eps and s2 >= -eps:
            return r_f

        if s0 < 0.0 and t0_0 >= -eps and t1_0 >= -eps:
            return distance_to_edge_line(q - p0, e0)
        if s1 < 0.0 and t0_1 >= -eps and t1_1 >= -eps:
            return distance_to_edge_line(q - p1, e1)
        if s2 < 0.0 and t0_2 >= -eps and t1_2 >= -eps:
            return distance_to_edge_line(q - p2, e2)

        if t0_0 < 0.0 and t1_2 < 0.0:
            return norm3(q - p0)
        if t0_1 < 0.0 and t1_0 < 0.0:
            return norm3(q - p1)
        if t0_2 < 0.0 and t1_1 < 0.0:
            return norm3(q - p2)

        raise ValueError("triangle_distance: region classification failed.")

# ****************************************************************************
# Helper for global to local vertex/edge index conversion 


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

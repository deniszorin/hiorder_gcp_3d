"""Numba-friendly smoothed offset potential kernels."""

from __future__ import annotations

import numpy as np
from numba import njit

from geometry_connectivity_numba import (
    ArrayF, ArrayI,
    NumbaConnectivity,
    singular_value, eps,
    dot3, norm3, safe_norm, unit_vec, unit_dir, cross3,
    face_normal, face_edge_endpoints, face_edge_inward,
    set_face_points, edge_projection,
    find_local_edge, _build_csr, _build_numba_connectivity,
    outside_face_scalar, edge_normal_from_faces,
    outside_edge_scalar, outside_vertex_scalar,
    get_vertices_and_edges,
)
from potential_parameters_numba import (
    PotentialParameters,
    H_scalar, H_alpha_scalar, h_local_scalar, h_epsilon_scalar,
)


# ****************************************************************************
# Potential directional terms Phi^{e,f}, Phi^{v,e}

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


def smoothed_offset_potential_accelerated(
    q: ArrayF,
    mesh, geom,
    alpha: float = 0.1, p: float = 2.0, epsilon: float = 0.1,
    include_faces: bool = True, include_edges: bool = True, include_vertices: bool = True,
    localized: bool = False, one_sided: bool = False,
) -> ArrayF:
    """
    Compute potential from a mesh at points in the array q using a VTK cell locator.
    Only intended for localized potentials.
    """

    if not localized:
        raise ValueError("smoothed_offset_potential_accelerated requires localized=True.")

    try:
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    except ImportError as exc:
        raise ImportError("vtk is required for accelerated potential.") from exc

    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        q = q[None, :]
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError("q must have shape (3,) or (nq, 3).")

    V = np.asarray(mesh.V, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (nf, 3).")

    vtk_faces = np.empty((faces.shape[0], 4), dtype=np.int64)
    vtk_faces[:, 0] = 3
    vtk_faces[:, 1:] = faces
    vtk_faces = vtk_faces.ravel()

    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(V, deep=True))
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)

    cells = vtk.vtkCellArray()
    cell_ids = numpy_to_vtkIdTypeArray(vtk_faces, deep=True)
    cells.SetCells(faces.shape[0], cell_ids)
    poly.SetPolys(cells)

    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(poly)
    locator.BuildLocator()

    id_list = vtk.vtkIdList()
    id_list.Allocate(faces.shape[0])
    closest_point = [0.0, 0.0, 0.0]
    cell_id = vtk.reference(0)
    sub_id = vtk.reference(0)
    dist2 = vtk.reference(0.0)

    conn = _build_numba_connectivity(mesh)
    pointed_vertices = np.asarray(geom.pointed_vertices, dtype=np.bool_)
    params = PotentialParameters(alpha, p, epsilon, True, one_sided)
    nq = q.shape[0]
    out = np.zeros(nq, dtype=np.float64)
    for i in range(nq):
        has_close = locator.FindClosestPointWithinRadius(
            q[i], epsilon, closest_point, cell_id, sub_id, dist2
        )
        if has_close == 0:
            out[i] = 0.0
            continue
        bounds = [
            q[i, 0] - epsilon, q[i, 0] + epsilon,
            q[i, 1] - epsilon, q[i, 1] + epsilon,
            q[i, 2] - epsilon, q[i, 2] + epsilon,
        ]
        id_list.Reset()
        locator.FindCellsWithinBounds(bounds, id_list)
        count = id_list.GetNumberOfIds()
        if count == 0:
            out[i] = 0.0
            continue
        face_indices = np.empty(count, dtype=np.int64)
        for j in range(count):
            face_indices[j] = id_list.GetId(j)
        out[i] = smoothed_offset_potential_point(
            q[i], face_indices,
            conn, pointed_vertices,
            params,
            include_faces, include_edges, include_vertices,
        )
    return out

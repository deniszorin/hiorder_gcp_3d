"""Numba-friendly simplified potential kernels."""

from __future__ import annotations

import numpy as np
from numba import njit

from geometry_connectivity_numba import (
    ArrayF, ArrayI,
    NumbaConnectivity,
    singular_value, eps,
    norm3,
    project_point_to_segment, distance_to_edge_line, triangle_distance,
    _build_numba_connectivity,
    get_vertices_and_edges,
)
from potential_parameters_numba import PotentialParameters, h_epsilon_scalar


@njit(cache=True)
def simplified_potential_face(
    q: ArrayF, face_points: ArrayF,
    params: PotentialParameters,
) -> float:
    p0, p1, p2 = face_points
    r_f = triangle_distance(q, p0, p1, p2)
    denom = r_f ** params.p
    if denom <= eps:
        I_f = singular_value
    else:
        I_f = 1.0 / denom

    if params.localized:
        I_f *= h_epsilon_scalar(r_f, params.epsilon)
    return I_f


@njit(cache=True)
def simplified_potential_edge(
    q: ArrayF, edge_points: ArrayF,
    params: PotentialParameters,
) -> float:
    p0, p1 = edge_points
    _proj, r_e, _inside = project_point_to_segment(q, p0, p1)
    denom = r_e ** params.p
    if denom <= eps:
        I_e = singular_value
    else:
        I_e = 1.0 / denom

    if params.localized:
        I_e *= h_epsilon_scalar(r_e, params.epsilon)
    return I_e


@njit(cache=True)
def simplified_potential_vertex(
    q: ArrayF, p_v: ArrayF,
    params: PotentialParameters,
) -> float:
    r_v = norm3(q - p_v)
    denom = r_v ** params.p
    if denom <= eps:
        I_v = singular_value
    else:
        I_v = 1.0 / denom

    if params.localized:
        I_v *= h_epsilon_scalar(r_v, params.epsilon)
    return I_v


@njit(cache=True)
def simplified_smoothed_offset_potential_point(
    q: ArrayF, face_indices: ArrayI,
    conn: NumbaConnectivity,
    params: PotentialParameters,
    include_faces: bool, include_edges: bool, include_vertices: bool,
    edge_valence: ArrayI, vertex_internal: ArrayI,
) -> float:
    if not (include_faces or include_edges or include_vertices):
        return 0.0

    total = 0.0
    edge_list, vertex_list = get_vertices_and_edges(face_indices, conn)

    if include_faces:
        for i in range(face_indices.size):
            fidx = face_indices[i]
            face_points = conn.V[conn.faces[fidx, :]]
            total += simplified_potential_face(q, face_points, params)

    if include_edges:
        for i in range(edge_list.size):
            edge_idx = edge_list[i]
            v0 = conn.edges[edge_idx, 0]
            v1 = conn.edges[edge_idx, 1]
            edge_points = np.empty((2, 3), dtype=np.float64)
            edge_points[0] = conn.V[v0]
            edge_points[1] = conn.V[v1]
            weight = edge_valence[edge_idx] - 1
            if weight != 0:
                total -= weight * simplified_potential_edge(q, edge_points, params)

    if include_vertices:
        for i in range(vertex_list.size):
            v_idx = vertex_list[i]
            if vertex_internal[v_idx]:
                total += simplified_potential_vertex(q, conn.V[v_idx], params)

    return total


def simplified_smoothed_offset_potential_numba(
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
    face_indices = np.arange(conn.faces.shape[0], dtype=np.int64)

    edge_valence = np.zeros(conn.edges.shape[0], dtype=np.int64)
    edge_valence += conn.edge_faces[:, 0] >= 0
    edge_valence += conn.edge_faces[:, 1] >= 0

    vertex_internal = np.ones(conn.V.shape[0], dtype=np.bool_)
    for edge_idx in range(conn.edges.shape[0]):
        if conn.edge_faces[edge_idx, 0] < 0 or conn.edge_faces[edge_idx, 1] < 0:
            v0 = conn.edges[edge_idx, 0]
            v1 = conn.edges[edge_idx, 1]
            vertex_internal[v0] = False
            vertex_internal[v1] = False
    params = PotentialParameters(alpha, p, epsilon, localized, one_sided)

    nq = q.shape[0]
    out = np.zeros(nq, dtype=np.float64)
    for i in range(nq):
        out[i] = simplified_smoothed_offset_potential_point(
            q[i], face_indices, conn, params,
            include_faces=include_faces,
            include_edges=include_edges,
            include_vertices=include_vertices,
            edge_valence=edge_valence,
            vertex_internal=vertex_internal,
        )
    return out


def simplified_smoothed_offset_potential_accelerated(
    q: ArrayF,
    mesh, geom,
    alpha: float = 0.1, p: float = 2.0, epsilon: float = 0.1,
    include_faces: bool = True, include_edges: bool = True, include_vertices: bool = True,
    localized: bool = False, one_sided: bool = False,
) -> ArrayF:
    if not localized:
        raise ValueError("simplified_smoothed_offset_potential_accelerated requires localized=True.")

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
    edge_valence = np.zeros(conn.edges.shape[0], dtype=np.int64)
    edge_valence += conn.edge_faces[:, 0] >= 0
    edge_valence += conn.edge_faces[:, 1] >= 0
    vertex_internal = np.ones(conn.V.shape[0], dtype=np.bool_)
    for edge_idx in range(conn.edges.shape[0]):
        if conn.edge_faces[edge_idx, 0] < 0 or conn.edge_faces[edge_idx, 1] < 0:
            v0 = conn.edges[edge_idx, 0]
            v1 = conn.edges[edge_idx, 1]
            vertex_internal[v0] = False
            vertex_internal[v1] = False
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
        out[i] = simplified_smoothed_offset_potential_point(
            q[i], face_indices, conn, params,
            include_faces=include_faces,
            include_edges=include_edges,
            include_vertices=include_vertices,
            edge_valence=edge_valence,
            vertex_internal=vertex_internal,
        )
    return out

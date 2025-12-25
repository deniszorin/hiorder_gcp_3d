"""Smoothed offset potential for triangle meshes (indexed edges)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geometry import FaceGeometry, MeshConnectivity, MeshData


ArrayF = np.ndarray


@dataclass(frozen=True)
class PotentialComponents:
    """Potential components at q points.

    Each array is shape (nq,).
    """

    face: ArrayF
    edge: ArrayF
    vertex: ArrayF


singular_value = 1e12
eps = 1e-12


def H(z: ArrayF) -> ArrayF:
    """Smoothed Heaviside on [-1, 1]."""

    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    out[z < -1.0] = 0.0
    out[z > 1.0] = 1.0
    mask = (z >= -1.0) & (z <= 1.0)
    z_mid = z[mask]
    out[mask] = ((2.0 - z_mid) * (z_mid + 1.0) ** 2) / 4.0
    return out


def H_shifted(z: ArrayF) -> ArrayF:
    """Smoothed Heaviside on [-1, 0]."""

    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    out[z < -1.0] = 0.0
    out[z > 0.0] = 1.0
    mask = (z >= -1.0) & (z <= 0.0)
    z_mid = z[mask]
    out[mask] = ((2.0 - 2.0 * z_mid - 1.0) * (2 * z_mid + 1.0 + 1.0) ** 2) / 4.0
    return out


def H_alpha(t: ArrayF, alpha: float) -> ArrayF:
    """Scaled Heaviside H(t / alpha)."""

    return H(np.asarray(t, dtype=float) / alpha)


def H_alpha_shifted(t: ArrayF, alpha: float) -> ArrayF:
    """Scaled shifted Heaviside H_shifted(t / alpha)."""

    return H_shifted(np.asarray(t, dtype=float) / alpha)


def get_local_index(eidx: int, fidx: int, mesh: MeshData, connectivity: MeshConnectivity) -> int:
    """Return the local edge index within face fidx for edge eidx."""

    v0, v1 = connectivity.vertices_per_edge[eidx]
    f0, f1, f2 = mesh.F[fidx].tolist()
    face_edges = [(f0, f1), (f1, f2), (f2, f0)]
    for local_idx, (a, b) in enumerate(face_edges):
        if (a == v0 and b == v1) or (a == v1 and b == v0):
            return local_idx
    raise ValueError(f"Edge {eidx} not found in face {fidx}.")


def smoothed_offset_potential(
    q: ArrayF,
    mesh: MeshData,
    connectivity: MeshConnectivity,
    geom: FaceGeometry,
    alpha: float = 0.1,
    p: float = 2.0,
    epsilon: float = 0.1,
    include_faces: bool = True,
    include_edges: bool = True,
    include_vertices: bool = True,
) -> PotentialComponents:
    """Compute smoothed offset potential components at q points."""

    # maker it work for a single input point
    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        q = q[None, :]
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError("q must have shape (3,) or (nq, 3).")

    V = mesh.V
    F = mesh.F
    nq = q.shape[0]
    nf = F.shape[0]

    # will store potential components here
    face_sum = np.zeros(nq, dtype=float)
    edge_sum = np.zeros(nq, dtype=float)
    vertex_sum = np.zeros(nq, dtype=float)

    r_e_face_edge = np.zeros((nf, 3, nq), dtype=float)
    phi_face_edge = np.zeros((nf, 3, nq), dtype=float)

    if not (include_faces or include_edges or include_vertices):
        return PotentialComponents(face_sum, edge_sum, vertex_sum)

    #  compute face directional terms and face potenials, save dir. terms for edges and vertices
    for f in range(nf):
        v0, v1, v2 = F[f].tolist()
        edge_origins = [V[v0], V[v1], V[v2]]

        n = geom.normals[f]
        # unit vectors along edges, ccw around a face
        edge_dirs = geom.edge_dirs[f]
        # unit vectors pointing towards face interior
        edge_inward = geom.edge_inward[f]

        B = np.ones(nq, dtype=float)
        # for each edge of a face
        for i in range(3):
            # distance from edge first endpoint to projection of q
            t = np.dot(q - edge_origins[i], edge_dirs[i])
            # projection of q position
            # P_e[j] = edge_origin[i] + t[j]*edge_dirs[i]
            P_e = edge_origins[i] + np.outer(t, edge_dirs[i])
            vec = q - P_e
            # distance from q to edge i
            edge_dist = np.linalg.norm(vec, axis=1)
            unit = vec / np.maximum(edge_dist[:, None], eps)
            # Phi^{e,f} := (q-P_e)_+ dot (n x d_e[i])
            phi = np.dot(unit, edge_inward[i])
            B *= H_alpha(phi, alpha)
            # save for edge and vertex terms
            r_e_face_edge[f, i] = edge_dist
            phi_face_edge[f, i] = phi

        if include_faces:
            # distance to the face plane
            r_f = np.abs(np.dot(q - edge_origins[0], n))
            denom = np.power(r_f, p)
            I_f = np.empty_like(B)
            np.divide(B, denom, out=I_f, where=denom > eps)
            I_f = np.where(denom <= eps, singular_value, I_f)
            face_sum += I_f

    # edges are indexed, no dict ordering concerns
    edges = list(range(len(connectivity.vertices_per_edge)))
    phi_edge_endpoint = np.zeros((len(edges), 2, nq), dtype=float)

    # compute edge directional terms, and edge potentials save for vertices
    for edge_idx in edges:
        # over all edg indices
        # p0 = V[edges[eidx][0]]
        # p1 = V[edges[eidx][1]]
        a, b = connectivity.vertices_per_edge[edge_idx]
        p0 = V[a]
        p1 = V[b]
        d = p1 - p0
        d_norm = np.linalg.norm(d)
        d_unit = d / d_norm

        # directions to endpoints
        vec0 = q - p0
        vec1 = q - p1
        unit0 = vec0 / np.maximum(np.linalg.norm(vec0, axis=1)[:, None], eps)
        unit1 = vec1 / np.maximum(np.linalg.norm(vec1, axis=1)[:, None], eps)
        # Phi^{i,e}, i=0,1 factors
        phi0 = np.dot(unit0, d_unit)
        phi1 = np.dot(unit1, -d_unit)
        # save for vertex terms
        phi_edge_endpoint[edge_idx, 0] = phi0
        phi_edge_endpoint[edge_idx, 1] = phi1

        # face_list = m_edges_to_faces[eidx]
        # local_list = [get_edge_index(face_list[0]),get_edge_index(face_list[1]) ]
        face_list = connectivity.faces_per_edge[edge_idx]

        if len(face_list) == 0:
            h_face_0 = np.zeros(nq, dtype=float)
            h_face_1 = np.zeros(nq, dtype=float)
        else:
            f0 = face_list[0]
            e0 = get_local_index(edge_idx, f0, mesh, connectivity)
            h_face_0 = H_alpha(phi_face_edge[f0, e0], alpha)
            if len(face_list) == 1:
                h_face_1 = np.zeros(nq, dtype=float)
            else:
                f1 = face_list[1]
                e1 = get_local_index(edge_idx, f1, mesh, connectivity)
                h_face_1 = H_alpha(phi_face_edge[f1, e1], alpha)

        B_edge = (1.0 - h_face_0 - h_face_1) * H_alpha(phi0, alpha) * H_alpha(phi1, alpha)

        if include_edges:
            # distance to edge r_e already computed for face terms
            if not face_list:
                raise ValueError(f"Missing face-edge mapping for edge {edge_idx}.")
            f_idx = face_list[0]
            local_edge = get_local_index(edge_idx, f_idx, mesh, connectivity)
            r_e = r_e_face_edge[f_idx, local_edge]
            denom = np.power(r_e, p)
            I_e = np.where(r_e <= eps, singular_value, B_edge / denom)
            edge_sum += I_e

    if include_vertices:
        for v in range(V.shape[0]):
            q_to_v = q - V[v]
            r_v = np.linalg.norm(q_to_v, axis=1)

            face_term = np.zeros(nq, dtype=float)
            for f in connectivity.faces_per_vertex[v]:
                # which local edges are incident at the vertex in face f?
                v0, v1, v2 = F[f].tolist()
                if v == v0:
                    e0, e1 = 0, 2
                elif v == v1:
                    e0, e1 = 0, 1
                elif v == v2:
                    e0, e1 = 1, 2
                else:
                    raise ValueError("Vertex not found in face.")

                # face directional factor affecting the vertex (incident edges only)
                h0 = H_alpha(phi_face_edge[f, e0], alpha)
                h1 = H_alpha(phi_face_edge[f, e1], alpha)
                face_term += h0 * h1

            edge_term = np.zeros(nq, dtype=float)
            # for eidx in m_vertices_to_edges
            for edge_idx in connectivity.edges_per_vertex[v]:
                # nop
                a, b = connectivity.vertices_per_edge[edge_idx]

                # retrieve relevant Phi^{v,e} term computed for the edge
                if v == a:
                    h_v = H_alpha(phi_edge_endpoint[edge_idx, 0], alpha)
                elif v == b:
                    h_v = H_alpha(phi_edge_endpoint[edge_idx, 1], alpha)
                else:
                    raise ValueError("Vertex not found in edge.")
                # retrive face terms that were used for the edge

                # face_list = m_edges_to_faces[eidx]
                # local_list = [get_edge_index(face_list[0]),get_edge_index(face_list[1]) ]
                face_list = connectivity.faces_per_edge[edge_idx]
                if len(face_list) == 0:
                    h_face_0 = np.zeros(nq, dtype=float)
                    h_face_1 = np.zeros(nq, dtype=float)
                else:
                    f0 = face_list[0]
                    e0 = get_local_index(edge_idx, f0, mesh, connectivity)
                    h_face_0 = H_alpha(phi_face_edge[f0, e0], alpha)
                    if len(face_list) == 1:
                        h_face_1 = np.zeros(nq, dtype=float)
                    else:
                        f1 = face_list[1]
                        e1 = get_local_index(edge_idx, f1, mesh, connectivity)
                        h_face_1 = H_alpha(phi_face_edge[f1, e1], alpha)
                # complete part of the edge directional factor to be used for the vertex
                # it is different from the complete factor as it only uses h_v = H^alpha(Phi^{v,e})
                # for this vertex, not both
                edge_term += (1.0 - h_face_0 - h_face_1) * h_v

            B_vertex = 1.0 - face_term - edge_term

            denom = np.power(r_v, p)
            I_v = np.where(denom <= eps, singular_value, B_vertex / denom)
            vertex_sum += I_v

    return PotentialComponents(face=face_sum, edge=edge_sum, vertex=vertex_sum)

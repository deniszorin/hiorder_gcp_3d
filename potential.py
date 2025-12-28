"""Smoothed offset potential for triangle meshes (indexed edges)."""

from __future__ import annotations

import numpy as np

from geometry import MeshGeometry, MeshData


ArrayF = np.ndarray

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


def h_local(z: ArrayF) -> ArrayF:
    """Localization polynomial h(z) = (2z + 1)(z - 1)^2."""

    z = np.asarray(z, dtype=float)
    return (2.0 * z + 1.0) * (z - 1.0) ** 2


def h_epsilon(z: ArrayF, epsilon: float) -> ArrayF:
    """Scaled localization polynomial h(z / epsilon)."""
    return h_local(np.asarray(z, dtype=float) / epsilon)


def precompute_potential_terms(
    q: ArrayF,
    mesh: MeshData,
    geom: MeshGeometry,
) -> tuple[ArrayF, ArrayF, ArrayF, ArrayF, ArrayF, ArrayF, ArrayF]:
    """Precompute geometric terms for potential evaluation."""

    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        q = q[None, :]
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError("q must have shape (3,) or (nq, 3).")

    V = mesh.V
    faces = mesh.faces
    nq = q.shape[0]
    nf = faces.shape[0]

    # Precomputed per-face/per-edge quantities used by face/edge/vertex terms.
    # r_f: signed distance to each face plane.
    # r_f_abs: absolute distance to each face plane.
    # r_e_face_edge: distance from q to each face edge line.
    # P_e_face_edge: projection of q to each face edge line.
    # phi_face_edge: Phi^{e,f} := (q - P_e)_+ dot (n x d_e).
    r_f = np.zeros((nf, nq), dtype=float)
    r_f_abs = np.zeros((nf, nq), dtype=float)
    r_e_face_edge = np.zeros((nf, 3, nq), dtype=float)
    P_e_face_edge = np.zeros((nf, 3, nq, 3), dtype=float)
    phi_face_edge = np.zeros((nf, 3, nq), dtype=float)

    for f in range(nf):
        v0, v1, v2 = faces[f].tolist()
        edge_origins = [V[v0], V[v1], V[v2]]

        n = geom.normals[f]
        r_f[f] = np.dot(q - edge_origins[0], n)
        r_f_abs[f] = np.abs(r_f[f])

        edge_dirs = geom.edge_dirs[f]
        edge_inward = geom.edge_inward[f]

        for i in range(3):
            # distance from edge first endpoint to projection of q
            t = np.dot(q - edge_origins[i], edge_dirs[i])
            # projection of q position
            # P_e[j] = edge_origin[i] + t[j] * edge_dirs[i]
            P_e = edge_origins[i] + np.outer(t, edge_dirs[i])
            P_e_face_edge[f, i] = P_e
            vec = q - P_e
            # distance from q to edge i
            edge_dist = np.linalg.norm(vec, axis=1)
            r_e_face_edge[f, i] = edge_dist
            unit = vec / np.maximum(edge_dist[:, None], eps)
            # Phi^{e,f} := (q - P_e)_+ dot (n x d_e[i])
            phi_face_edge[f, i] = np.dot(unit, edge_inward[i])

    # Phi^{0,e} and Phi^{1,-e} for each edge, used by edge/vertex terms.
    edges = mesh.edges
    phi_edge_endpoint = np.zeros((len(edges), 2, nq), dtype=float)
    for edge_idx in range(len(edges)):
        a, b = edges[edge_idx]
        p0 = V[a]
        p1 = V[b]
        d = p1 - p0
        d_norm = np.linalg.norm(d)
        d_unit = d / d_norm

        vec0 = q - p0
        vec1 = q - p1
        unit0 = vec0 / np.maximum(np.linalg.norm(vec0, axis=1)[:, None], eps)
        unit1 = vec1 / np.maximum(np.linalg.norm(vec1, axis=1)[:, None], eps)
        # Phi^{i,e}, i=0,1 factors
        phi_edge_endpoint[edge_idx, 0] = np.dot(unit0, d_unit)
        phi_edge_endpoint[edge_idx, 1] = np.dot(unit1, -d_unit)

    return q, r_f, r_f_abs, r_e_face_edge, P_e_face_edge, phi_face_edge, phi_edge_endpoint


def get_local_index(eidx: int, fidx: int, mesh: MeshData) -> int:
    """Return the local edge index within face fidx for edge eidx."""

    v0, v1 = mesh.edges[eidx]
    f0, f1, f2 = mesh.faces[fidx].tolist()
    face_edges = [(f0, f1), (f1, f2), (f2, f0)]
    for local_idx, (a, b) in enumerate(face_edges):
        if (a == v0 and b == v1) or (a == v1 and b == v0):
            return local_idx
    raise ValueError(f"Edge {eidx} not found in face {fidx}.")


def outside_face(face_idx: int, r_f: ArrayF) -> ArrayF:
    """Check Out^f for a face using signed distances r_f."""

    return r_f[face_idx] > 0


def outside_edge(
    edge_idx: int, q: ArrayF, 
    mesh: MeshData, geom: MeshGeometry,
    r_f: ArrayF, r_e_face_edge: ArrayF, P_e_face_edge: ArrayF, phi_face_edge: ArrayF,
) -> ArrayF:
    """Check Out^e for a single edge."""

    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        q = q[None, :]
    nq = q.shape[0]

    face_list = mesh.edges_to_faces[edge_idx]
    if len(face_list) <= 1:
        return np.ones(nq, dtype=bool)

    f0, f1 = face_list
    e0 = get_local_index(edge_idx, f0, mesh)
    e1 = get_local_index(edge_idx, f1, mesh)

    phi0 = phi_face_edge[f0, e0]
    phi1 = phi_face_edge[f1, e1]
    r_e = r_e_face_edge[f0, e0]

    # initialize r_min with the distance to the edge
    r_min = r_e.copy()
    src = np.zeros(nq, dtype=np.int8)  # 0=edge, 1=f0, 2=f1

    # check where the distance to f0 is less than edge 
    # if the projection is within the halfplane (Phi^{e,f0} > 0)
    r0 = np.abs(r_f[f0])
    mask0 = phi0 > 0
    better0 = mask0 & (r0 < r_min)
    r_min = np.where(better0, r0, r_min)
    src = np.where(better0, 1, src)

    # same for f1
    r1 = np.abs(r_f[f1])
    mask1 = phi1 > 0
    better1 = mask1 & (r1 < r_min)
    r_min = np.where(better1, r1, r_min)
    src = np.where(better1, 2, src)

    out = np.zeros(nq, dtype=bool)
    # if f0 is closest, check if  signed distance to it  r_{f_0} is positive 
    mask_face0 = src == 1
    if np.any(mask_face0):
        idx = np.nonzero(mask_face0)[0]
        out[idx] = r_f[f0, idx] > 0
    # if f1 is closest, check if  signed distance to it  r_{f_1} is positive 
    mask_face1 = src == 2
    if np.any(mask_face1):
        idx = np.nonzero(mask_face1)[0]
        out[idx] = r_f[f1, idx] > 0
    # ifthe edge itself is closest,  check dot product with the average normal 
    # it always points outside, and the sector where edge is closest is within pi/2
    # of the average normal. 
    mask_edge = src == 0
    if np.any(mask_edge):
        n_e = geom.edge_normals[edge_idx]
        P_e = P_e_face_edge[f0, e0]
        diff = q - P_e
#        out[mask_edge] = np.einsum("ij,j->i", diff[mask_edge], n_e) > 0
        out[mask_edge] = np.dot(diff[mask_edge], n_e) > 0
    return out


def _vertex_face_edge_candidates(
    v_idx: int,
    mesh: MeshData,
    r_f: ArrayF, r_e_face_edge: ArrayF, phi_face_edge: ArrayF, phi_edge_endpoint: ArrayF,
) -> tuple[ArrayF, ArrayF, ArrayF, ArrayF]:
    """Return closest face/edge distances for a vertex."""

    faces = mesh.faces
    nq = r_f.shape[1]

    # initialize to inf
    r_face_min = np.full(nq, np.inf, dtype=float)
    # ids of closest faces 
    face_min = np.full(nq, -1, dtype=int)

    for f in mesh.vertices_to_faces[v_idx]:
        # figure out which edges are incident at the vertex
        v0, v1, v2 = faces[f].tolist()
        if v_idx == v0:
            e0, e1 = 0, 2
        elif v_idx == v1:
            e0, e1 = 0, 1
        elif v_idx == v2:
            e0, e1 = 1, 2
        else:
            raise ValueError("Vertex not found in face.")

        # is the projection inside the face, determined by Phi^{e_i,f} signs, i= 0,1
        mask = (phi_face_edge[f, e0] > 0) & (phi_face_edge[f, e1] > 0)
        r_abs = np.abs(r_f[f])
        # if it is, then compare to the current min distance, and replace if less
        better = mask & (r_abs < r_face_min)
        r_face_min = np.where(better, r_abs, r_face_min)
        face_min = np.where(better, f, face_min)

    # Closest edge
    r_edge_min = np.full(nq, np.inf, dtype=float)
    edge_min = np.full(nq, -1, dtype=int)
    for edge_idx in mesh.vertices_to_edges[v_idx]:
        # figure out which vertex, pick the right Phi^{v,e}
        a, b = mesh.edges[edge_idx]
        if v_idx == a:
            phi_v = phi_edge_endpoint[edge_idx, 0]
        elif v_idx == b:
            phi_v = phi_edge_endpoint[edge_idx, 1]
        else:
            raise ValueError("Vertex not found in edge.")

        # is the projection of q to the ray starting at vertex along the edge inside the ray
        mask = phi_v > 0
        face_list = mesh.edges_to_faces[edge_idx]
        f0 = face_list[0]
        e0 = get_local_index(edge_idx, f0, mesh)
        r_e = r_e_face_edge[f0, e0]
        # replace the distance if projection is inside and the distance is less
        better = mask & (r_e < r_edge_min)
        r_edge_min = np.where(better, r_e, r_edge_min)
        edge_min = np.where(better, edge_idx, edge_min)

    return r_face_min, face_min, r_edge_min, edge_min


def _outside_vertex_face_edge(
    v_idx: int,
    q: ArrayF,
    mesh: MeshData,
    geom: MeshGeometry,
    r_f: ArrayF,
    r_e_face_edge: ArrayF,
    P_e_face_edge: ArrayF,
    phi_face_edge: ArrayF,
    phi_edge_endpoint: ArrayF,
) -> tuple[ArrayF, ArrayF, ArrayF]:
    """Check Out^{v,fe} for a vertex (face/edge only)."""

    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        q = q[None, :]
    nq = q.shape[0]

    # get closest face sector and edge ray if any
    r_face_min, face_min, r_edge_min, edge_min = _vertex_face_edge_candidates(
        v_idx, mesh, r_f, r_e_face_edge, phi_face_edge, phi_edge_endpoint
    )

    use_face = r_face_min <= r_edge_min
    out = np.zeros(nq, dtype=bool)

    #  mask_face = there is closest face and it is closer than edge 
    #  use_face may be true if r_face_min = r_edge_min
    mask_face = use_face & (face_min >= 0)

    if np.any(mask_face):
        # indices of points for which face is closest, and indices of these faces
        idx = np.nonzero(mask_face)[0]
        faces_idx = face_min[idx]
        # in this casde we check if signed distance is positive
        out[idx] = r_f[faces_idx, idx] > 0

    #  points closest to edge ray
    mask_edge = (~use_face) & (edge_min >= 0)
    if np.any(mask_edge):
        for edge_idx in np.unique(edge_min[mask_edge]):
            if edge_idx < 0:
                continue
            edge_mask = mask_edge & (edge_min == edge_idx)
            face_list = mesh.edges_to_faces[edge_idx]
            if not face_list:
                out[edge_mask] = True
                continue
            f0 = face_list[0]
            e0 = get_local_index(edge_idx, f0, mesh)
            n_e = geom.edge_normals[edge_idx]
            P_e = P_e_face_edge[f0, e0]
            diff = q[edge_mask] - P_e[edge_mask]
            out[edge_mask] = np.dot(diff, n_e) > 0

    return out, r_face_min, r_edge_min


def outside_vertex(
    v_idx: int,q: ArrayF,
    mesh: MeshData, geom: MeshGeometry,
    r_f: ArrayF, r_e_face_edge: ArrayF, 
    P_e_face_edge: ArrayF, phi_face_edge: ArrayF, phi_edge_endpoint: ArrayF,
) -> ArrayF:
    """Out^v(q) for a vertex."""

    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        q = q[None, :]

    out_fe, r_face_min, r_edge_min = _outside_vertex_face_edge(
        v_idx, q,
        mesh, geom,
        r_f, r_e_face_edge, P_e_face_edge,phi_face_edge, phi_edge_endpoint,
        )

    r_v = np.linalg.norm(q - mesh.V[v_idx], axis=1)
    r_min_fe = np.minimum(r_face_min, r_edge_min)
    use_vertex = r_v < r_min_fe

    out = out_fe.copy()
    if np.any(use_vertex):
        q_reflect = 2.0 * mesh.V[v_idx] - q[use_vertex]
        (   q_reflect,
            r_f_reflect,_, r_e_reflect, P_e_reflect, phi_face_reflect, phi_edge_reflect,
        ) = precompute_potential_terms(q_reflect, mesh, geom)
        out_fe_reflect, _, _ = _outside_vertex_face_edge(
            v_idx, q_reflect,
            mesh, geom,
            r_f_reflect, r_e_reflect, P_e_reflect, phi_face_reflect, phi_edge_reflect,
        )
        out[use_vertex] = ~out_fe_reflect

    return out


def smoothed_offset_potential(
    q: ArrayF,
    mesh: MeshData,
    geom: MeshGeometry,
    alpha: float = 0.1,
    p: float = 2.0,
    epsilon: float = 0.1,
    include_faces: bool = True,
    include_edges: bool = True,
    include_vertices: bool = True,
    localized: bool = False,
    one_sided: bool = False
) -> ArrayF:
    """Compute smoothed offset potential at q points."""

    # maker it work for a single input point
    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        q = q[None, :]
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError("q must have shape (3,) or (nq, 3).")

    q, r_f, r_f_abs, r_e_face_edge, _, phi_face_edge, phi_edge_endpoint = (
        precompute_potential_terms(q, mesh, geom)
    )
    V = mesh.V
    faces = mesh.faces
    nq = q.shape[0]
    nf = faces.shape[0]

    # will store potential components here
    face_sum = np.zeros(nq, dtype=float)
    edge_sum = np.zeros(nq, dtype=float)
    vertex_sum = np.zeros(nq, dtype=float)
    
    if not (include_faces or include_edges or include_vertices):
        return face_sum

    #  compute face directional terms and face potenials, save dir. terms for edges and vertices
    for f in range(nf):
        r_f_abs_f = r_f_abs[f]
        # unit vectors along edges, ccw around a face
        B = np.ones(nq, dtype=float)
        for i in range(3):
            B *= H_alpha(phi_face_edge[f, i], alpha)

        if include_faces:
            denom = np.power(r_f_abs_f, p)
            I_f = np.empty_like(B)
            np.divide(B, denom, out=I_f, where=denom > eps)
            I_f = np.where(denom <= eps, singular_value, I_f)
            if localized:
                I_f *= h_epsilon(r_f_abs_f, epsilon)
            face_sum += I_f

    # edges are indexed, no dict ordering concerns
    edges = list(range(len(mesh.edges)))
    # compute edge directional terms, and edge potentials save for vertices
    for edge_idx in edges:
        phi0 = phi_edge_endpoint[edge_idx, 0]
        phi1 = phi_edge_endpoint[edge_idx, 1]

        face_list = mesh.edges_to_faces[edge_idx]

        f0 = face_list[0]
        e0 = get_local_index(edge_idx, f0, mesh)
        h_face_0 = H_alpha(phi_face_edge[f0, e0], alpha)
        if len(face_list) == 1:
            h_face_1 = np.zeros(nq, dtype=float)
        else:
            f1 = face_list[1]
            e1 = get_local_index(edge_idx, f1, mesh)
            h_face_1 = H_alpha(phi_face_edge[f1, e1], alpha)

        B_edge = (1.0 - h_face_0 - h_face_1) * H_alpha(phi0, alpha) * H_alpha(phi1, alpha)

        if include_edges:
            # distance to edge r_e already computed for face terms
            f_idx = face_list[0]
            local_edge = get_local_index(edge_idx, f_idx, mesh)
            r_e = r_e_face_edge[f_idx, local_edge]
            denom = np.power(r_e, p)
            I_e = np.where(r_e <= eps, singular_value, B_edge / denom)
            if localized:
                I_e *= h_epsilon(r_e, epsilon)
            edge_sum += I_e

    if include_vertices:
        for v in range(V.shape[0]):
            q_to_v = q - V[v]
            r_v = np.linalg.norm(q_to_v, axis=1)

            face_term = np.zeros(nq, dtype=float)
            for f in mesh.vertices_to_faces[v]:
                # which local edges are incident at the vertex in face f?
                v0, v1, v2 = faces[f].tolist()
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
            for edge_idx in mesh.vertices_to_edges[v]:
                # nop
                a, b = mesh.edges[edge_idx]

                # retrieve relevant Phi^{v,e} term computed for the edge
                if v == a:
                    h_v = H_alpha(phi_edge_endpoint[edge_idx, 0], alpha)
                elif v == b:
                    h_v = H_alpha(phi_edge_endpoint[edge_idx, 1], alpha)
                else:
                    raise ValueError("Vertex not found in edge.")
                # retrive face terms that were used for the edge

                face_list = mesh.edges_to_faces[edge_idx]
                f0 = face_list[0]
                e0 = get_local_index(edge_idx, f0, mesh)
                h_face_0 = H_alpha(phi_face_edge[f0, e0], alpha)
                if len(face_list) == 1:
                    h_face_1 = np.zeros(nq, dtype=float)
                else:
                    f1 = face_list[1]
                    e1 = get_local_index(edge_idx, f1, mesh)
                    h_face_1 = H_alpha(phi_face_edge[f1, e1], alpha)
                # complete part of the edge directional factor to be used for the vertex
                # it is different from the complete factor as it only uses h_v = H^alpha(Phi^{v,e})
                # for this vertex, not both
                edge_term += (1.0 - h_face_0 - h_face_1) * h_v

            B_vertex = 1.0 - face_term - edge_term

            denom = np.power(r_v, p)
            I_v = np.where(denom <= eps, singular_value, B_vertex / denom)
            if localized:
                I_v *= h_epsilon(r_v, epsilon)
            vertex_sum += I_v

    return face_sum + edge_sum + vertex_sum

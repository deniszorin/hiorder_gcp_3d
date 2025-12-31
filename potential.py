"""Smoothed offset potential for triangle meshes (indexed edges)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geometry import MeshGeometry, MeshData


ArrayF = np.ndarray

singular_value = 1e12
eps = 1e-12


@dataclass(frozen=True)
class PotentialTerms:
    """Precomputed potential terms for a mesh and query points."""

    r_f: ArrayF
    r_f_abs: ArrayF
    P_f: ArrayF
    r_e: ArrayF
    P_e: ArrayF
    phi_ef: ArrayF
    phi_ve: ArrayF


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
) -> tuple[ArrayF, PotentialTerms]:
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
    # P_f: projection of q to each face plane.
    # r_e: distance from q to each edge line.
    # P_e: projection of q to each edge line.
    # phi_ef: Phi^{e,f} := (q - P_e)_+ dot (n x d_e).
    r_f = np.zeros((nf, nq), dtype=float)
    r_f_abs = np.zeros((nf, nq), dtype=float)
    phi_ef = np.zeros((nf, 3, nq), dtype=float)
    P_f = np.zeros((nf, nq, 3), dtype=float)

    # Phi^{0,e} and Phi^{1,-e} for each edge, used by edge/vertex terms.
    edges = mesh.edges
    r_e = np.zeros((len(edges), nq), dtype=float)
    P_e = np.zeros((len(edges), nq, 3), dtype=float)
    phi_ve = np.zeros((len(edges), 2, nq), dtype=float)
    for edge_idx, (a, b) in enumerate(edges):
        p0 = V[a]
        p1 = V[b]
        d = p1 - p0
        d_norm = np.linalg.norm(d)
        d_unit = d / d_norm
        # the choice of the direction on the edge does not affect
        # projected position P_e can use any
        t = np.dot(q - p0, d_unit)
        P_e[edge_idx] = p0 + np.outer(t, d_unit)
        vec = q - P_e[edge_idx]
        r_e[edge_idx] = np.linalg.norm(vec, axis=1)

        vec0 = q - p0
        vec1 = q - p1
        unit0 = vec0 / np.maximum(np.linalg.norm(vec0, axis=1)[:, None], eps)
        unit1 = vec1 / np.maximum(np.linalg.norm(vec1, axis=1)[:, None], eps)
        # Phi^{i,e}, i=0,1 factors per edge
        phi_ve[edge_idx, 0] = np.dot(unit0, d_unit)
        phi_ve[edge_idx, 1] = np.dot(unit1, -d_unit)

    for f in range(nf):
        v0, v1, v2 = faces[f].tolist()
        edge_origins = [V[v0], V[v1], V[v2]]

        n = geom.normals[f]
        r_f[f] = np.dot(q - edge_origins[0], n)
        r_f_abs[f] = np.abs(r_f[f])
        # projection points on each face plane
        P_f[f] = q - np.outer(r_f[f], n)

        edge_inward = geom.edge_inward[f]
        face_edges = [(v0, v1), (v1, v2), (v2, v0)]

        for i, (a, b) in enumerate(face_edges):
            edge_idx = _edge_index_from_vertices(a, b, mesh)
            edge_dist = r_e[edge_idx]
            unit = (q - P_e[edge_idx]) / np.maximum(edge_dist[:, None], eps)
            # Phi^{e,f} := (q - P_e)_+ dot (n x d_e[i]) per face
            phi_ef[f, i] = np.dot(unit, edge_inward[i])

    return q, PotentialTerms(
        r_f=r_f, r_f_abs=r_f_abs, P_f=P_f,
        r_e=r_e, P_e=P_e,
        phi_ef=phi_ef, phi_ve=phi_ve,
    )


def _edge_index_from_vertices(a: int, b: int, mesh: MeshData) -> int:
    """Return edge index for an undirected vertex pair."""

    for eidx in mesh.vertices_to_edges[a]:
        v0, v1 = mesh.edges[eidx]
        if (v0 == a and v1 == b) or (v0 == b and v1 == a):
            return eidx
    raise ValueError(f"Edge ({a}, {b}) not found.")


def get_local_index(eidx: int, fidx: int, mesh: MeshData) -> int:
    """Return the local edge index within face fidx for edge eidx."""

    v0, v1 = mesh.edges[eidx]
    f0, f1, f2 = mesh.faces[fidx].tolist()
    face_edges = [(f0, f1), (f1, f2), (f2, f0)]
    for local_idx, (a, b) in enumerate(face_edges):
        if (a == v0 and b == v1) or (a == v1 and b == v0):
            return local_idx
    raise ValueError(f"Edge {eidx} not found in face {fidx}.")


def outside_face(face_idx: int, terms: PotentialTerms) -> tuple[ArrayF, ArrayF]:
    """Check Out^f for a face using signed distances r_f."""

    return terms.r_f[face_idx] > 0, terms.P_f[face_idx]


def outside_edge(
    edge_idx: int, q: ArrayF, 
    mesh: MeshData, geom: MeshGeometry, terms: PotentialTerms,
) -> tuple[ArrayF, ArrayF]:
    """Check Out^e for a single edge."""

    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        q = q[None, :]
    nq = q.shape[0]

    face_list = mesh.edges_to_faces[edge_idx]
    if not face_list:
        return np.ones(nq, dtype=bool), terms.P_e[edge_idx]

    f0 = face_list[0]
    f1 = face_list[1] if len(face_list) > 1 else None
    e0 = get_local_index(edge_idx, f0, mesh)
    phi0 = terms.phi_ef[f0, e0]
    if f1 is not None:
        e1 = get_local_index(edge_idx, f1, mesh)
        phi1 = terms.phi_ef[f1, e1]
    else:
        phi1 = None
    r_e = terms.r_e[edge_idx]

    # initialize r_min with the distance to the edge
    r_min = r_e.copy()
    src = np.zeros(nq, dtype=np.int8)  # 0=edge, 1=f0, 2=f1

    # check where the distance to f0 is less than edge 
    # if the projection is within the halfplane (Phi^{e,f0} > 0)
    r0 = np.abs(terms.r_f[f0])
    mask0 = phi0 > 0
    better0 = mask0 & (r0 < r_min)
    r_min = np.where(better0, r0, r_min)
    src = np.where(better0, 1, src)

    # same for f1
    if f1 is not None:
        r1 = np.abs(terms.r_f[f1])
        mask1 = phi1 > 0
        better1 = mask1 & (r1 < r_min)
        r_min = np.where(better1, r1, r_min)
        src = np.where(better1, 2, src)

    out = np.zeros(nq, dtype=bool)
    closest = np.zeros((nq, 3), dtype=float)
    # if f0 is closest, check if  signed distance to it  r_{f_0} is positive 
    mask_face0 = src == 1
    if np.any(mask_face0):
        idx = np.nonzero(mask_face0)[0]
        out[idx] = terms.r_f[f0, idx] > 0
        closest[idx] = terms.P_f[f0, idx]
    # if f1 is closest, check if  signed distance to it  r_{f_1} is positive 
    mask_face1 = src == 2
    if np.any(mask_face1):
        idx = np.nonzero(mask_face1)[0]
        out[idx] = terms.r_f[f1, idx] > 0
        closest[idx] = terms.P_f[f1, idx]
    # ifthe edge itself is closest,  check dot product with the average normal 
    # it always points outside, and the sector where edge is closest is within pi/2
    # of the average normal. 
    mask_edge = src == 0
    if np.any(mask_edge):
        n_e = geom.edge_normals[edge_idx]
        P_e = terms.P_e[edge_idx]
        diff = q - P_e
#        out[mask_edge] = np.einsum("ij,j->i", diff[mask_edge], n_e) > 0
        out[mask_edge] = np.dot(diff[mask_edge], n_e) > 0
        closest[mask_edge] = P_e[mask_edge]
    return out, closest


def _vertex_face_edge_candidates(v_idx: int, 
                                 mesh: MeshData, terms: PotentialTerms,
) -> tuple[ArrayF, ArrayF, ArrayF, ArrayF]:
    """Return closest face/edge distances for a vertex."""

    faces = mesh.faces
    nq = terms.r_f.shape[1]

    # initialize to inf
    r_f_min = np.full(nq, np.inf, dtype=float)
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
        mask = (terms.phi_ef[f, e0] > 0) & (terms.phi_ef[f, e1] > 0)
        r_abs = np.abs(terms.r_f[f])
        # if it is, then compare to the current min distance, and replace if less
        better = mask & (r_abs < r_f_min)
        r_f_min = np.where(better, r_abs, r_f_min)
        face_min = np.where(better, f, face_min)

    # Closest edge
    r_e_min = np.full(nq, np.inf, dtype=float)
    edge_min = np.full(nq, -1, dtype=int)
    for edge_idx in mesh.vertices_to_edges[v_idx]:
        # figure out which vertex, pick the right Phi^{v,e}
        a, b = mesh.edges[edge_idx]
        if v_idx == a:
            phi_v = terms.phi_ve[edge_idx, 0]
        elif v_idx == b:
            phi_v = terms.phi_ve[edge_idx, 1]
        else:
            raise ValueError("Vertex not found in edge.")

        # is the projection of q to the ray starting at vertex along the edge inside the ray
        mask = phi_v > 0
        face_list = mesh.edges_to_faces[edge_idx]
        f0 = face_list[0]
        e0 = get_local_index(edge_idx, f0, mesh)
        r_e = terms.r_e[edge_idx]
        # replace the distance if projection is inside and the distance is less
        better = mask & (r_e < r_e_min)
        r_e_min = np.where(better, r_e, r_e_min)
        edge_min = np.where(better, edge_idx, edge_min)

    return r_f_min, face_min, r_e_min, edge_min


def outside_vertex(
    v_idx: int,q: ArrayF,
    mesh: MeshData, geom: MeshGeometry,terms: PotentialTerms,
) -> tuple[ArrayF, ArrayF]:
    """Out^v(q) for a vertex."""

    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        q = q[None, :]

    # get closest face sector and edge ray if any
    r_f_min, face_min, r_e_min, edge_min = _vertex_face_edge_candidates(
        v_idx, mesh, terms
    )

    r_v = np.linalg.norm(q - mesh.V[v_idx], axis=1)
    r_min_fe = np.minimum(r_f_min, r_e_min)
    use_vertex = r_v < r_min_fe

    use_face = (face_min >= 0) & (r_f_min <= r_e_min) & (~use_vertex)
    use_edge = (edge_min >= 0) & (~use_face) & (~use_vertex)

    out = np.zeros(q.shape[0], dtype=bool)
    closest = np.zeros((q.shape[0], 3), dtype=float)

    if np.any(use_face):
        idx = np.nonzero(use_face)[0]
        faces_idx = face_min[idx]
        out[idx] = terms.r_f[faces_idx, idx] > 0
        closest[idx] = terms.P_f[faces_idx, idx]

    if np.any(use_edge):
        idx = np.nonzero(use_edge)[0]
        edge_ids = edge_min[idx]
        n_e = geom.edge_normals[edge_ids]
        P_e = terms.P_e[edge_ids, idx]
        diff = q[idx] - P_e
        out[idx] = np.einsum("ij,ij->i", diff, n_e) > 0
        closest[idx] = P_e
    
    if np.any(use_vertex):
        idx = np.nonzero(use_vertex)[0]
        p_v = mesh.V[v_idx]
        q_v = q[idx] - p_v
        out_v = np.zeros(idx.shape[0], dtype=bool)
        assigned = np.zeros(idx.shape[0], dtype=bool)

        for edge_idx in mesh.vertices_to_edges[v_idx]:
            a, b = mesh.edges[edge_idx]
            # the other endpoint of the edge
            other = b if a == v_idx else a
            # vector along the edge
            edge_vec = mesh.V[other] - p_v
            dots = q_v @ edge_vec
            # if not assigned outside/inside and dot product (q-p_v ) dot (p_j -p_v)
            # is large enough
            mask = (~assigned) & (np.abs(dots) > eps)
            out_v[mask] = dots[mask] < 0
            # mark as assigned 
            assigned |= mask
            if np.all(assigned):
                break
        # if any points left unassigned after a pass over all edges
        # all their neighnors are in the plane perp to q - p_v
        if np.any(~assigned):
            face_list = mesh.vertices_to_faces[v_idx]
            n0 = geom.normals[face_list[0]] # if face_list else np.zeros(3, dtype=float)
            dots = q_v @ n0
            out_v[~assigned] = dots[~assigned] > 0
        out[idx] = out_v
        closest[idx] = p_v

    return out, closest


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
    one_sided: bool = False,
    use_numba: bool = False,
) -> ArrayF:
    """Compute smoothed offset potential at q points."""

    if use_numba:
        from potential_numba import smoothed_offset_potential_numba

        return smoothed_offset_potential_numba(
            q,
            mesh,
            geom,
            alpha=alpha, p=p, epsilon=epsilon,
            include_faces=include_faces, include_edges=include_edges,
            include_vertices=include_vertices,
            localized=localized, one_sided=one_sided,
        )

    # maker it work for a single input point
    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        q = q[None, :]
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError("q must have shape (3,) or (nq, 3).")

    q, terms = precompute_potential_terms(q, mesh, geom)
    r_f = terms.r_f
    r_f_abs = terms.r_f_abs
    r_e = terms.r_e
    phi_ef = terms.phi_ef
    phi_ve = terms.phi_ve
    V = mesh.V
    faces = mesh.faces
    nq = q.shape[0]
    nf = faces.shape[0]

    # will store potential components here
    face_sum = np.zeros(nq, dtype=float)
    edge_sum = np.zeros(nq, dtype=float)
    vertex_sum = np.zeros(nq, dtype=float)

    outside_faces = None
    outside_edges = None
    outside_vertices = None
    if one_sided:
        outside_faces = np.zeros((nf, nq), dtype=bool)
        for f in range(nf):
            outside_faces[f], _ = outside_face(f, terms)
        outside_edges = np.zeros((len(mesh.edges), nq), dtype=bool)
        for edge_idx in range(len(mesh.edges)):
            outside_edges[edge_idx], _ = outside_edge(edge_idx, q, mesh, geom, terms)
        outside_vertices = np.zeros((V.shape[0], nq), dtype=bool)
        for v in range(V.shape[0]):
            outside_vertices[v], _ = outside_vertex(v, q, mesh, geom, terms)
    
    #     
    if not (include_faces or include_edges or include_vertices):
        return face_sum

    #  compute face directional terms and face potenials, save dir. terms for edges and vertices
    for f in range(nf):
        r_f_abs_f = r_f_abs[f]
        # unit vectors along edges, ccw around a face
        B = np.ones(nq, dtype=float)
        for i in range(3):
            B *= H_alpha(phi_ef[f, i], alpha)

        if include_faces:
            denom = np.power(r_f_abs_f, p)
            I_f = np.empty_like(B)
            np.divide(B, denom, out=I_f, where=denom > eps)
            I_f = np.where(denom <= eps, singular_value, I_f)
            if one_sided:
                I_f = np.where(outside_faces[f], I_f, 0.0)
            if localized:
                I_f *= h_epsilon(r_f_abs_f, epsilon)
            face_sum += I_f

    # edges are indexed, no dict ordering concerns
    edges = list(range(len(mesh.edges)))
    # compute edge directional terms, and edge potentials save for vertices
    for edge_idx in edges:
        phi0 = phi_ve[edge_idx, 0]
        phi1 = phi_ve[edge_idx, 1]

        face_list = mesh.edges_to_faces[edge_idx]

        f0 = face_list[0]
        e0 = get_local_index(edge_idx, f0, mesh)
        h_face_0 = H_alpha(phi_ef[f0, e0], alpha)
        if len(face_list) == 1:
            h_face_1 = np.zeros(nq, dtype=float)
        else:
            f1 = face_list[1]
            e1 = get_local_index(edge_idx, f1, mesh)
            h_face_1 = H_alpha(phi_ef[f1, e1], alpha)
        if one_sided:
            h_face_0 = h_face_0 * outside_faces[f0]
            if len(face_list) > 1:
                h_face_1 = h_face_1 * outside_faces[f1]

        B_edge = (1.0 - h_face_0 - h_face_1) * H_alpha(phi0, alpha) * H_alpha(phi1, alpha)

        if include_edges:
            # distance to edge r_e already computed per edge
            denom = np.power(r_e[edge_idx], p)
            I_e = np.empty_like(B_edge)
            np.divide(B_edge, denom, out=I_e, where=denom > eps)
            I_e = np.where(denom <= eps, singular_value, I_e)
            if one_sided:
                I_e = np.where(outside_edges[edge_idx], I_e, 0.0)
            if localized:
                I_e *= h_epsilon(r_e[edge_idx], epsilon)
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
                h0 = H_alpha(phi_ef[f, e0], alpha)
                h1 = H_alpha(phi_ef[f, e1], alpha)
                if one_sided:
                    h0 = h0 * outside_faces[f]
                    h1 = h1 * outside_faces[f]
                face_term += h0 * h1

            edge_term = np.zeros(nq, dtype=float)
            # for eidx in m_vertices_to_edges
            for edge_idx in mesh.vertices_to_edges[v]:
                # nop
                a, b = mesh.edges[edge_idx]

                # retrieve relevant Phi^{v,e} term computed for the edge
                if v == a:
                    h_v = H_alpha(phi_ve[edge_idx, 0], alpha)
                elif v == b:
                    h_v = H_alpha(phi_ve[edge_idx, 1], alpha)
                else:
                    raise ValueError("Vertex not found in edge.")
                if one_sided:
                    h_v = h_v * outside_edges[edge_idx]
                # retrive face terms that were used for the edge

                face_list = mesh.edges_to_faces[edge_idx]
                f0 = face_list[0]
                e0 = get_local_index(edge_idx, f0, mesh)
                h_face_0 = H_alpha(phi_ef[f0, e0], alpha)
                if len(face_list) == 1:
                    h_face_1 = np.zeros(nq, dtype=float)
                else:
                    f1 = face_list[1]
                    e1 = get_local_index(edge_idx, f1, mesh)
                    h_face_1 = H_alpha(phi_ef[f1, e1], alpha)
                if one_sided:
                    h_face_0 = h_face_0 * outside_faces[f0]
                    if len(face_list) > 1:
                        h_face_1 = h_face_1 * outside_faces[f1]
                # complete part of the edge directional factor to be used for the vertex
                # it is different from the complete factor as it only uses h_v = H^alpha(Phi^{v,e})
                # for this vertex, not both
                edge_term += (1.0 - h_face_0 - h_face_1) * h_v

            B_vertex = 1.0 - face_term - edge_term

            denom = np.power(r_v, p)
            I_v = np.where(denom <= eps, singular_value, B_vertex / denom)
            if one_sided:
                I_v = np.where(outside_vertices[v], I_v, 0.0)
            if localized:
                I_v *= h_epsilon(r_v, epsilon)
            vertex_sum += I_v
    return face_sum + edge_sum + vertex_sum

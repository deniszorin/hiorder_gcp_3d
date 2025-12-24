"""Smoothed offset potential for triangle meshes (skeleton)."""

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

# not used
def H_shifted(z: ArrayF) -> ArrayF:
    """Smoothed Heaviside on [-1, 0]."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    out[z < -1.0] = 0.0
    out[z > 0.0] = 1.0
    mask = (z >= -1.0) & (z <= 0.0)
    z_mid = z[mask]
    out[mask] = ((2.0 - 2.0*z_mid-1.0) * (2*z_mid+1.0 + 1.0) ** 2) / 4.0
    return out


def H_alpha(t: ArrayF, alpha: float) -> ArrayF:
    """Scaled Heaviside H(t / alpha)."""

    return H(np.asarray(t, dtype=float) / alpha)


def smoothed_offset_potential(
    q: ArrayF,
    mesh: MeshData,
    connectivity: MeshConnectivity,
    geom: FaceGeometry,
    alpha: float = 0.1,
    p: float = 2.0,
    edge_potential_mode: str = "heaviside",
    vertex_potential_mode: str = "heaviside",
    include_faces: bool = True,
    include_edges: bool = True,
    include_vertices: bool = True,
    parallel: bool = False,
    max_workers: Optional[int] = 4,
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

    # face directional factors per eval point
    B_f = np.zeros((nf, nq), dtype=float)
    edge_to_face_edges = {}
    r_e_face_edge = np.zeros((nf, 3, nq), dtype=float)
    phi_face_edge = np.zeros((nf, 3, nq), dtype=float)

    def _chunk_ranges(count: int, workers: int) -> list[tuple[int, int]]:
        step = max(1, count // workers)
        ranges = []
        start = 0
        while start < count:
            end = min(count, start + step)
            ranges.append((start, end))
            start = end
        return ranges

    if include_faces or include_edges or include_vertices:
        # always compute face directional terms, we need them for vertex and edge terms too
        if parallel and nf > 1:
            from concurrent.futures import ThreadPoolExecutor

            def _face_chunk(start: int, end: int):
                local_face_sum = np.zeros(nq, dtype=float)
                for f in range(start, end):
                    v0, v1, v2 = F[f].tolist()
                    p0 = V[v0]
                    p1 = V[v1]
                    p2 = V[v2]

                    n = geom.normals[f]
                    edge_dirs = geom.edge_dirs[f]
                    edge_inward = geom.edge_inward[f]

                    origins = [p0, p1, p2]
                    face_edges = [(v0, v1), (v1, v2), (v2, v0)]
                    B = np.ones(nq, dtype=float)
                    for i in range(3):
                        origin = origins[i]
                        d_e = edge_dirs[i]
                        inward = edge_inward[i]
                        a, b = face_edges[i]
                        key = (a, b) if a < b else (b, a)
                        edge_to_face_edges.setdefault(key, []).append((f, i))

                        q_to_origin = q - origin
                        t = np.dot(q_to_origin, d_e)
                        P_e = origin + np.outer(t, d_e)
                        vec = q - P_e
                        vec_norm = np.linalg.norm(vec, axis=1)
                        unit = vec / np.maximum(vec_norm[:, None], eps)
                        phi = np.einsum("ij,j->i", unit, inward)
                        B *= H_alpha(phi, alpha)
                        r_e_face_edge[f, i] = vec_norm
                        phi_face_edge[f, i] = phi

                    B_f[f] = B

                    if include_faces:
                        r_f = np.abs(np.dot(q - p0, n))
                        denom = np.power(r_f, p)
                        I_f = np.where(denom <= eps, singular_value, B / denom)
                        local_face_sum += I_f
                return local_face_sum

            ranges = _chunk_ranges(nf, max_workers or 4)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for local_sum in executor.map(lambda r: _face_chunk(*r), ranges):
                    face_sum += local_sum
        else:
            for f in range(nf):
                v0, v1, v2 = F[f].tolist()
                p0 = V[v0]
                p1 = V[v1]
                p2 = V[v2]

                n = geom.normals[f]
                edge_dirs = geom.edge_dirs[f]
                edge_inward = geom.edge_inward[f]

                origins = [p0, p1, p2]
                face_edges = [(v0, v1), (v1, v2), (v2, v0)]
                B = np.ones(nq, dtype=float)
                for i in range(3):
                    origin = origins[i]
                    d_e = edge_dirs[i]
                    inward = edge_inward[i]
                    a, b = face_edges[i]
                    key = (a, b) if a < b else (b, a)
                    edge_to_face_edges.setdefault(key, []).append((f, i))

                    q_to_origin = q - origin
                    t = np.dot(q_to_origin, d_e)
                    P_e = origin + np.outer(t, d_e)
                    vec = q - P_e
                    vec_norm = np.linalg.norm(vec, axis=1)
                    unit = vec / np.maximum(vec_norm[:, None], eps)
                    phi = np.einsum("ij,j->i", unit, inward)
                    B *= H_alpha(phi, alpha)
                    r_e_face_edge[f, i] = vec_norm
                    phi_face_edge[f, i] = phi

                B_f[f] = B

                if include_faces:
                    r_f = np.abs(np.dot(q - p0, n))
                    denom = np.power(r_f, p)
                    I_f = np.where(denom <= eps, singular_value, B / denom)
                    face_sum += I_f

    edges = sorted(connectivity.faces_per_edge.keys())
    edge_index = {edge: idx for idx, edge in enumerate(edges)}
    B_e = np.zeros((len(edges), nq), dtype=float)
    phi_edge_endpoint = np.zeros((len(edges), 2, nq), dtype=float)

    if parallel and len(edges) > 1:
        from concurrent.futures import ThreadPoolExecutor

        def _edge_chunk(start: int, end: int):
            local_edge_sum = np.zeros(nq, dtype=float)
            for idx in range(start, end):
                edge = edges[idx]
                a, b = edge
                p0 = V[a]
                p1 = V[b]
                d = p1 - p0
                d_norm = np.linalg.norm(d)
                if d_norm <= eps:
                    raise ValueError(f"Degenerate edge between vertices {edge}.")
                d_unit = d / d_norm

                vec0 = q - p0
                vec1 = q - p1
                unit0 = vec0 / np.maximum(np.linalg.norm(vec0, axis=1)[:, None], eps)
                unit1 = vec1 / np.maximum(np.linalg.norm(vec1, axis=1)[:, None], eps)
                phi0 = np.einsum("ij,j->i", unit0, d_unit)
                phi1 = np.einsum("ij,j->i", unit1, -d_unit)
                h0 = H_alpha(phi0, alpha)
                h1 = H_alpha(phi1, alpha)
                phi_edge_endpoint[idx, 0] = phi0
                phi_edge_endpoint[idx, 1] = phi1

                faces = connectivity.faces_per_edge.get(edge, [])
                if edge_potential_mode == "heaviside":
                    face_edges = edge_to_face_edges.get(edge, [])
                    if len(face_edges) == 0:
                        h_face_0 = np.zeros(nq, dtype=float)
                        h_face_1 = np.zeros(nq, dtype=float)
                    elif len(face_edges) == 1:
                        f_idx, local_edge = face_edges[0]
                        h_face_0 = H_alpha(phi_face_edge[f_idx, local_edge], alpha)
                        h_face_1 = np.zeros(nq, dtype=float)
                    else:
                        f0, e0 = face_edges[0]
                        f1, e1 = face_edges[1]
                        h_face_0 = H_alpha(phi_face_edge[f0, e0], alpha)
                        h_face_1 = H_alpha(phi_face_edge[f1, e1], alpha)
                    B_edge = (1.0 - h_face_0 - h_face_1) * h0 * h1
                elif edge_potential_mode == "blend":
                    if len(faces) == 0:
                        B_edge = np.ones(nq, dtype=float)
                    elif len(faces) == 1:
                        B_edge = 1.0 - B_f[faces[0]]
                    else:
                        B_edge = 1.0 - B_f[faces[0]] - B_f[faces[1]]
                    B_edge *= h0 * h1
                else:
                    raise ValueError(f"Unknown edge_potential_mode: {edge_potential_mode}")

                B_e[idx] = B_edge

                if include_edges:
                    face_edges = edge_to_face_edges.get(edge)
                    if not face_edges:
                        raise ValueError(f"Missing face-edge mapping for edge {edge}.")
                    f_idx, local_edge = face_edges[0]
                    r_e = r_e_face_edge[f_idx, local_edge]
                    denom = np.power(r_e, p)
                    I_e = np.where(denom <= eps, singular_value, B_edge / denom)
                    local_edge_sum += I_e
            return local_edge_sum

        ranges = _chunk_ranges(len(edges), max_workers or 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for local_sum in executor.map(lambda r: _edge_chunk(*r), ranges):
                edge_sum += local_sum
    else:
        for idx, edge in enumerate(edges):
            a, b = edge
            p0 = V[a]
            p1 = V[b]
            d = p1 - p0
            d_norm = np.linalg.norm(d)
            if d_norm <= eps:
                raise ValueError(f"Degenerate edge between vertices {edge}.")
            d_unit = d / d_norm

            vec0 = q - p0
            vec1 = q - p1
            unit0 = vec0 / np.maximum(np.linalg.norm(vec0, axis=1)[:, None], eps)
            unit1 = vec1 / np.maximum(np.linalg.norm(vec1, axis=1)[:, None], eps)
            phi0 = np.einsum("ij,j->i", unit0, d_unit)
            phi1 = np.einsum("ij,j->i", unit1, -d_unit)
            h0 = H_alpha(phi0, alpha)
            h1 = H_alpha(phi1, alpha)
            phi_edge_endpoint[idx, 0] = phi0
            phi_edge_endpoint[idx, 1] = phi1

            faces = connectivity.faces_per_edge.get(edge, [])
            if edge_potential_mode == "heaviside":
                face_edges = edge_to_face_edges.get(edge, [])
                if len(face_edges) == 0:
                    h_face_0 = np.zeros(nq, dtype=float)
                    h_face_1 = np.zeros(nq, dtype=float)
                elif len(face_edges) == 1:
                    f_idx, local_edge = face_edges[0]
                    h_face_0 = H_alpha(phi_face_edge[f_idx, local_edge], alpha)
                    h_face_1 = np.zeros(nq, dtype=float)
                else:
                    f0, e0 = face_edges[0]
                    f1, e1 = face_edges[1]
                    h_face_0 = H_alpha(phi_face_edge[f0, e0], alpha)
                    h_face_1 = H_alpha(phi_face_edge[f1, e1], alpha)
                B_edge = (1.0 - h_face_0 - h_face_1) * h0 * h1
            elif edge_potential_mode == "blend":
                if len(faces) == 0:
                    B_edge = np.ones(nq, dtype=float)
                elif len(faces) == 1:
                    B_edge = 1.0 - B_f[faces[0]]
                else:
                    B_edge = 1.0 - B_f[faces[0]] - B_f[faces[1]]
                B_edge *= h0 * h1
            else:
                raise ValueError(f"Unknown edge_potential_mode: {edge_potential_mode}")

            B_e[idx] = B_edge

            if include_edges:
                face_edges = edge_to_face_edges.get(edge)
                if not face_edges:
                    raise ValueError(f"Missing face-edge mapping for edge {edge}.")
                f_idx, local_edge = face_edges[0]
                r_e = r_e_face_edge[f_idx, local_edge]

                denom = np.power(r_e, p)
                I_e = np.where(r_e <= eps, singular_value, B_edge / denom)
                edge_sum += I_e

    if include_vertices:
        nv = V.shape[0]
        if parallel and nv > 1:
            from concurrent.futures import ThreadPoolExecutor

            def _vertex_chunk(start: int, end: int):
                local_vertex_sum = np.zeros(nq, dtype=float)
                for v in range(start, end):
                    q_to_v = q - V[v]
                    r_v = np.linalg.norm(q_to_v, axis=1)

                    if vertex_potential_mode == "heaviside":
                        face_term = np.zeros(nq, dtype=float)
                        for f in connectivity.faces_per_vertex[v]:
                            v0, v1, v2 = F[f].tolist()
                            if v == v0:
                                e0, e1 = 0, 2
                            elif v == v1:
                                e0, e1 = 0, 1
                            elif v == v2:
                                e0, e1 = 1, 2
                            else:
                                raise ValueError("Vertex not found in face.")
                            h0 = H_alpha(phi_face_edge[f, e0], alpha)
                            h1 = H_alpha(phi_face_edge[f, e1], alpha)
                            face_term += h0 * h1

                        edge_term = np.zeros(nq, dtype=float)
                        for edge_key in connectivity.edges_per_vertex[v]:
                            idx = edge_index[edge_key]
                            a, b = edge_key
                            if v == a:
                                h_v = H_alpha(phi_edge_endpoint[idx, 0], alpha)
                            elif v == b:
                                h_v = H_alpha(phi_edge_endpoint[idx, 1], alpha)
                            else:
                                raise ValueError("Vertex not found in edge.")
                            face_edges = edge_to_face_edges.get(edge_key, [])
                            if len(face_edges) == 0:
                                h_face_0 = np.zeros(nq, dtype=float)
                                h_face_1 = np.zeros(nq, dtype=float)
                            elif len(face_edges) == 1:
                                f_idx, local_edge = face_edges[0]
                                h_face_0 = H_alpha(phi_face_edge[f_idx, local_edge], alpha)
                                h_face_1 = np.zeros(nq, dtype=float)
                            else:
                                f0, e0 = face_edges[0]
                                f1, e1 = face_edges[1]
                                h_face_0 = H_alpha(phi_face_edge[f0, e0], alpha)
                                h_face_1 = H_alpha(phi_face_edge[f1, e1], alpha)
                            edge_term += (1.0 - h_face_0 - h_face_1) * h_v

                        B_vertex = 1.0 - face_term - edge_term
                    elif vertex_potential_mode == "blend":
                        B_vertex = np.ones(nq, dtype=float)
                        for f in connectivity.faces_per_vertex[v]:
                            B_vertex -= B_f[f]
                        for edge_key in connectivity.edges_per_vertex[v]:
                            idx = edge_index[edge_key]
                            B_vertex -= B_e[idx]
                    else:
                        raise ValueError(f"Unknown vertex_potential_mode: {vertex_potential_mode}")

                    denom = np.power(r_v, p)
                    I_v = np.where(denom <= eps, singular_value, B_vertex / denom)
                    local_vertex_sum += I_v
                return local_vertex_sum

            ranges = _chunk_ranges(nv, max_workers or 4)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for local_sum in executor.map(lambda r: _vertex_chunk(*r), ranges):
                    vertex_sum += local_sum
        else:
            for v in range(nv):
                q_to_v = q - V[v]
                r_v = np.linalg.norm(q_to_v, axis=1)

                if vertex_potential_mode == "heaviside":
                    face_term = np.zeros(nq, dtype=float)
                    for f in connectivity.faces_per_vertex[v]:
                        v0, v1, v2 = F[f].tolist()
                        if v == v0:
                            e0, e1 = 0, 2
                        elif v == v1:
                            e0, e1 = 0, 1
                        elif v == v2:
                            e0, e1 = 1, 2
                        else:
                            raise ValueError("Vertex not found in face.")
                        h0 = H_alpha(phi_face_edge[f, e0], alpha)
                        h1 = H_alpha(phi_face_edge[f, e1], alpha)
                        face_term += h0 * h1

                    edge_term = np.zeros(nq, dtype=float)
                    for edge_key in connectivity.edges_per_vertex[v]:
                        idx = edge_index[edge_key]
                        a, b = edge_key
                        if v == a:
                            h_v = H_alpha(phi_edge_endpoint[idx, 0], alpha)
                        elif v == b:
                            h_v = H_alpha(phi_edge_endpoint[idx, 1], alpha)
                        else:
                            raise ValueError("Vertex not found in edge.")
                        face_edges = edge_to_face_edges.get(edge_key, [])
                        if len(face_edges) == 0:
                            h_face_0 = np.zeros(nq, dtype=float)
                            h_face_1 = np.zeros(nq, dtype=float)
                        elif len(face_edges) == 1:
                            f_idx, local_edge = face_edges[0]
                            h_face_0 = H_alpha(phi_face_edge[f_idx, local_edge], alpha)
                            h_face_1 = np.zeros(nq, dtype=float)
                        else:
                            f0, e0 = face_edges[0]
                            f1, e1 = face_edges[1]
                            h_face_0 = H_alpha(phi_face_edge[f0, e0], alpha)
                            h_face_1 = H_alpha(phi_face_edge[f1, e1], alpha)
                        edge_term += (1.0 - h_face_0 - h_face_1) * h_v

                    B_vertex = 1.0 - face_term - edge_term
                elif vertex_potential_mode == "blend":
                    B_vertex = np.ones(nq, dtype=float)
                    for f in connectivity.faces_per_vertex[v]:
                        B_vertex -= B_f[f]
                    for edge_key in connectivity.edges_per_vertex[v]:
                        idx = edge_index[edge_key]
                        B_vertex -= B_e[idx]
                else:
                    raise ValueError(f"Unknown vertex_potential_mode: {vertex_potential_mode}")

                denom = np.power(r_v, p)
                I_v = np.where(denom <= eps, singular_value, B_vertex / denom)
                vertex_sum += I_v

    return PotentialComponents(face=face_sum, edge=edge_sum, vertex=vertex_sum)

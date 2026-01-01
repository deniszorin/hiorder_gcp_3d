"""Geometry and mesh connectivity helpers using indexed edges."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from cone_convex_hull import pointed_vertex


ArrayF = np.ndarray
ArrayI = np.ndarray


@dataclass(frozen=True)
class MeshData:
    """Mesh data with indexed connectivity.

    V: (N, 3) float array of vertex positions.
    faces: (M, 3) int array of face indices (CCW order).
    edges: edge index -> [v0, v1]
    vertices_to_faces: vertex index -> list of adjacent face indices
    edges_to_faces: edge index -> list of incident face indices
    vertices_to_edges: vertex index -> list of incident edge indices
    """

    V: ArrayF
    faces: ArrayI
    edges: List[List[int]] = field(init=False)
    vertices_to_faces: List[List[int]] = field(init=False)
    edges_to_faces: List[List[int]] = field(init=False)
    vertices_to_edges: List[List[int]] = field(init=False)

    def __post_init__(self) -> None:
        edges, vertices_to_faces, edges_to_faces, vertices_to_edges = build_connectivity(self)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "vertices_to_faces", vertices_to_faces)
        object.__setattr__(self, "edges_to_faces", edges_to_faces)
        object.__setattr__(self, "vertices_to_edges", vertices_to_edges)


@dataclass(frozen=True)
class MeshGeometry:
    """Per-face and per-edge geometry for fast evaluation.

    normals: (M, 3) unit normals.
    edge_dirs: (M, 3, 3) edge direction unit vectors for edges (v0->v1, v1->v2, v2->v0).
    edge_inward: (M, 3, 3) inward edge normals (n x d_e).
    edge_normals: (E, 3) average normals per edge (sum of incident face normals, normalized).
    pointed_vertices: (N,) boolean array of pointed vertex flags.
    """

    normals: ArrayF
    edge_dirs: ArrayF
    edge_inward: ArrayF
    edge_normals: ArrayF
    pointed_vertices: ArrayF


def load_obj_mesh(path: str) -> MeshData:
    """Load a triangle mesh OBJ with libigl."""

    try:
        import igl
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("libigl is required to load OBJ meshes here.") from exc

    v, f = igl.read_triangle_mesh(path)
    return MeshData(V=np.asarray(v, dtype=float), faces=np.asarray(f, dtype=int))


def build_connectivity(mesh: MeshData) -> tuple[
    List[List[int]],
    List[List[int]],
    List[List[int]],
    List[List[int]],
]:
    """Build connectivity maps using indexed edges."""

    nv = mesh.V.shape[0]
    nf = mesh.faces.shape[0]
    vertices_to_faces: List[List[int]] = [[] for _ in range(nv)]
    vertices_to_edges: List[List[int]] = [[] for _ in range(nv)]

    edges: List[List[int]] = []
    edges_to_faces: List[List[int]] = []
    edge_index = {}

    for f in range(nf):
        v0, v1, v2 = mesh.faces[f].tolist()
        vertices_to_faces[v0].append(f)
        vertices_to_faces[v1].append(f)
        vertices_to_faces[v2].append(f)

        face_edges = [(v0, v1), (v1, v2), (v2, v0)]
        for a, b in face_edges:
            key = (a, b) if a < b else (b, a)
            if key not in edge_index:
                edge_index[key] = len(edges)
                edges.append([key[0], key[1]])
                edges_to_faces.append([])
            eidx = edge_index[key]
            edges_to_faces[eidx].append(f)

        per_vertex_edges = {
            v0: [(v0, v1), (v0, v2)],
            v1: [(v1, v2), (v1, v0)],
            v2: [(v2, v0), (v2, v1)],
        }
        for v, edge_pairs in per_vertex_edges.items():
            for a, b in edge_pairs:
                key = (a, b) if a < b else (b, a)
                eidx = edge_index[key]
                if eidx not in vertices_to_edges[v]:
                    vertices_to_edges[v].append(eidx)

    return edges, vertices_to_faces, edges_to_faces, vertices_to_edges


def _order_vectors_ccw(
    vectors: List[ArrayF], normal: ArrayF, eps: float = 1e-12
) -> np.ndarray | None:
    n_norm = np.linalg.norm(normal)
    if n_norm <= eps:
        return None
    n = normal / n_norm

    projections: List[ArrayF] = []
    for vec in vectors:
        proj = vec - np.dot(vec, n) * n
        proj_norm = np.linalg.norm(proj)
        if proj_norm > eps:
            proj = proj / proj_norm
        projections.append(proj)

    ref = None
    for proj in projections:
        if np.linalg.norm(proj) > eps:
            ref = proj
            break
    if ref is None:
        return None

    angles = []
    for proj in projections:
        proj_norm = np.linalg.norm(proj)
        if proj_norm <= eps:
            angle = 0.0
        else:
            angle = np.arctan2(np.dot(n, np.cross(ref, proj)), np.dot(ref, proj))
        angles.append(angle)

    return np.argsort(np.asarray(angles))

# stupid but unavoidable without extra assumptions on the collision mesh data 
# it is unclear if it guarantees ordering
def _ordered_vertex_neighbors(mesh: MeshData, v_idx: int) -> List[int] | None:
    next_map: dict[int, int] = {}
    prev_map: dict[int, int] = {}
    neighbors: set[int] = set()

    for f in mesh.vertices_to_faces[v_idx]:
        face = mesh.faces[f]
        loc = None
        for i in range(3):
            if face[i] == v_idx:
                loc = i
                break
        if loc is None:
            continue
        neighbor_after = int(face[(loc + 1) % 3])
        neighbor_before = int(face[(loc + 2) % 3])
        neighbors.add(neighbor_after)
        neighbors.add(neighbor_before)

        if neighbor_after in next_map and next_map[neighbor_after] != neighbor_before:
            return None
        next_map[neighbor_after] = neighbor_before

        if neighbor_before in prev_map and prev_map[neighbor_before] != neighbor_after:
            return None
        prev_map[neighbor_before] = neighbor_after

    if not next_map:
        return None

    start = None
    for neighbor in next_map:
        if neighbor not in prev_map:
            start = neighbor
            break
    if start is None:
        start = min(next_map)

    order: List[int] = []
    current = start
    visited: set[int] = set()
    while current is not None and current not in visited:
        visited.add(current)
        order.append(current)
        current = next_map.get(current)
        if current is None or current == start:
            break

    if len(order) != len(neighbors):
        return None
    return order


def precompute_mesh_geometry(mesh: MeshData) -> MeshGeometry:
    """Precompute mesh geometry used by potentials."""

    V = mesh.V
    faces = mesh.faces
    nf = faces.shape[0]

    normals = np.zeros((nf, 3), dtype=float)
    edge_dirs = np.zeros((nf, 3, 3), dtype=float)
    edge_inward = np.zeros((nf, 3, 3), dtype=float)

    for f in range(nf):
        v0, v1, v2 = faces[f].tolist()
        p0 = V[v0]
        p1 = V[v1]
        p2 = V[v2]

        e0 = p1 - p0
        e1 = p2 - p1
        e2 = p0 - p2

        n = np.cross(p1 - p0, p2 - p0)
        n_norm = np.linalg.norm(n)
        if n_norm <= 1e-12:
            raise ValueError(f"Degenerate face normal at index {f}.")
        n = n / n_norm
        normals[f] = n

        edges = [e0, e1, e2]
        for i, e in enumerate(edges):
            e_norm = np.linalg.norm(e)
            if e_norm <= 1e-12:
                raise ValueError(f"Degenerate edge at face {f}, edge {i}.")
            d_e = e / e_norm
            edge_dirs[f, i] = d_e
            edge_inward[f, i] = np.cross(n, d_e)

    edge_normals = np.zeros((len(mesh.edges), 3), dtype=float)
    for eidx, face_list in enumerate(mesh.edges_to_faces):
        n_sum = np.zeros(3, dtype=float)
        for f in face_list:
            n_sum += normals[f]
        n_norm = np.linalg.norm(n_sum)
        if n_norm > 1e-12:
            n_sum = n_sum / n_norm
        edge_normals[eidx] = n_sum

    nv = V.shape[0]
    pointed_vertices = np.zeros(nv, dtype=bool)
    for v in range(nv):
        neighbors = _ordered_vertex_neighbors(mesh, v)
        if neighbors is None:
            raise ValueError(f"Failed to order vertex neighbors at vertex {v}.")
        vectors = []
        for other in neighbors:
            vec = V[other] - V[v]
            vec_norm = np.linalg.norm(vec)
            if vec_norm <= 1e-12:
                continue
            vectors.append(vec / vec_norm)
        if vectors:
            ordered_vectors = np.asarray(vectors, dtype=float)
        else:
            ordered_vectors = np.zeros((0, 3), dtype=float)
        pointed_vertices[v] = pointed_vertex(ordered_vectors)

    return MeshGeometry(
        normals=normals,
        edge_dirs=edge_dirs,
        edge_inward=edge_inward,
        edge_normals=edge_normals,
        pointed_vertices=pointed_vertices,
    )

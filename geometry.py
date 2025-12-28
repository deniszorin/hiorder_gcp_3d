"""Geometry and mesh connectivity helpers using indexed edges."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


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
    """

    normals: ArrayF
    edge_dirs: ArrayF
    edge_inward: ArrayF
    edge_normals: ArrayF


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

    return MeshGeometry(
        normals=normals,
        edge_dirs=edge_dirs,
        edge_inward=edge_inward,
        edge_normals=edge_normals,
    )

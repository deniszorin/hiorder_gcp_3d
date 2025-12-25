"""Geometry and mesh connectivity helpers using indexed edges."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


ArrayF = np.ndarray
ArrayI = np.ndarray


@dataclass(frozen=True)
class MeshData:
    """Raw mesh data.

    V: (N, 3) float array of vertex positions.
    F: (M, 3) int array of face indices (CCW order).
    """

    V: ArrayF
    F: ArrayI


@dataclass(frozen=True)
class MeshConnectivity:
    """Connectivity maps built from MeshData.

    vertices_per_edge: edge index -> [v0, v1]
    faces_per_vertex: vertex index -> list of adjacent face indices
    faces_per_edge: edge index -> list of incident face indices
    edges_per_vertex: vertex index -> list of incident edge indices
    """

    vertices_per_edge: List[List[int]]
    faces_per_vertex: List[List[int]]
    faces_per_edge: List[List[int]]
    edges_per_vertex: List[List[int]]


@dataclass(frozen=True)
class FaceGeometry:
    """Per-face geometry for fast evaluation.

    normals: (M, 3) unit normals.
    edge_dirs: (M, 3, 3) edge direction unit vectors for edges (v0->v1, v1->v2, v2->v0).
    edge_inward: (M, 3, 3) inward edge normals (n x d_e).
    """

    normals: ArrayF
    edge_dirs: ArrayF
    edge_inward: ArrayF


def load_obj_mesh(path: str) -> MeshData:
    """Load a triangle mesh OBJ with libigl."""

    try:
        import igl
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("libigl is required to load OBJ meshes here.") from exc

    v, f = igl.read_triangle_mesh(path)
    return MeshData(V=np.asarray(v, dtype=float), F=np.asarray(f, dtype=int))


def build_connectivity(mesh: MeshData) -> MeshConnectivity:
    """Build connectivity maps using indexed edges."""

    nv = mesh.V.shape[0]
    nf = mesh.F.shape[0]
    faces_per_vertex: List[List[int]] = [[] for _ in range(nv)]
    edges_per_vertex: List[List[int]] = [[] for _ in range(nv)]

    vertices_per_edge: List[List[int]] = []
    faces_per_edge: List[List[int]] = []
    edge_index = {}

    for f in range(nf):
        v0, v1, v2 = mesh.F[f].tolist()
        faces_per_vertex[v0].append(f)
        faces_per_vertex[v1].append(f)
        faces_per_vertex[v2].append(f)

        face_edges = [(v0, v1), (v1, v2), (v2, v0)]
        for a, b in face_edges:
            key = (a, b) if a < b else (b, a)
            if key not in edge_index:
                edge_index[key] = len(vertices_per_edge)
                vertices_per_edge.append([key[0], key[1]])
                faces_per_edge.append([])
            eidx = edge_index[key]
            faces_per_edge[eidx].append(f)

        per_vertex_edges = {
            v0: [(v0, v1), (v0, v2)],
            v1: [(v1, v2), (v1, v0)],
            v2: [(v2, v0), (v2, v1)],
        }
        for v, edge_pairs in per_vertex_edges.items():
            for a, b in edge_pairs:
                key = (a, b) if a < b else (b, a)
                eidx = edge_index[key]
                if eidx not in edges_per_vertex[v]:
                    edges_per_vertex[v].append(eidx)

    return MeshConnectivity(
        vertices_per_edge=vertices_per_edge,
        faces_per_vertex=faces_per_vertex,
        faces_per_edge=faces_per_edge,
        edges_per_vertex=edges_per_vertex,
    )


def precompute_face_geometry(mesh: MeshData) -> FaceGeometry:
    """Precompute per-face geometry used by potentials."""

    V = mesh.V
    F = mesh.F
    nf = F.shape[0]

    normals = np.zeros((nf, 3), dtype=float)
    edge_dirs = np.zeros((nf, 3, 3), dtype=float)
    edge_inward = np.zeros((nf, 3, 3), dtype=float)

    for f in range(nf):
        v0, v1, v2 = F[f].tolist()
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

    return FaceGeometry(normals=normals, edge_dirs=edge_dirs, edge_inward=edge_inward)

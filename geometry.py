"""Geometry and mesh connectivity helpers (skeleton)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


ArrayF = np.ndarray
ArrayI = np.ndarray
EdgeKey = Tuple[int, int]


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

    faces_per_vertex: list of face indices for each vertex.
    faces_per_edge: dict edge -> list of incident face indices.
    edges_per_vertex: list of edges (as sorted index pairs) for each vertex.
    edge_local_indices: dict edge -> list of local edge indices per incident face.
    """
    faces_per_vertex: List[List[int]]
    faces_per_edge: Dict[EdgeKey, List[int]]
    edges_per_vertex: List[List[EdgeKey]]
    edge_local_indices: Dict[EdgeKey, List[int]]


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
    """Build connectivity maps for vertices and edges."""

    nv = mesh.V.shape[0]
    nf = mesh.F.shape[0]
    faces_per_vertex: List[List[int]] = [[] for _ in range(nv)]
    faces_per_edge: Dict[EdgeKey, List[int]] = {}
    edge_local_indices: Dict[EdgeKey, List[int]] = {}
    edges_per_vertex: List[List[EdgeKey]] = [[] for _ in range(nv)]
    edges_per_vertex_set: List[set[EdgeKey]] = [set() for _ in range(nv)]

    for f in range(nf):
        v0, v1, v2 = mesh.F[f].tolist()
        faces_per_vertex[v0].append(f)
        faces_per_vertex[v1].append(f)
        faces_per_vertex[v2].append(f)

        edges = [(v0, v1), (v1, v2), (v2, v0)]
        for local_idx, (a, b) in enumerate(edges):
            key = (a, b) if a < b else (b, a)
            faces_per_edge.setdefault(key, []).append(f)
            edge_local_indices.setdefault(key, []).append(local_idx)

        per_vertex_edges = {
            v0: [(v0, v1), (v0, v2)],
            v1: [(v1, v2), (v1, v0)],
            v2: [(v2, v0), (v2, v1)],
        }
        for v, edge_pairs in per_vertex_edges.items():
            for a, b in edge_pairs:
                key = (a, b) if a < b else (b, a)
                if key not in edges_per_vertex_set[v]:
                    edges_per_vertex_set[v].add(key)
                    edges_per_vertex[v].append(key)

    return MeshConnectivity(
        faces_per_vertex=faces_per_vertex,
        faces_per_edge=faces_per_edge,
        edges_per_vertex=edges_per_vertex,
        edge_local_indices=edge_local_indices,
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

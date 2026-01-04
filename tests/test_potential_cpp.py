import numpy as np

import geometry
import potential_cpp
import potential_numba
import viz


def _sample_mesh_grid(mesh, resolution):
    bounds_min = mesh.V.min(axis=0) - 0.5
    bounds_max = mesh.V.max(axis=0) + 0.5
    return viz.sample_volume_grid(bounds_min, bounds_max, resolution=resolution)


def _bounding_box_size(mesh):
    bounds_min = mesh.V.min(axis=0)
    bounds_max = mesh.V.max(axis=0)
    return float(np.max(bounds_max - bounds_min))


def _ensure_cpp_extension():
    potential_cpp._build_extension()


def _cpp_vs_numba(q, mesh, one_sided=False):
    geom = geometry.precompute_mesh_geometry(mesh)
    ref = potential_numba.smoothed_offset_potential_numba(
        q,
        mesh,
        geom,
        one_sided=one_sided,
    )
    cpp_vals = potential_cpp.smoothed_offset_potential_cpp(
        q,
        mesh.V,
        mesh.faces,
        one_sided=one_sided,
    )
    np.testing.assert_allclose(cpp_vals, ref, rtol=1e-6, atol=1e-6)


def test_smoothed_offset_potential_cpp_matches_validation_scenes():
    scenes = viz.build_validation_scene_specs()
    for scene in scenes:
        mesh = scene.mesh
        q = _sample_mesh_grid(mesh, resolution=10)
        _cpp_vs_numba(q, mesh, one_sided=scene.one_sided)


def test_smoothed_offset_potential_cpp_tetrahedron_one_sided_variants():
    V = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    faces = np.array(
        [
            [0, 1, 2],
            [1, 0, 3],
            [3, 0, 2],
            [2, 1, 3],
        ],
        dtype=int,
    )
    mesh = geometry.MeshData(V=V, faces=faces)
    q = _sample_mesh_grid(mesh, resolution=10)

    for one_sided in (False, True):
        _cpp_vs_numba(q, mesh, one_sided=one_sided)


def test_smoothed_offset_potential_cpp_flipped_normals():
    V = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    faces = np.array(
        [
            [0, 1, 2],
            [1, 0, 3],
            [3, 0, 2],
            [2, 1, 3],
        ],
        dtype=int,
    )
    faces = faces[:, [0, 2, 1]]
    mesh = geometry.MeshData(V=V, faces=faces)
    q = _sample_mesh_grid(mesh, resolution=10)
    _cpp_vs_numba(q, mesh, one_sided=False)

    flipped_scenes = viz.build_validation_scene_specs(reverse_faces=True)
    for scene in flipped_scenes:
        if scene.name not in ("double_cone_nonconvex_out", "double_cone_nonconvex_both"):
            continue
        mesh = scene.mesh
        q = _sample_mesh_grid(mesh, resolution=10)
        _cpp_vs_numba(q, mesh, one_sided=scene.one_sided)


def test_smoothed_offset_potential_cpp_localized_matches_numba_three_meshes():
    _ensure_cpp_extension()
    scenes = viz.build_validation_scene_specs()
    for scene in scenes[:3]:
        mesh = scene.mesh
        geom = geometry.precompute_mesh_geometry(mesh)
        q = _sample_mesh_grid(mesh, resolution=10)
        epsilon = 0.1 * _bounding_box_size(mesh)
        ref = potential_numba.smoothed_offset_potential_numba(
            q,
            mesh,
            geom,
            epsilon=epsilon,
            localized=True,
            one_sided=scene.one_sided,
        )
        cpp_vals = potential_cpp.smoothed_offset_potential_cpp(
            q,
            mesh.V,
            mesh.faces,
            epsilon=epsilon,
            localized=True,
            one_sided=scene.one_sided,
        )
        np.testing.assert_allclose(cpp_vals, ref, rtol=1e-6, atol=1e-6)


def test_smoothed_offset_potential_cpp_accelerated_matches_numba_three_meshes():
    _ensure_cpp_extension()
    scenes = viz.build_validation_scene_specs()
    for scene in scenes[:3]:
        mesh = scene.mesh
        geom = geometry.precompute_mesh_geometry(mesh)
        q = _sample_mesh_grid(mesh, resolution=10)
        epsilon = 0.1 * _bounding_box_size(mesh)
        ref = potential_numba.smoothed_offset_potential_numba(
            q,
            mesh,
            geom,
            epsilon=epsilon,
            localized=True,
            one_sided=scene.one_sided,
        )
        cpp_vals = potential_cpp.smoothed_offset_potential_accelerated_cpp(
            q,
            mesh.V,
            mesh.faces,
            epsilon=epsilon,
            localized=True,
            one_sided=scene.one_sided,
        )
        np.testing.assert_allclose(cpp_vals, ref, rtol=1e-6, atol=1e-6)

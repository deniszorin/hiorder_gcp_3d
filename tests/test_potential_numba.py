import numpy as np

import geometry
import potential
import potential_numba
import viz


def _mesh_single_triangle():
    V = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2]], dtype=int)
    return geometry.MeshData(V=V, faces=faces)


def _sample_mesh_grid(mesh, resolution):
    bounds_min = mesh.V.min(axis=0) - 0.5
    bounds_max = mesh.V.max(axis=0) + 0.5
    return viz.sample_volume_grid(bounds_min, bounds_max, resolution=resolution)


def test_smoothed_offset_potential_numba_triangle_matches():
    mesh = _mesh_single_triangle()
    geom = geometry.precompute_mesh_geometry(mesh)
    q = _sample_mesh_grid(mesh, resolution=10)

    ref = potential.smoothed_offset_potential(q, mesh, geom)
    numba_vals = potential_numba.smoothed_offset_potential_numba(q, mesh, geom)

    np.testing.assert_allclose(numba_vals, ref, rtol=1e-6, atol=1e-6)


def test_smoothed_offset_potential_numba_matches_validation_scenes():
    scenes = viz.build_validation_scene_specs()
    for scene in scenes:
        print(f"scene: {scene.name}")
        mesh = scene.mesh
        geom = geometry.precompute_mesh_geometry(mesh)
        q = _sample_mesh_grid(mesh, resolution=10)

        ref = potential.smoothed_offset_potential(
            q,
            mesh,
            geom,
            one_sided=scene.one_sided,
        )
        numba_vals = potential_numba.smoothed_offset_potential_numba(
            q,
            mesh,
            geom,
            one_sided=scene.one_sided,
        )

        np.testing.assert_allclose(numba_vals, ref, rtol=1e-6, atol=1e-6)


def test_smoothed_offset_potential_numba_tetrahedron_one_sided_variants():
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
    geom = geometry.precompute_mesh_geometry(mesh)
    q = _sample_mesh_grid(mesh, resolution=10)

    for one_sided in (False, True):
        print(f"scene: tetrahedron one_sided={one_sided}")
        ref = potential.smoothed_offset_potential(
            q,
            mesh,
            geom,
            one_sided=one_sided,
        )
        numba_vals = potential_numba.smoothed_offset_potential_numba(
            q,
            mesh,
            geom,
            one_sided=one_sided,
        )
        np.testing.assert_allclose(numba_vals, ref, rtol=1e-6, atol=1e-6)


def test_smoothed_offset_potential_numba_flipped_normals():
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
    geom = geometry.precompute_mesh_geometry(mesh)
    q = _sample_mesh_grid(mesh, resolution=10)

    print("scene: tetrahedron_flipped")
    ref = potential.smoothed_offset_potential(q, mesh, geom, one_sided=False)
    numba_vals = potential_numba.smoothed_offset_potential_numba(
        q, mesh, geom, one_sided=False
    )
    np.testing.assert_allclose(numba_vals, ref, rtol=1e-6, atol=1e-6)

    flipped_scenes = viz.build_validation_scene_specs(reverse_faces=True)
    for scene in flipped_scenes:
        if scene.name not in ("double_cone_nonconvex_out", "double_cone_nonconvex_both"):
            continue
        print(f"scene: {scene.name}_flipped")
        mesh = scene.mesh
        geom = geometry.precompute_mesh_geometry(mesh)
        q = _sample_mesh_grid(mesh, resolution=10)

        ref = potential.smoothed_offset_potential(
            q,
            mesh,
            geom,
            one_sided=scene.one_sided,
        )
        numba_vals = potential_numba.smoothed_offset_potential_numba(
            q,
            mesh,
            geom,
            one_sided=scene.one_sided,
        )
        np.testing.assert_allclose(numba_vals, ref, rtol=1e-6, atol=1e-6)

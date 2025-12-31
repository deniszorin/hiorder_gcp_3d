import numpy as np

import geometry
import potential_cpp
import viz


def _assert_pointed_vertices_match(mesh, name):
    geom = geometry.precompute_mesh_geometry(mesh)
    cpp_pointed = potential_cpp.pointed_vertices_cpp(
        mesh.V, mesh.faces,
    )
    py_pointed = geom.pointed_vertices
    if not np.array_equal(cpp_pointed, py_pointed):
        mismatched = np.where(cpp_pointed != py_pointed)[0]
        message = (
            f"Pointed vertices mismatch for {name}: "
            f"{mismatched.size} mismatched indices, "
            f"first 10 {mismatched[:10].tolist()}"
        )
        raise AssertionError(message)


def test_pointed_vertices_cpp_matches_validation_scenes():
    scenes = viz.build_validation_scene_specs()
    for scene in scenes:
        _assert_pointed_vertices_match(scene.mesh, scene.name)


def test_pointed_vertices_cpp_matches_validation_scenes_flipped():
    scenes = viz.build_validation_scene_specs(reverse_faces=True)
    for scene in scenes:
        _assert_pointed_vertices_match(scene.mesh, f"{scene.name}_flipped")

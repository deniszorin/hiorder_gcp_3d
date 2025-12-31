import test_cone_convex_hull as ref_tests

import cone_convex_hull_cpp


def _run_with_cpp(test_func):
    original = ref_tests.cone_convex_hull
    ref_tests.cone_convex_hull = cone_convex_hull_cpp.cone_convex_hull_cpp
    try:
        test_func()
    finally:
        ref_tests.cone_convex_hull = original


def test_cone_convex_hull_cpp_matches_reference():
    tests = [
        ref_tests.test_convex_polygon_inside_apex,
        ref_tests.test_convex_polygon_outside_apex,
        ref_tests.test_convex_polygon_with_midpoints,
        ref_tests.test_star_polygon_alternating,
        ref_tests.test_all_coplanar_returns_input,
        ref_tests.test_spiral_polygon,
        ref_tests.test_convex_lift_two_vertices,
        ref_tests.test_convex_lift_two_vertices_perturbed,
        ref_tests.test_nonconvex_lift_two_vertices,
        ref_tests.test_nonconvex_lift_two_vertices_perturbed,
        ref_tests.test_convex_lift_three_vertices,
        ref_tests.test_nonconvex_lift_three_vertices,
    ]
    for test_func in tests:
        _run_with_cpp(test_func)


def test_cone_convex_hull_cpp_corner_case_exact():
    for (
        _name,
        m,
        n,
        expected,
        expected_valid,
        expected_coplanar,
        _expected_perturbed,
        _expected_perturbed_valid,
    ) in ref_tests._CORNER_CASES:
        e = ref_tests._build_corner_case(m, n)
        D, coplanar, fullspace = cone_convex_hull_cpp.cone_convex_hull_cpp(e)
        assert coplanar is expected_coplanar
        assert not fullspace
        assert ref_tests.np.array_equal(D, ref_tests.np.array(expected, dtype=int))
        assert (
            ref_tests.validate_cone_convex_hull(e, D, eps=1e-12)
            is expected_valid
        )


def test_cone_convex_hull_cpp_corner_case_perturbed():
    for (
        _name,
        m,
        n,
        _expected,
        _expected_valid,
        _expected_coplanar,
        expected_perturbed,
        expected_perturbed_valid,
    ) in ref_tests._CORNER_CASES:
        e = ref_tests._build_corner_case(m, n)
        e = ref_tests._perturb_edges(e)
        D, coplanar, fullspace = cone_convex_hull_cpp.cone_convex_hull_cpp(e)
        assert not coplanar
        assert not fullspace
        assert ref_tests.np.array_equal(D, ref_tests.np.array(expected_perturbed, dtype=int))
        assert (
            ref_tests.validate_cone_convex_hull(e, D, eps=1e-12)
            is expected_perturbed_valid
        )

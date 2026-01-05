#include "collision_mesh.hpp"

#include <Eigen/Core>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

Eigen::MatrixXi make_matrix(
    int rows, int cols, const std::initializer_list<int>& values)
{
    assert(static_cast<int>(values.size()) == rows * cols);
    Eigen::MatrixXi mat(rows, cols);
    int idx = 0;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            mat(r, c) = *(values.begin() + idx);
            idx++;
        }
    }
    return mat;
}

void expect_matrix_eq(
    const Eigen::MatrixXi& actual,
    const Eigen::MatrixXi& expected,
    const std::string& name)
{
    if (actual.rows() != expected.rows()
        || actual.cols() != expected.cols()) {
        throw std::runtime_error(name + " shape mismatch");
    }
    for (int r = 0; r < actual.rows(); r++) {
        for (int c = 0; c < actual.cols(); c++) {
            if (actual(r, c) != expected(r, c)) {
                throw std::runtime_error(name + " value mismatch");
            }
        }
    }
}

void expect_vecvec_eq(
    const std::vector<std::vector<int>>& actual,
    const std::vector<std::vector<int>>& expected,
    const std::string& name)
{
    if (actual.size() != expected.size()) {
        throw std::runtime_error(name + " size mismatch");
    }
    for (size_t i = 0; i < expected.size(); i++) {
        if (actual[i] != expected[i]) {
            throw std::runtime_error(name + " value mismatch");
        }
    }
}

void expect_throw(const std::function<void()>& fn, const std::string& name)
{
    try {
        fn();
    } catch (const std::runtime_error&) {
        return;
    }
    throw std::runtime_error(name + " expected exception");
}

} // namespace

static void test_single_triangle()
{
    Eigen::MatrixXd V(3, 3);
    V << 0, 0, 0, 1, 0, 0, 0, 1, 0;
    Eigen::MatrixXi F = make_matrix(1, 3, {0, 1, 2});

    ipc::CollisionMesh mesh(V, F);

    expect_matrix_eq(
        mesh.edges(),
        make_matrix(3, 2, {0, 1, 1, 2, 0, 2}),
        "edges");
    expect_matrix_eq(
        mesh.faces_to_edges(),
        make_matrix(1, 3, {0, 1, 2}),
        "faces_to_edges");
    expect_matrix_eq(
        mesh.edges_to_faces(),
        make_matrix(3, 2, {0, -1, 0, -1, 0, -1}),
        "edges_to_faces");
    expect_vecvec_eq(
        mesh.vertices_to_faces(),
        {{0}, {0}, {0}},
        "vertices_to_faces");
    expect_vecvec_eq(
        mesh.vertices_to_edges(),
        {{0, 2}, {0, 1}, {1, 2}},
        "vertices_to_edges");
}

static void test_two_triangles_shared_edge()
{
    Eigen::MatrixXd V(4, 3);
    V << 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0;
    Eigen::MatrixXi F = make_matrix(2, 3, {0, 1, 2, 1, 3, 2});

    ipc::CollisionMesh mesh(V, F);

    expect_matrix_eq(
        mesh.edges(),
        make_matrix(5, 2, {0, 1, 1, 2, 0, 2, 1, 3, 2, 3}),
        "edges");
    expect_matrix_eq(
        mesh.faces_to_edges(),
        make_matrix(2, 3, {0, 1, 2, 3, 4, 1}),
        "faces_to_edges");
    expect_matrix_eq(
        mesh.edges_to_faces(),
        make_matrix(5, 2, {0, -1, 0, 1, 0, -1, 1, -1, 1, -1}),
        "edges_to_faces");
    expect_vecvec_eq(
        mesh.vertices_to_faces(),
        {{0}, {0, 1}, {0, 1}, {1}},
        "vertices_to_faces");
    expect_vecvec_eq(
        mesh.vertices_to_edges(),
        {{0, 2}, {0, 1, 3}, {1, 2, 4}, {3, 4}},
        "vertices_to_edges");
}

static void test_tetrahedron()
{
    Eigen::MatrixXd V(4, 3);
    V << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1;
    Eigen::MatrixXi F = make_matrix(
        4, 3, {0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2});

    ipc::CollisionMesh mesh(V, F);

    expect_matrix_eq(
        mesh.edges(),
        make_matrix(6, 2, {0, 1, 1, 2, 0, 2, 2, 3, 0, 3, 1, 3}),
        "edges");
    expect_matrix_eq(
        mesh.faces_to_edges(),
        make_matrix(4, 3, {0, 1, 2, 2, 3, 4, 4, 5, 0, 5, 3, 1}),
        "faces_to_edges");
    expect_matrix_eq(
        mesh.edges_to_faces(),
        make_matrix(6, 2, {0, 2, 0, 3, 0, 1, 1, 3, 1, 2, 2, 3}),
        "edges_to_faces");
    expect_vecvec_eq(
        mesh.vertices_to_faces(),
        {{0, 1, 2}, {0, 2, 3}, {0, 1, 3}, {1, 2, 3}},
        "vertices_to_faces");
    expect_vecvec_eq(
        mesh.vertices_to_edges(),
        {{0, 2, 4}, {0, 1, 5}, {1, 2, 3}, {3, 4, 5}},
        "vertices_to_edges");
}

static void test_tetrahedron_missing_face()
{
    Eigen::MatrixXd V(4, 3);
    V << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1;
    Eigen::MatrixXi F = make_matrix(3, 3, {0, 1, 2, 0, 2, 3, 0, 3, 1});

    ipc::CollisionMesh mesh(V, F);

    expect_matrix_eq(
        mesh.edges(),
        make_matrix(6, 2, {0, 1, 1, 2, 0, 2, 2, 3, 0, 3, 1, 3}),
        "edges");
    expect_matrix_eq(
        mesh.faces_to_edges(),
        make_matrix(3, 3, {0, 1, 2, 2, 3, 4, 4, 5, 0}),
        "faces_to_edges");
    expect_matrix_eq(
        mesh.edges_to_faces(),
        make_matrix(6, 2, {0, 2, 0, -1, 0, 1, 1, -1, 1, 2, 2, -1}),
        "edges_to_faces");
    expect_vecvec_eq(
        mesh.vertices_to_faces(),
        {{0, 1, 2}, {0, 2}, {0, 1}, {1, 2}},
        "vertices_to_faces");
    expect_vecvec_eq(
        mesh.vertices_to_edges(),
        {{0, 2, 4}, {0, 1, 5}, {1, 2, 3}, {3, 4, 5}},
        "vertices_to_edges");
}

static void test_cube()
{
    Eigen::MatrixXd V(8, 3);
    V << 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1,
        1, 0, 1, 1;
    Eigen::MatrixXi F = make_matrix(
        12, 3,
        {0, 1, 2, 0, 2, 3, 0, 1, 5, 0, 5, 4, 0, 3, 7, 0, 7, 4, 4, 5, 6, 4,
         6, 7, 3, 2, 6, 3, 6, 7, 1, 2, 6, 1, 6, 5});

    ipc::CollisionMesh mesh(V, F);

    expect_matrix_eq(
        mesh.edges(),
        make_matrix(
            18, 2,
            {0, 1, 1, 2, 0, 2, 2, 3, 0, 3, 1, 5, 0, 5, 4, 5, 0, 4, 3, 7, 0,
             7, 4, 7, 5, 6, 4, 6, 6, 7, 2, 6, 3, 6, 1, 6}),
        "edges");
    expect_matrix_eq(
        mesh.faces_to_edges(),
        make_matrix(
            12, 3,
            {0, 1, 2, 2, 3, 4, 0, 5, 6, 6, 7, 8, 4, 9, 10, 10, 11, 8, 7, 12,
             13, 13, 14, 11, 3, 15, 16, 16, 14, 9, 1, 15, 17, 17, 12, 5}),
        "faces_to_edges");
    expect_matrix_eq(
        mesh.edges_to_faces(),
        make_matrix(
            18, 2,
            {0, 2, 0, 10, 0, 1, 1, 8, 1, 4, 2, 11, 2, 3, 3, 6, 3, 5, 4, 9,
             4, 5, 5, 7, 6, 11, 6, 7, 7, 9, 8, 10, 8, 9, 10, 11}),
        "edges_to_faces");
    expect_vecvec_eq(
        mesh.vertices_to_faces(),
        {{0, 1, 2, 3, 4, 5},
         {0, 2, 10, 11},
         {0, 1, 8, 10},
         {1, 4, 8, 9},
         {3, 5, 6, 7},
         {2, 3, 6, 11},
         {6, 7, 8, 9, 10, 11},
         {4, 5, 7, 9}},
        "vertices_to_faces");
    expect_vecvec_eq(
        mesh.vertices_to_edges(),
        {{0, 2, 4, 6, 8, 10},
         {0, 1, 5, 17},
         {1, 2, 3, 15},
         {3, 4, 9, 16},
         {7, 8, 11, 13},
         {5, 6, 7, 12},
         {12, 13, 14, 15, 16, 17},
         {9, 10, 11, 14}},
        "vertices_to_edges");
}

static void test_cube_missing_face()
{
    Eigen::MatrixXd V(8, 3);
    V << 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1,
        1, 0, 1, 1;
    Eigen::MatrixXi F = make_matrix(
        10, 3,
        {0, 1, 2, 0, 2, 3, 0, 1, 5, 0, 5, 4, 0, 3, 7, 0, 7, 4, 3, 2, 6, 3,
         6, 7, 1, 2, 6, 1, 6, 5});

    ipc::CollisionMesh mesh(V, F);

    expect_matrix_eq(
        mesh.edges(),
        make_matrix(
            17, 2,
            {0, 1, 1, 2, 0, 2, 2, 3, 0, 3, 1, 5, 0, 5, 4, 5, 0, 4, 3, 7, 0,
             7, 4, 7, 2, 6, 3, 6, 6, 7, 1, 6, 5, 6}),
        "edges");
    expect_matrix_eq(
        mesh.faces_to_edges(),
        make_matrix(
            10, 3,
            {0, 1, 2, 2, 3, 4, 0, 5, 6, 6, 7, 8, 4, 9, 10, 10, 11, 8, 3, 12,
             13, 13, 14, 9, 1, 12, 15, 15, 16, 5}),
        "faces_to_edges");
    expect_matrix_eq(
        mesh.edges_to_faces(),
        make_matrix(
            17, 2,
            {0, 2, 0, 8, 0, 1, 1, 6, 1, 4, 2, 9, 2, 3, 3, -1, 3, 5, 4, 7,
             4, 5, 5, -1, 6, 8, 6, 7, 7, -1, 8, 9, 9, -1}),
        "edges_to_faces");
    expect_vecvec_eq(
        mesh.vertices_to_faces(),
        {{0, 1, 2, 3, 4, 5},
         {0, 2, 8, 9},
         {0, 1, 6, 8},
         {1, 4, 6, 7},
         {3, 5},
         {2, 3, 9},
         {6, 7, 8, 9},
         {4, 5, 7}},
        "vertices_to_faces");
    expect_vecvec_eq(
        mesh.vertices_to_edges(),
        {{0, 2, 4, 6, 8, 10},
         {0, 1, 5, 15},
         {1, 2, 3, 12},
         {3, 4, 9, 13},
         {7, 8, 11},
         {5, 6, 7, 16},
         {12, 13, 14, 15, 16},
         {9, 10, 11, 14}},
        "vertices_to_edges");
}

static void test_bunny()
{
    ipc::CollisionMesh mesh =
        ipc::CollisionMesh::load_from_obj("tests/Bunny-LowPoly.obj");

    const auto& edges_to_faces = mesh.edges_to_faces();
    for (int e = 0; e < edges_to_faces.rows(); e++) {
        if (edges_to_faces(e, 0) < 0 || edges_to_faces(e, 1) < 0) {
            throw std::runtime_error("Bunny has boundary edges.");
        }
    }
    for (size_t v = 0; v < mesh.num_vertices(); v++) {
        if (mesh.is_vertex_on_boundary(static_cast<int>(v))) {
            throw std::runtime_error("Bunny has boundary vertices.");
        }
    }

    const int V = static_cast<int>(mesh.num_vertices());
    const int E = static_cast<int>(mesh.num_edges());
    const int F = static_cast<int>(mesh.num_faces());
    const int chi = V - E + F;
    const double genus = (2.0 - static_cast<double>(chi)) / 2.0;
    if (std::abs(genus) > 1e-8) {
        throw std::runtime_error("Bunny genus is not zero.");
    }
}

static void test_nonmanifold_edge()
{
    Eigen::MatrixXd V(5, 3);
    V << 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, -1, 0;
    Eigen::MatrixXi F = make_matrix(3, 3, {0, 1, 2, 1, 0, 3, 0, 1, 4});

    expect_throw([&]() { ipc::CollisionMesh mesh(V, F); }, "nonmanifold_edge");
}

static void test_nonmanifold_vertex()
{
    Eigen::MatrixXd V(5, 3);
    V << 0, 0, 0, 1, 0, 0, 0, 1, 0, -1, 0, 0, 0, -1, 0;
    Eigen::MatrixXi F = make_matrix(2, 3, {0, 1, 2, 0, 3, 4});

    expect_throw([&]() { ipc::CollisionMesh mesh(V, F); }, "nonmanifold_vertex");
}

int main()
{
    test_single_triangle();
    test_two_triangles_shared_edge();
    test_tetrahedron();
    test_tetrahedron_missing_face();
    test_cube();
    test_cube_missing_face();
    test_bunny();
    test_nonmanifold_edge();
    test_nonmanifold_vertex();

    std::cout << "collision_mesh tests passed\n";
    return 0;
}

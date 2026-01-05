#pragma once

#include "eigen_ext.hpp"

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace ipc {

/// @brief A minimal collision mesh for triangle or edge meshes.
/// stripped down version of the structure from ipc_toolkit 
/// to minimize dependencies/potential issues. 
/// main parts removed: everything related to the full <-> collision mesh maps 
///  geometry computations (only positions are stored)
///  everything related to codimensional edges and vertices

class CollisionMesh {
public:
    /// @brief Construct an empty collision mesh.
    CollisionMesh() = default;

    /// @brief Construct a collision mesh from vertex positions and faces.
    /// Keeping "rest_positions" name from the original
    explicit CollisionMesh(
        Eigen::ConstRef<Eigen::MatrixXd> rest_positions,
        Eigen::ConstRef<Eigen::MatrixXi> faces = Eigen::MatrixXi());

    /// @brief Construct a collision mesh (factory helper).
    static CollisionMesh build(
        Eigen::ConstRef<Eigen::MatrixXd> rest_positions,
        Eigen::ConstRef<Eigen::MatrixXi> faces = Eigen::MatrixXi());

    /// @brief Load a collision mesh from an OBJ file.
    static CollisionMesh load_from_obj(const std::string& path);

    /// @brief Destroy the Collision Mesh object.
    ~CollisionMesh() = default;

    /// @brief Get the number of vertices in the collision mesh.
    size_t num_vertices() const
    {
        return static_cast<size_t>(m_rest_positions.rows());
    }

    /// @brief Get the number of edges in the collision mesh.
    size_t num_edges() const { return static_cast<size_t>(m_edges.rows()); }

    /// @brief Get the number of faces in the collision mesh.
    size_t num_faces() const { return static_cast<size_t>(m_faces.rows()); }

    /// @brief Get the dimension of the mesh.
    size_t dim() const { return static_cast<size_t>(m_rest_positions.cols()); }

    /// @brief Get the number of degrees of freedom in the collision mesh.
    size_t ndof() const { return num_vertices() * dim(); }

    /// @brief Get the vertices of the collision mesh at rest (|V| × dim).
    const Eigen::MatrixXd& rest_positions() const { return m_rest_positions; }

    /// @brief Get the edges of the collision mesh (|E| × 2).
    const Eigen::MatrixXi& edges() const { return m_edges; }

    /// @brief Get the faces of the collision mesh (|F| × 3).
    const Eigen::MatrixXi& faces() const { return m_faces; }

    /// @brief Get the mapping from faces to edges of the collision mesh (|F| × 3).
    const Eigen::MatrixXi& faces_to_edges() const { return m_faces_to_edges; }

    /// @brief Get the mapping from edges to faces of the collision mesh (|E| × 2).
    const Eigen::MatrixXi& edges_to_faces() const { return m_edges_to_faces; }

    /// @brief Compute the rest length of an edge.
    double edge_length(int edge_id) const;

    /// @brief Compute the maximum rest length of all edges.
    double max_edge_length() const;

    /// @brief Get the mapping from vertices to edges of the collision mesh.
    const std::vector<std::vector<int>>& vertices_to_edges() const
    {
        return m_vertices_to_edges;
    }
    std::vector<std::vector<int>>& vertices_to_edges() { return m_vertices_to_edges; }

    /// @brief Get the mapping from vertices to faces of the collision mesh.
    const std::vector<std::vector<int>>& vertices_to_faces() const
    {
        return m_vertices_to_faces;
    }

    /// @brief Is a vertex on the boundary of the collision mesh?
    /// @param vi Vertex ID.
    /// @return True if the vertex is on the boundary of the collision mesh.
    bool is_vertex_on_boundary(const int vi) const
    {
        return m_is_vertex_on_boundary[vi];
    }

    /// @brief Construct a matrix that maps from the faces' edges to rows in the edges matrix.
    /// @param faces The face matrix of mesh (|F| × 3).
    /// @param edges The edge matrix of mesh (|E| × 2).
    /// @return Matrix that maps from the faces' edges to rows in the edges matrix.
    static Eigen::MatrixXi construct_faces_to_edges(
        Eigen::ConstRef<Eigen::MatrixXi> faces,
        Eigen::ConstRef<Eigen::MatrixXi> edges);

    /// A function that takes two vertex IDs and returns true if the vertices
    /// (and faces or edges containing the vertices) can collide. By default all
    /// primitives can collide with all other primitives.
    std::function<bool(size_t, size_t)> can_collide = default_can_collide;

protected:
    /// @brief Initialize map from edges to adjacent faces (|E| × 2).
    void init_edges_to_faces();

    /// @brief Initialize vertex to edge adjacency.
    void init_vertices_to_edges();

    /// @brief Initialize vertex to face adjacency.
    void init_vertices_to_faces();

    /// @brief Initialize vertex boundary flags.
    void init_boundary();

    /// @brief The vertex positions at rest (|V| × dim).
    Eigen::MatrixXd m_rest_positions;
    /// @brief Edges as rows of indicies into vertices (|E| × 2).
    Eigen::MatrixXi m_edges;
    /// @brief Triangular faces as rows of indicies into vertices (|F| × 3).
    Eigen::MatrixXi m_faces;
    /// @brief Map from faces edges to rows of edges (|F| × 3).
    Eigen::MatrixXi m_faces_to_edges;
    /// @brief Map from edges to adjacent faces (|E| × 2).
    Eigen::MatrixXi m_edges_to_faces;

    /// @brief For each vertex, the faces adjacent to it.
    std::vector<std::vector<int>> m_vertices_to_faces;
    /// @brief For each vertex, the edges adjacent to it.
    std::vector<std::vector<int>> m_vertices_to_edges;

    /// @brief Is vertex on the boundary of the triangle mesh in 3D or polyline in 2D?
    std::vector<bool> m_is_vertex_on_boundary;

private:
    /// @brief By default all primitives can collide with all other primitives.
    static bool default_can_collide(size_t /*unused*/, size_t /*unused*/)
    {
        return true;
    }
};

} // namespace ipc

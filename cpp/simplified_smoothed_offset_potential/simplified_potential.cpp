#include "simplified_potential_impl.hpp"
#include "simplified_potential.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

namespace ipc {

double simplified_smoothed_offset_potential_point(
    const Eigen::Vector3d& q, const std::vector<int>& face_indices,
    const PotentialCollisionMesh& mesh,
    const PotentialParameters& params,
    const std::vector<int>& edge_valence,
    const std::vector<char>& vertex_internal,
    const bool include_faces, const bool include_edges, const bool include_vertices)
{
    std::vector<int> edge_list;
    std::vector<int> vertex_list;
    get_vertices_and_edges(
        face_indices, mesh,
        edge_list, vertex_list);
    return simplified_smoothed_offset_potential_point_impl<double>(
        q, face_indices,
        edge_list, vertex_list,
        mesh,
        params,
        edge_valence,
        vertex_internal,
        include_faces, include_edges, include_vertices);
}

Eigen::VectorXd simplified_smoothed_offset_potential(
    Eigen::ConstRef<Eigen::MatrixXd> q,
    const PotentialCollisionMesh& mesh,
    const PotentialParameters& params,
    bool include_faces, bool include_edges, bool include_vertices)
{
    if (q.cols() != 3) {
        throw std::runtime_error("q must have shape (nq, 3).");
    }

    std::vector<int> face_indices(static_cast<size_t>(mesh.num_faces()));
    for (int i = 0; i < mesh.faces().rows(); i++) {
        face_indices[static_cast<size_t>(i)] = i;
    }
    std::vector<int> edge_list;
    std::vector<int> vertex_list;
    get_vertices_and_edges(
        face_indices, mesh,
        edge_list, vertex_list);

    const std::vector<int> edge_valence = build_edge_valence(mesh);
    const std::vector<char> vertex_internal = build_vertex_internal(mesh);

    Eigen::VectorXd out(q.rows());
    for (int i = 0; i < q.rows(); i++) {
        const Eigen::Vector3d qi = q.row(i).transpose();
        out[i] = simplified_smoothed_offset_potential_point_impl<double>(
            qi, face_indices,
            edge_list, vertex_list,
            mesh,
            params,
            edge_valence,
            vertex_internal,
            include_faces, include_edges, include_vertices);
    }

    return out;
}

Eigen::VectorXd simplified_smoothed_offset_potential_cpp(
    Eigen::ConstRef<Eigen::MatrixXd> q,
    Eigen::ConstRef<Eigen::MatrixXd> V,
    Eigen::ConstRef<Eigen::MatrixXi> F,
    double alpha, double p, double epsilon,
    bool include_faces, bool include_edges, bool include_vertices,
    bool localized, bool one_sided)
{
    PotentialCollisionMesh mesh(V, F);
    const PotentialParameters params{
        alpha,
        p,
        epsilon,
        localized,
        one_sided,
    };
    return simplified_smoothed_offset_potential(
        q,
        mesh,
        params,
        include_faces, include_edges, include_vertices);
}

double simplified_potential_face(
    const Eigen::Vector3d& q,
    const Eigen::Matrix<double, 3, 3>& face_points,
    const PotentialParameters& params)
{
    const FacePoints face = face_points_from_rows(face_points);
    return simplified_potential_face<double>(q, face, params);
}

double simplified_potential_edge(
    const Eigen::Vector3d& q,
    const Eigen::Matrix<double, 2, 3>& edge_points,
    const PotentialParameters& params)
{
    const Vector3<double> p0 = edge_points.row(0).transpose();
    const Vector3<double> p1 = edge_points.row(1).transpose();
    return simplified_potential_edge<double>(q, p0, p1, params);
}

double simplified_potential_vertex(
    const Eigen::Vector3d& q, const Eigen::Vector3d& p_v,
    const PotentialParameters& params)
{
    return simplified_potential_vertex<double>(q, p_v, params);
}

} // namespace ipc

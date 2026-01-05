#pragma once

#include "potential_collision_mesh.hpp"
#include "potential_parameters.hpp"

#include <Eigen/Dense>
#include <utility>

namespace ipc {

Eigen::VectorXd simplified_smoothed_offset_potential(
    Eigen::ConstRef<Eigen::MatrixXd> q,
    const PotentialCollisionMesh& mesh,
    const PotentialParameters& params,
    bool include_faces, bool include_edges, bool include_vertices);

Eigen::VectorXd simplified_smoothed_offset_potential_accelerated(
    Eigen::ConstRef<Eigen::MatrixXd> q,
    const PotentialCollisionMesh& mesh,
    const PotentialParameters& params,
    bool include_faces, bool include_edges, bool include_vertices);

Eigen::VectorXd simplified_smoothed_offset_potential_cpp(
    Eigen::ConstRef<Eigen::MatrixXd> q,
    Eigen::ConstRef<Eigen::MatrixXd> V,
    Eigen::ConstRef<Eigen::MatrixXi> F,
    double alpha, double p, double epsilon,
    bool include_faces, bool include_edges, bool include_vertices,
    bool localized, bool one_sided);

Eigen::VectorXd simplified_smoothed_offset_potential_accelerated_cpp(
    Eigen::ConstRef<Eigen::MatrixXd> q,
    Eigen::ConstRef<Eigen::MatrixXd> V,
    Eigen::ConstRef<Eigen::MatrixXi> F,
    double alpha, double p, double epsilon,
    bool include_faces, bool include_edges, bool include_vertices,
    bool localized, bool one_sided);

std::pair<double, Eigen::Vector3d> simplified_smoothed_offset_potential_cpp_tinyad(
    const Eigen::Vector3d& q,
    const PotentialCollisionMesh& mesh,
    const PotentialParameters& params,
    bool include_faces, bool include_edges, bool include_vertices);

std::pair<double, Eigen::Vector3d> simplified_smoothed_offset_potential_cpp_tinyad(
    const Eigen::Vector3d& q,
    Eigen::ConstRef<Eigen::MatrixXd> V,
    Eigen::ConstRef<Eigen::MatrixXi> F,
    double alpha, double p, double epsilon,
    bool include_faces, bool include_edges, bool include_vertices,
    bool localized, bool one_sided);

double simplified_potential_face(
    const Eigen::Vector3d& q,
    const Eigen::Matrix<double, 3, 3>& face_points,
    const PotentialParameters& params);

double simplified_potential_edge(
    const Eigen::Vector3d& q,
    const Eigen::Matrix<double, 2, 3>& edge_points,
    const PotentialParameters& params);

double simplified_potential_vertex(
    const Eigen::Vector3d& q, const Eigen::Vector3d& p_v,
    const PotentialParameters& params);

std::pair<double, Eigen::Vector3d> simplified_potential_face_cpp_tinyad(
    const Eigen::Vector3d& q,
    const Eigen::Matrix<double, 3, 3>& face_points,
    const PotentialParameters& params);

std::pair<double, Eigen::Vector3d> simplified_potential_edge_cpp_tinyad(
    const Eigen::Vector3d& q,
    const Eigen::Matrix<double, 2, 3>& edge_points,
    const PotentialParameters& params);

std::pair<double, Eigen::Vector3d> simplified_potential_vertex_cpp_tinyad(
    const Eigen::Vector3d& q, const Eigen::Vector3d& p_v,
    const PotentialParameters& params);

// Gradients/Hessians are w.r.t. q and element points in xyzxyz... order.
void simplified_potential_face_grad_hess(
    const Eigen::Vector3d& q,
    const Eigen::Matrix<double, 3, 3>& face_points,
    const PotentialParameters& params,
    Eigen::VectorXd& grad,
    Eigen::MatrixXd& hess);

void simplified_potential_edge_grad_hess(
    const Eigen::Vector3d& q,
    const Eigen::Matrix<double, 2, 3>& edge_points,
    const PotentialParameters& params,
    Eigen::VectorXd& grad,
    Eigen::MatrixXd& hess);

void simplified_potential_vertex_grad_hess(
    const Eigen::Vector3d& q, const Eigen::Vector3d& p_v,
    const PotentialParameters& params,
    Eigen::VectorXd& grad,
    Eigen::MatrixXd& hess);

} // namespace ipc

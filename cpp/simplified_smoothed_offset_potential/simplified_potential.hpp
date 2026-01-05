#pragma once

#include "potential_collision_mesh.hpp"
#include "potential_parameters.hpp"

#include <Eigen/Dense>
#include <utility>
#include <vector>

namespace ipc {

// Compute potential from a mesh at points in the array q.
// mesh:  connectivity and vertex positions
// params: potential parameters, see potential_description.tex
// include_faces, include_edges, include_vertices: turn on/off the components of the potential.

Eigen::VectorXd simplified_smoothed_offset_potential(
    Eigen::ConstRef<Eigen::MatrixXd> q,
    const PotentialCollisionMesh& mesh,
    const PotentialParameters& params,
    bool include_faces, bool include_edges, bool include_vertices);

#if defined(IPC_HAS_VTK)
/// accelerated versions of the same using VTK
Eigen::VectorXd simplified_smoothed_offset_potential_accelerated(
    Eigen::ConstRef<Eigen::MatrixXd> q,
    const PotentialCollisionMesh& mesh,
    const PotentialParameters& params,
    bool include_faces, bool include_edges, bool include_vertices);

Eigen::VectorXd simplified_smoothed_offset_potential_accelerated_cpp(
    Eigen::ConstRef<Eigen::MatrixXd> q,
    Eigen::ConstRef<Eigen::MatrixXd> V,
    Eigen::ConstRef<Eigen::MatrixXi> F,
    double alpha, double p, double epsilon,
    bool include_faces, bool include_edges, bool include_vertices,
    bool localized, bool one_sided);
#endif

// Compute the potential at a single point, the other arguments are the same as above
double simplified_smoothed_offset_potential_point(
    const Eigen::Vector3d& q, const std::vector<int>& face_indices,
    const PotentialCollisionMesh& mesh,
    const PotentialParameters& params,
    const std::vector<int>& edge_valence,
    const std::vector<char>& vertex_internal,
    bool include_faces, bool include_edges, bool include_vertices);

// This version is for compatibility with python interface visualization in particular 
// builds PotentialCollisionMesh from (V,F)
// potential parameters not packed into a structure for binding simplicity
// alpha is not used by this potential, one-sided not implemented yet
Eigen::VectorXd simplified_smoothed_offset_potential_cpp(
    Eigen::ConstRef<Eigen::MatrixXd> q,
    Eigen::ConstRef<Eigen::MatrixXd> V,
    Eigen::ConstRef<Eigen::MatrixXi> F,
    double alpha, double p, double epsilon,
    bool include_faces, bool include_edges, bool include_vertices,
    bool localized, bool one_sided);


//  point/face, point/edge, point/vertex

// face_points: 3 vertex positions
double simplified_potential_face(
    const Eigen::Vector3d& q,
    const Eigen::Matrix<double, 3, 3>& face_points,
    const PotentialParameters& params);

// edge_points: 2 edge endpoints
double simplified_potential_edge(
    const Eigen::Vector3d& q,
    const Eigen::Matrix<double, 2, 3>& edge_points,
    const PotentialParameters& params);

// p_v: vertex position
double simplified_potential_vertex(
    const Eigen::Vector3d& q, const Eigen::Vector3d& p_v,
    const PotentialParameters& params);

// Same potential evaluation but with gradient and Hessian computed

// grad/hessian order:  4 * 3, xyzxyz... order corresponds to q, face_points[0].. face_points[2]
// Gradients/Hessians are w.r.t. q and element points in xyzxyz... order.
void simplified_potential_face_grad_hess(
    const Eigen::Vector3d& q,
    const Eigen::Matrix<double, 3, 3>& face_points,
    const PotentialParameters& params,
    Eigen::VectorXd& grad,
    Eigen::MatrixXd& hess);

// grad/hessian order:  3 * 3 xyzxyz... order corresponds to q, edge_points[0]..edge_points[1]
void simplified_potential_edge_grad_hess(
    const Eigen::Vector3d& q,
    const Eigen::Matrix<double, 2, 3>& edge_points,
    const PotentialParameters& params,
    Eigen::VectorXd& grad,
    Eigen::MatrixXd& hess);

// grad/hessian order:  2 * 3 xyzxyz... order corresponds to q, p_v
void simplified_potential_vertex_grad_hess(
    const Eigen::Vector3d& q, const Eigen::Vector3d& p_v,
    const PotentialParameters& params,
    Eigen::VectorXd& grad,
    Eigen::MatrixXd& hess);

// TinyAD helpers (tests only).
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
} // namespace ipc

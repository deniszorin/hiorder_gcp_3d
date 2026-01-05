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

Eigen::VectorXd smoothed_offset_potential(
    Eigen::ConstRef<Eigen::MatrixXd> q,
    const PotentialCollisionMesh& mesh,
    const PotentialParameters& params,
    bool include_faces, bool include_edges, bool include_vertices);

Eigen::VectorXd smoothed_offset_potential_accelerated(
    Eigen::ConstRef<Eigen::MatrixXd> q,
    const PotentialCollisionMesh& mesh,
    const PotentialParameters& params,
    bool include_faces, bool include_edges, bool include_vertices);

// This version is for compatibility with python interface plotting in particular 
// builds PotentialCollisionMesh from (V,F)
// consider packing parameters into json
Eigen::VectorXd smoothed_offset_potential_cpp(
    Eigen::ConstRef<Eigen::MatrixXd> q,
    Eigen::ConstRef<Eigen::MatrixXd> V,
    Eigen::ConstRef<Eigen::MatrixXi> F,
    double alpha, double p, double epsilon,
    bool include_faces, bool include_edges, bool include_vertices,
    bool localized, bool one_sided);

Eigen::VectorXd smoothed_offset_potential_accelerated_cpp(
    Eigen::ConstRef<Eigen::MatrixXd> q,
    Eigen::ConstRef<Eigen::MatrixXd> V,
    Eigen::ConstRef<Eigen::MatrixXi> F,
    double alpha, double p, double epsilon,
    bool include_faces, bool include_edges, bool include_vertices,
    bool localized, bool one_sided);

//  point/face, point/edge, point/vertex, just potential 

// face_points: 3 vertex positions
double potential_face(
    const Eigen::Vector3d& q,
    const Eigen::Matrix<double, 3, 3>& face_points,
    const PotentialParameters& params);

// edge_points: 4 in the order  two edge endpoints, opposite point in face 0, opposite point 
// in face 1 (if it exists as indicated by has_f1
double potential_edge(
    const Eigen::Vector3d& q,
    const Eigen::Matrix<double, 4, 3>& edge_points,
    bool has_f1,
    const PotentialParameters& params);

// p_v: vertex position
// neightbor_points: list of incident vertices in ccw order, starting from a boundary vertex, 
// of  v is boundary 
// pointed_vertex: array of flags whether each vertex of the mesh is "pointed", i.e., its polar cone
// is on the outside of the mesh.  Needed for outside_vertex function.  

double potential_vertex(
    const Eigen::Vector3d& q, const Eigen::Vector3d& p_v,
    Eigen::ConstRef<Eigen::MatrixXd> neighbor_points, bool is_boundary,
    bool pointed_vertex,
    const PotentialParameters& params);

// Same potential evaluation but with gradient and Hessian computed

// grad/hessian order:  4 * 3, xyzxyz... order corresponds to q, face_points[0].. face_points[2]
void potential_face_grad_hess(
    const Eigen::Vector3d& q,
    const Eigen::Matrix<double, 3, 3>& face_points,
    const PotentialParameters& params,
    Eigen::VectorXd& grad,
    Eigen::MatrixXd& hess);

// grad/hessian order:  5 * 3 xyzxyz... order corresponds to  q, edge_points[0]..edge_points[3]
void potential_edge_grad_hess(
    const Eigen::Vector3d& q,
    const Eigen::Matrix<double, 4, 3>& edge_points,
    bool has_f1,
    const PotentialParameters& params,
    Eigen::VectorXd& grad,
    Eigen::MatrixXd& hess);

// grad/hessian order:  5 * 3 xyzxyz... order corresponds to  q, edge_points[0]..edge_points[3]
void potential_vertex_grad_hess(
    const Eigen::Vector3d& q, const Eigen::Vector3d& p_v,
    Eigen::ConstRef<Eigen::MatrixXd> neighbor_points, bool is_boundary,
    bool pointed_vertex,
    const PotentialParameters& params,
    Eigen::VectorXd& grad,
    Eigen::MatrixXd& hess);



// TinyAD helpers (tests only).
std::pair<double, Eigen::Vector3d> potential_face_cpp_tinyad(
    const Eigen::Vector3d& q,
    const Eigen::Matrix<double, 3, 3>& face_points,
    const PotentialParameters& params);

std::pair<double, Eigen::Vector3d> potential_edge_cpp_tinyad(
    const Eigen::Vector3d& q,
    const Eigen::Matrix<double, 4, 3>& edge_points,
    bool has_f1,
    const PotentialParameters& params);

std::pair<double, Eigen::Vector3d> potential_vertex_cpp_tinyad(
    const Eigen::Vector3d& q, const Eigen::Vector3d& p_v,
    Eigen::ConstRef<Eigen::MatrixXd> neighbor_points, bool is_boundary,
    bool pointed_vertex,
    const PotentialParameters& params);

} // namespace ipc

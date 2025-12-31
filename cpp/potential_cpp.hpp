#pragma once

#include "potential_collision_mesh.hpp"

#include <Eigen/Dense>
#include <vector>

namespace ipc {

// Compute potential from a mesh at points in the array q.
// mesh: connectivity and mesh geometry, see geometry.py
// alpha, p, epsilon: potential parameters, see potential_description.tex
// include_faces, include_edges, include_vertices: turn on/off the components of the potential.
// localized: enable h^epsilon factor.
// one_sided: enable localized sidedness checks in the potential.
Eigen::VectorXd smoothed_offset_potential(
    Eigen::ConstRef<Eigen::MatrixXd> q,
    const PotentialCollisionMesh& mesh,
    double alpha, double p, double epsilon,
    bool include_faces, bool include_edges, bool include_vertices,
    bool localized, bool one_sided);

Eigen::VectorXd smoothed_offset_potential_cpp(
    Eigen::ConstRef<Eigen::MatrixXd> q,
    Eigen::ConstRef<Eigen::MatrixXd> V,
    Eigen::ConstRef<Eigen::MatrixXi> F,
    double alpha, double p, double epsilon,
    bool include_faces, bool include_edges, bool include_vertices,
    bool localized, bool one_sided);

} // namespace ipc

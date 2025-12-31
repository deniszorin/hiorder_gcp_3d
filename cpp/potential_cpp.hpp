#pragma once

#include "potential_collision_mesh.hpp"

#include <Eigen/Dense>
#include <vector>

namespace ipc {

Eigen::VectorXd smoothed_offset_potential(
    Eigen::ConstRef<Eigen::MatrixXd> q,
    const PotentialCollisionMesh& mesh,
    double alpha,
    double p,
    double epsilon,
    bool include_faces,
    bool include_edges,
    bool include_vertices,
    bool localized,
    bool one_sided);

Eigen::VectorXd smoothed_offset_potential_cpp(
    Eigen::ConstRef<Eigen::MatrixXd> q,
    Eigen::ConstRef<Eigen::MatrixXd> V,
    Eigen::ConstRef<Eigen::MatrixXi> F,
    double alpha,
    double p,
    double epsilon,
    bool include_faces,
    bool include_edges,
    bool include_vertices,
    bool localized,
    bool one_sided);

} // namespace ipc

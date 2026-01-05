#pragma once

#include <Eigen/Dense>

#include <vector>

namespace ipc {

struct ConeHullResult {
    std::vector<int> indices;
    bool coplanar;
    bool fullspace;
};

ConeHullResult cone_convex_hull(
    const std::vector<Eigen::Vector3d>& e,
    double eps = 1e-12);

bool validate_cone_convex_hull(
    const std::vector<Eigen::Vector3d>& e,
    const std::vector<int>& indices,
    double eps = 1e-12);

bool pointed_vertex(
    const std::vector<Eigen::Vector3d>& e,
    double eps = 1e-12);

} // namespace ipc

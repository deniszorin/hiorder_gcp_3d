#include "cone_convex_hull.hpp"
#include "predicates_c.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <optional>
#include <stdexcept>

namespace ipc {

namespace {

void ensure_exactinit()
{
    static const bool initialized = []() {
        exactinit();
        return true;
    }();
    (void)initialized;
}

double orient3d_origin(
    const Eigen::Vector3d& a,
    const Eigen::Vector3d& b,
    const Eigen::Vector3d& c)
{
    ensure_exactinit();
    double o[3] = { 0.0, 0.0, 0.0 };
    double aa[3] = { a.x(), a.y(), a.z() };
    double bb[3] = { b.x(), b.y(), b.z() };
    double cc[3] = { c.x(), c.y(), c.z() };
    return orient3d(o, aa, bb, cc);
}

std::optional<int> find_start_index(
    const std::vector<Eigen::Vector3d>& e)
{
    const int n = static_cast<int>(e.size());
    for (int i = 0; i < n - 2; i++) {
        if (orient3d_origin(e[static_cast<size_t>(i)],
                e[static_cast<size_t>(i + 1)],
                e[static_cast<size_t>(i + 2)])
            != 0.0) {
            return i;
        }
    }
    return std::nullopt;
}

bool are_opposite_collinear(
    const Eigen::Vector3d& a,
    const Eigen::Vector3d& b)
{
    const Eigen::Vector3d cross = a.cross(b);
    if (cross.x() == 0.0 && cross.y() == 0.0 && cross.z() == 0.0
        && a.dot(b) < 0.0) {
        return true;
    }
    const double norm_prod = a.norm() * b.norm();
    return a.dot(b) == -norm_prod;
}

bool would_create_opposite_pair_tail(
    const std::vector<Eigen::Vector3d>& e_rot,
    const std::deque<int>& deque_indices,
    const int new_idx)
{
    if (deque_indices.size() < 2) {
        return false;
    }
    const int prev_idx = deque_indices[deque_indices.size() - 2];
    return are_opposite_collinear(
        e_rot[static_cast<size_t>(prev_idx)],
        e_rot[static_cast<size_t>(new_idx)]);
}

bool would_create_opposite_pair_head(
    const std::vector<Eigen::Vector3d>& e_rot,
    const std::deque<int>& deque_indices,
    const int new_idx)
{
    if (deque_indices.size() < 2) {
        return false;
    }
    const int next_idx = deque_indices[1];
    return are_opposite_collinear(
        e_rot[static_cast<size_t>(new_idx)],
        e_rot[static_cast<size_t>(next_idx)]);
}

} // namespace

// This is an adaptaton of Melkman's algorithm for polygon convex hull to 
// 3D polyhedral cone convex hull.  The algorithm is largely the same, 

ConeHullResult cone_convex_hull(
    const std::vector<Eigen::Vector3d>& e,
    double /*eps*/)
{
    const int n = static_cast<int>(e.size());
    if (n < 3) {
        throw std::runtime_error("Need at least 3 edge directions.");
    }

    const auto start = find_start_index(e);
    if (!start.has_value()) {
        std::vector<int> indices;
        indices.reserve(static_cast<size_t>(n));
        for (int i = 0; i < n; i++) {
            indices.push_back(i);
        }
        const bool fullspace = indices.size() < 3;
        return ConeHullResult{ indices, true, fullspace };
    }

    std::vector<int> order;
    order.reserve(static_cast<size_t>(n));
    for (int i = start.value(); i < n; i++) {
        order.push_back(i);
    }
    for (int i = 0; i < start.value(); i++) {
        order.push_back(i);
    }

    std::vector<Eigen::Vector3d> e_rot;
    e_rot.reserve(static_cast<size_t>(n));
    for (const int idx : order) {
        e_rot.push_back(e[static_cast<size_t>(idx)]);
    }

    std::deque<int> D;
    const double o = orient3d_origin(e_rot[0], e_rot[1], e_rot[2]);
    if (o > 0.0) {
        D.push_front(0);
        D.push_front(2);
        D.push_back(1);
        D.push_back(2);
    } else {
        D.push_front(1);
        D.push_front(2);
        D.push_back(0);
        D.push_back(2);
    }

    for (int i = 3; i < n; i++) {
        if (orient3d_origin(
                e_rot[static_cast<size_t>(D[D.size() - 2])],
                e_rot[static_cast<size_t>(D[D.size() - 1])],
                e_rot[static_cast<size_t>(i)])
                > 0.0
            && orient3d_origin(
                   e_rot[static_cast<size_t>(i)],
                   e_rot[static_cast<size_t>(D[0])],
                   e_rot[static_cast<size_t>(D[1])])
                > 0.0) {
            continue;
        }

        while (D.size() > 1
            && orient3d_origin(
                   e_rot[static_cast<size_t>(D[D.size() - 2])],
                   e_rot[static_cast<size_t>(D[D.size() - 1])],
                   e_rot[static_cast<size_t>(i)])
                < 0.0) {
            if (would_create_opposite_pair_tail(e_rot, D, i)) {
                break;
            }
            D.pop_back();
        }
        D.push_back(i);

        while (D.size() > 1
            && orient3d_origin(
                   e_rot[static_cast<size_t>(i)],
                   e_rot[static_cast<size_t>(D[0])],
                   e_rot[static_cast<size_t>(D[1])])
                < 0.0) {
            if (would_create_opposite_pair_head(e_rot, D, i)) {
                break;
            }
            D.pop_front();
        }
        D.push_front(i);
    }

    if (D.size() > 1 && D.back() == D.front()) {
        D.pop_back();
    }

    std::vector<int> indices;
    indices.reserve(D.size());
    for (const int idx : D) {
        indices.push_back(order[static_cast<size_t>(idx)]);
    }
    if (!indices.empty()) {
        std::vector<char> hull_set(static_cast<size_t>(n), 0);
        for (const int idx : indices) {
            if (idx >= 0 && idx < n) {
                hull_set[static_cast<size_t>(idx)] = 1;
            }
        }
        std::vector<int> filtered;
        filtered.reserve(indices.size());
        for (int i = 0; i < n; i++) {
            if (hull_set[static_cast<size_t>(i)]) {
                filtered.push_back(i);
            }
        }
        indices = filtered;
    }

    const bool fullspace = indices.size() < 3;
    return ConeHullResult{ indices, false, fullspace };
}

bool validate_cone_convex_hull(
    const std::vector<Eigen::Vector3d>& e,
    const std::vector<int>& indices,
    double eps)
{
    if (indices.empty()) {
        return false;
    }
    const int m = static_cast<int>(indices.size());
    for (int j = 0; j < m; j++) {
        const Eigen::Vector3d a = e[static_cast<size_t>(indices[static_cast<size_t>(j)])];
        const Eigen::Vector3d b = e[static_cast<size_t>(indices[static_cast<size_t>((j + 1) % m)])];
        const Eigen::Vector3d n = a.cross(b);
        for (const auto& vec : e) {
            if (vec.dot(n) > eps) {
                return false;
            }
        }
    }
    return true;
}

/// Determine if the cone formed by the given edge directions with given outside normals is pointed, i.e., has a convex hull 
/// that is not a full space, containing it in its interior, that is not a full space.   

bool pointed_vertex(
    const std::vector<Eigen::Vector3d>& e,
    double eps)
{
    if (e.size() < 3) {
        return false;
    }

    const ConeHullResult result = cone_convex_hull(e, eps);
    if (result.coplanar || result.fullspace) {
        return false;
    }

    const int m = static_cast<int>(result.indices.size());
    if (m < 3) {
        return false;
    }

    for (int i = 0; i < m; i++) {
        const Eigen::Vector3d a = e[static_cast<size_t>(result.indices[static_cast<size_t>(i)])];
        const Eigen::Vector3d b = e[static_cast<size_t>(result.indices[static_cast<size_t>((i + 1) % m)])];
        const Eigen::Vector3d n = a.cross(b);
        if (n.norm() <= eps) {
            continue;
        }
        for (int j = 0; j < m; j++) {
            if (j == i || j == (i + 1) % m) {
                continue;
            }
            const Eigen::Vector3d c = e[static_cast<size_t>(result.indices[static_cast<size_t>(j)])];
            const double dot = n.dot(c);
            if (std::abs(dot) > eps) {
                return dot < 0.0;
            }
        }
    }
    return false;
}

} // namespace ipc

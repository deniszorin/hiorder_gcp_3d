#include "potential_collision_mesh.hpp"

#include "cone_convex_hull.hpp"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace ipc {

namespace {

constexpr double kEps = 1e-12;

struct PairHash {
    size_t operator()(const std::pair<int, int>& key) const
    {
        return std::hash<int>()(key.first) ^ (std::hash<int>()(key.second) << 1);
    }
};

std::optional<std::vector<int>> ordered_vertex_neighbors(
    const PotentialCollisionMesh& mesh,
    const int v_idx)
{
    std::unordered_map<int, int> next_map;
    std::unordered_map<int, int> prev_map;
    std::vector<int> next_order;
    std::unordered_set<int> neighbors;

    const auto& face_list = mesh.vertices_to_faces()[static_cast<size_t>(v_idx)];
    for (const int f : face_list) {
        int loc = -1;
        for (int i = 0; i < 3; i++) {
            if (mesh.faces()(f, i) == v_idx) {
                loc = i;
                break;
            }
        }
        if (loc < 0) {
            continue;
        }
        const int neighbor_after = mesh.faces()(f, (loc + 1) % 3);
        const int neighbor_before = mesh.faces()(f, (loc + 2) % 3);
        neighbors.insert(neighbor_after);
        neighbors.insert(neighbor_before);

        const auto it_next = next_map.find(neighbor_after);
        if (it_next != next_map.end()) {
            if (it_next->second != neighbor_before) {
                return std::nullopt;
            }
        } else {
            next_map[neighbor_after] = neighbor_before;
            next_order.push_back(neighbor_after);
        }

        const auto it_prev = prev_map.find(neighbor_before);
        if (it_prev != prev_map.end()) {
            if (it_prev->second != neighbor_after) {
                return std::nullopt;
            }
        } else {
            prev_map[neighbor_before] = neighbor_after;
        }
    }

    if (next_map.empty()) {
        return std::nullopt;
    }

    int start = -1;
    for (const int neighbor : next_order) {
        if (prev_map.find(neighbor) == prev_map.end()) {
            start = neighbor;
            break;
        }
    }
    if (start < 0) {
        start = next_order.front();
        for (const int neighbor : next_order) {
            if (neighbor < start) {
                start = neighbor;
            }
        }
    }

    std::vector<int> order;
    order.reserve(neighbors.size());
    std::unordered_set<int> visited;
    int current = start;
    while (current >= 0 && visited.find(current) == visited.end()) {
        visited.insert(current);
        order.push_back(current);
        const auto it = next_map.find(current);
        if (it == next_map.end()) {
            current = -1;
            break;
        }
        current = it->second;
        if (current == start) {
            break;
        }
    }

    if (order.size() != neighbors.size()) {
        return std::nullopt;
    }
    return order;
}

} // namespace

PotentialCollisionMesh::PotentialCollisionMesh(
    Eigen::ConstRef<Eigen::MatrixXd> rest_positions,
    Eigen::ConstRef<Eigen::MatrixXi> faces)
    : CollisionMesh(rest_positions, faces)
{
    reorder_vertices_to_edges();
    compute_face_geometry();
    compute_edge_normals();
    compute_pointed_vertices();
}

void PotentialCollisionMesh::compute_face_geometry()
{
    const Eigen::MatrixXd& V = rest_positions();
    const Eigen::MatrixXi& F = faces();
    const int nf = F.rows();

    m_normals.resize(nf, 3);
    m_edge_inward.resize(static_cast<size_t>(nf));

    for (int f = 0; f < nf; f++) {
        const int v0 = F(f, 0);
        const int v1 = F(f, 1);
        const int v2 = F(f, 2);

        const Eigen::Vector3d p0 = V.row(v0);
        const Eigen::Vector3d p1 = V.row(v1);
        const Eigen::Vector3d p2 = V.row(v2);

        const Eigen::Vector3d e0 = p1 - p0;
        const Eigen::Vector3d e1 = p2 - p1;
        const Eigen::Vector3d e2 = p0 - p2;

        Eigen::Vector3d n = (p1 - p0).cross(p2 - p0);
        const double n_norm = n.norm();
        if (n_norm <= kEps) {
            throw std::runtime_error("Degenerate face normal.");
        }
        n /= n_norm;
        m_normals.row(f) = n;

        const Eigen::Vector3d edges[3] = { e0, e1, e2 };
        for (int i = 0; i < 3; i++) {
            const double e_norm = edges[i].norm();
            if (e_norm <= kEps) {
                throw std::runtime_error("Degenerate face edge.");
            }
            const Eigen::Vector3d d_e = edges[i] / e_norm;
            m_edge_inward[static_cast<size_t>(f)][static_cast<size_t>(i)] = n.cross(d_e);
        }
    }
}

void PotentialCollisionMesh::compute_edge_normals()
{
    const Eigen::MatrixXi& edge_faces = edges_to_faces();
    const int ne = edge_faces.rows();

    m_edge_normals.resize(ne, 3);
    for (int e = 0; e < ne; e++) {
        Eigen::Vector3d n_sum(0.0, 0.0, 0.0);
        for (int k = 0; k < edge_faces.cols(); k++) {
            const int f = edge_faces(e, k);
            if (f >= 0) {
                n_sum += m_normals.row(f);
            }
        }
        const double n_norm = n_sum.norm();
        if (n_norm > kEps) {
            n_sum /= n_norm;
        }
        m_edge_normals.row(e) = n_sum;
    }
}

void PotentialCollisionMesh::compute_pointed_vertices()
{
    const Eigen::MatrixXd& V = rest_positions();
    const int nv = static_cast<int>(num_vertices());
    m_pointed_vertices.assign(static_cast<size_t>(nv), 0);

    for (int v = 0; v < nv; v++) {
        std::optional<std::vector<int>> neighbors = ordered_vertex_neighbors(*this, v);
        if (!neighbors.has_value()) {
            throw std::runtime_error(
                "Failed to order vertex neighbors at vertex "
                + std::to_string(v) + ".");
        }
        std::vector<Eigen::Vector3d> vectors;
        vectors.reserve(neighbors->size());
        for (const int other : neighbors.value()) {
            Eigen::Vector3d vec = V.row(other) - V.row(v);
            const double vec_norm = vec.norm();
            if (vec_norm <= kEps) {
                continue;
            }
            vectors.push_back(vec / vec_norm);
        }
        if (pointed_vertex(vectors)) {
            m_pointed_vertices[static_cast<size_t>(v)] = 1;
        }
    }
}

void PotentialCollisionMesh::reorder_vertices_to_edges()
{
    const Eigen::MatrixXi& edges = this->edges();
    const Eigen::MatrixXi& edge_faces = edges_to_faces();

    std::unordered_map<std::pair<int, int>, int, PairHash> edge_index;
    edge_index.reserve(static_cast<size_t>(edges.rows()));
    for (int e = 0; e < edges.rows(); e++) {
        const int a = edges(e, 0);
        const int b = edges(e, 1);
        const int lo = std::min(a, b);
        const int hi = std::max(a, b);
        edge_index[std::make_pair(lo, hi)] = e;
    }

    std::vector<bool> boundary_edge(static_cast<size_t>(edges.rows()), false);
    for (int e = 0; e < edges.rows(); e++) {
        boundary_edge[static_cast<size_t>(e)] =
            edge_faces(e, 0) < 0 || edge_faces(e, 1) < 0;
    }

    auto& vertices_to_edges = this->vertices_to_edges();
    for (int v = 0; v < static_cast<int>(num_vertices()); v++) {
        std::optional<std::vector<int>> neighbors = ordered_vertex_neighbors(*this, v);
        if (!neighbors.has_value()) {
            throw std::runtime_error(
                "Failed to order vertex neighbors at vertex "
                + std::to_string(v) + ".");
        }
        std::vector<int> ordered_edges;
        ordered_edges.reserve(neighbors->size());
        for (const int neighbor : neighbors.value()) {
            const int lo = std::min(v, neighbor);
            const int hi = std::max(v, neighbor);
            auto it = edge_index.find(std::make_pair(lo, hi));
            if (it == edge_index.end()) {
                throw std::runtime_error("Missing edge for ordered vertex neighbor.");
            }
            ordered_edges.push_back(it->second);
        }

        std::vector<int> boundary_positions;
        boundary_positions.reserve(2);
        for (int i = 0; i < static_cast<int>(ordered_edges.size()); i++) {
            const int edge_idx = ordered_edges[i];
            if (boundary_edge[static_cast<size_t>(edge_idx)]) {
                boundary_positions.push_back(i);
            }
        }
        if (boundary_positions.size() == 2) {
            const bool ok = boundary_positions.front() == 0
                && boundary_positions.back() == static_cast<int>(ordered_edges.size()) - 1;
            if (!ok) {
                throw std::runtime_error(
                    "Boundary edge order mismatch at vertex "
                    + std::to_string(v) + ".");
            }
        }

        vertices_to_edges[static_cast<size_t>(v)] = ordered_edges;
    }
}

void PotentialCollisionMesh::set_pointed_vertices(
    const std::vector<char>& pointed_vertices)
{
    if (pointed_vertices.size() != num_vertices()) {
        throw std::runtime_error("pointed_vertices size mismatch.");
    }
    m_pointed_vertices = pointed_vertices;
}

} // namespace ipc

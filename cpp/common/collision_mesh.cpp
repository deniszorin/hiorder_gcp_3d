#include "collision_mesh.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace ipc {

namespace {

struct PairHash {
    size_t operator()(const std::pair<int, int>& key) const
    {
        return std::hash<int>()(key.first) ^ (std::hash<int>()(key.second) << 1);
    }
};

void build_edges_from_faces(
    Eigen::ConstRef<Eigen::MatrixXi> faces,
    Eigen::MatrixXi& edges,
    Eigen::MatrixXi& faces_to_edges)
{
    if (faces.size() == 0) {
        edges.resize(0, 2);
        faces_to_edges.resize(faces.rows(), faces.cols());
        return;
    }
    if (faces.cols() != 3) {
        throw std::runtime_error("Faces must be triangles (|F| x 3).");
    }

    std::unordered_map<std::pair<int, int>, int, PairHash> edge_map;
    std::vector<std::pair<int, int>> edge_list;
    edge_map.reserve(static_cast<size_t>(faces.rows()) * 2);

    faces_to_edges.resize(faces.rows(), faces.cols());
    for (int fi = 0; fi < faces.rows(); fi++) {
        for (int fj = 0; fj < faces.cols(); fj++) {
            const int vi = faces(fi, fj);
            const int vj = faces(fi, (fj + 1) % faces.cols());
            const int a = std::min(vi, vj);
            const int b = std::max(vi, vj);
            auto it = edge_map.find(std::make_pair(a, b));
            if (it == edge_map.end()) {
                const int edge_id = static_cast<int>(edge_list.size());
                edge_list.emplace_back(a, b);
                edge_map.emplace(std::make_pair(a, b), edge_id);
                faces_to_edges(fi, fj) = edge_id;
            } else {
                faces_to_edges(fi, fj) = it->second;
            }
        }
    }

    edges.resize(static_cast<int>(edge_list.size()), 2);
    for (int i = 0; i < edges.rows(); i++) {
        edges(i, 0) = edge_list[i].first;
        edges(i, 1) = edge_list[i].second;
    }
}

} // namespace

CollisionMesh::CollisionMesh(
    Eigen::ConstRef<Eigen::MatrixXd> rest_positions,
    Eigen::ConstRef<Eigen::MatrixXi> faces)
    : m_rest_positions(rest_positions)
    , m_faces(faces)
{
    for (int i = 0; i < m_faces.rows(); i++) {
        for (int j = 0; j < m_faces.cols(); j++) {
            const int vi = m_faces(i, j);
            if (vi < 0 || vi >= m_rest_positions.rows()) {
                throw std::runtime_error("Face index out of bounds.");
            }
        }
    }

    build_edges_from_faces(m_faces, m_edges, m_faces_to_edges);
    init_edges_to_faces();
    init_vertices_to_edges();
    init_vertices_to_faces();
    init_boundary();

    // Ensure vertex stars are connected (no disjoint face fans).
    for (size_t v = 0; v < num_vertices(); v++) {
        const auto& incident_faces = m_vertices_to_faces[v];
        if (incident_faces.size() <= 1) {
            continue;
        }
        std::unordered_map<int, int> face_to_local;
        face_to_local.reserve(incident_faces.size());
        for (int i = 0; i < static_cast<int>(incident_faces.size()); i++) {
            face_to_local[incident_faces[i]] = i;
        }

        std::vector<std::vector<int>> adjacency(incident_faces.size());
        for (int f_id : incident_faces) {
            for (int fj = 0; fj < m_faces.cols(); fj++) {
                const int vi = m_faces(f_id, fj);
                const int vj = m_faces(f_id, (fj + 1) % m_faces.cols());
                if (vi != static_cast<int>(v) && vj != static_cast<int>(v)) {
                    continue;
                }
                const int edge_id = m_faces_to_edges(f_id, fj);
                for (int k = 0; k < 2; k++) {
                    const int nbr = m_edges_to_faces(edge_id, k);
                    if (nbr < 0 || nbr == f_id) {
                        continue;
                    }
                    auto it = face_to_local.find(nbr);
                    if (it != face_to_local.end()) {
                        adjacency[face_to_local[f_id]].push_back(it->second);
                    }
                }
            }
        }

        std::vector<char> visited(incident_faces.size(), 0);
        std::vector<int> stack;
        stack.push_back(0);
        visited[0] = 1;
        while (!stack.empty()) {
            const int cur = stack.back();
            stack.pop_back();
            for (int nbr : adjacency[cur]) {
                if (!visited[nbr]) {
                    visited[nbr] = 1;
                    stack.push_back(nbr);
                }
            }
        }
        if (std::find(visited.begin(), visited.end(), 0) != visited.end()) {
            throw std::runtime_error("Non-manifold vertex detected.");
        }
    }
}

CollisionMesh CollisionMesh::build(
    Eigen::ConstRef<Eigen::MatrixXd> rest_positions,
    Eigen::ConstRef<Eigen::MatrixXi> faces)
{
    return CollisionMesh(rest_positions, faces);
}

CollisionMesh CollisionMesh::load_from_obj(const std::string& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Unable to open OBJ file: " + path);
    }

    std::vector<Eigen::Vector3d> verts;
    std::vector<Eigen::Vector3i> faces;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "v") {
            double x = 0, y = 0, z = 0;
            ss >> x >> y >> z;
            verts.emplace_back(x, y, z);
        } else if (tag == "f") {
            std::vector<int> indices;
            std::string token;
            while (ss >> token) {
                const auto slash = token.find('/');
                if (slash != std::string::npos) {
                    token = token.substr(0, slash);
                }
                if (token.empty()) {
                    continue;
                }
                int idx = std::stoi(token);
                if (idx < 0) {
                    idx = static_cast<int>(verts.size()) + idx + 1;
                }
                indices.push_back(idx - 1);
            }
            if (indices.size() < 3) {
                throw std::runtime_error("OBJ face has fewer than 3 vertices.");
            }
            for (size_t i = 1; i + 1 < indices.size(); i++) {
                faces.emplace_back(indices[0], indices[i], indices[i + 1]);
            }
        }
    }

    Eigen::MatrixXd V(static_cast<int>(verts.size()), 3);
    for (int i = 0; i < V.rows(); i++) {
        V.row(i) = verts[static_cast<size_t>(i)];
    }
    Eigen::MatrixXi F(static_cast<int>(faces.size()), 3);
    for (int i = 0; i < F.rows(); i++) {
        F.row(i) = faces[static_cast<size_t>(i)];
    }
    return CollisionMesh(V, F);
}

void CollisionMesh::init_edges_to_faces()
{
    m_edges_to_faces.resize(m_edges.rows(), 2);
    m_edges_to_faces.setConstant(-1);
    if (m_faces.size() == 0 || m_faces_to_edges.size() == 0) {
        return;
    }
    for (int f = 0; f < m_faces_to_edges.rows(); f++) {
        for (int le = 0; le < m_faces_to_edges.cols(); le++) {
            const int edge_id = m_faces_to_edges(f, le);
            if (m_edges_to_faces(edge_id, 0) < 0) {
                m_edges_to_faces(edge_id, 0) = f;
            } else if (m_edges_to_faces(edge_id, 1) < 0) {
                m_edges_to_faces(edge_id, 1) = f;
            } else {
                throw std::runtime_error("Edge has more than two incident faces.");
            }
        }
    }
}

void CollisionMesh::init_vertices_to_edges()
{
    m_vertices_to_edges.assign(num_vertices(), {});
    for (int e = 0; e < m_edges.rows(); e++) {
        for (int j = 0; j < m_edges.cols(); j++) {
            const int v = m_edges(e, j);
            m_vertices_to_edges[v].push_back(e);
        }
    }
}

void CollisionMesh::init_vertices_to_faces()
{
    m_vertices_to_faces.assign(num_vertices(), {});
    for (int f = 0; f < m_faces.rows(); f++) {
        for (int j = 0; j < m_faces.cols(); j++) {
            const int v = m_faces(f, j);
            m_vertices_to_faces[v].push_back(f);
        }
    }
}

void CollisionMesh::init_boundary()
{
    m_is_vertex_on_boundary.assign(num_vertices(), true);
    if (num_vertices() == 0) {
        return;
    }
    if (dim() == 2) {
        for (size_t v = 0; v < num_vertices(); v++) {
            m_is_vertex_on_boundary[v] = m_vertices_to_edges[v].size() <= 1;
        }
        return;
    }

    for (int e = 0; e < m_edges.rows(); e++) {
        if (m_edges_to_faces(e, 0) >= 0 && m_edges_to_faces(e, 1) >= 0) {
            for (int j = 0; j < m_edges.cols(); j++) {
                m_is_vertex_on_boundary[m_edges(e, j)] = false;
            }
        }
    }
}

Eigen::MatrixXi CollisionMesh::construct_faces_to_edges(
    Eigen::ConstRef<Eigen::MatrixXi> faces,
    Eigen::ConstRef<Eigen::MatrixXi> edges)
{
    if (faces.size() == 0) {
        return Eigen::MatrixXi(faces.rows(), faces.cols());
    }
    if (edges.size() == 0) {
        throw std::runtime_error("Edges must be provided when faces are non-empty.");
    }

    std::unordered_map<std::pair<int, int>, int, PairHash> edge_map;
    edge_map.reserve(static_cast<size_t>(edges.rows()));
    for (int ei = 0; ei < edges.rows(); ei++) {
        const int a = std::min(edges(ei, 0), edges(ei, 1));
        const int b = std::max(edges(ei, 0), edges(ei, 1));
        edge_map.emplace(std::make_pair(a, b), ei);
    }

    Eigen::MatrixXi faces_to_edges(faces.rows(), faces.cols());
    for (int fi = 0; fi < faces.rows(); fi++) {
        for (int fj = 0; fj < faces.cols(); fj++) {
            const int vi = faces(fi, fj);
            const int vj = faces(fi, (fj + 1) % faces.cols());
            const int a = std::min(vi, vj);
            const int b = std::max(vi, vj);
            auto search = edge_map.find(std::make_pair(a, b));
            if (search == edge_map.end()) {
                throw std::runtime_error("Unable to find edge for face.");
            }
            faces_to_edges(fi, fj) = search->second;
        }
    }

    return faces_to_edges;
}

double CollisionMesh::edge_length(int edge_id) const
{
    const Eigen::RowVectorXd p0 = m_rest_positions.row(m_edges(edge_id, 0));
    const Eigen::RowVectorXd p1 = m_rest_positions.row(m_edges(edge_id, 1));
    return (p0 - p1).norm();
}

double CollisionMesh::max_edge_length() const
{
    double val = 0.0;
    for (int i = 0; i < m_edges.rows(); i++) {
        val = std::max(val, edge_length(i));
    }
    return val;
}

} // namespace ipc

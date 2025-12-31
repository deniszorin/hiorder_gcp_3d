#include "potential_cpp.hpp"

#include <cmath>
#include <stdexcept>

namespace ipc {

namespace {

constexpr double kSingularValue = 1e12;
constexpr double kEps = 1e-12;

struct EdgeProjection {
    Eigen::Vector3d P_e;
    double r_e;
    Eigen::Vector3d d_unit;
    Eigen::Vector3d unit_edge;
};

inline double safe_norm(const Eigen::Vector3d& v)
{
    const double n = v.norm();
    return n < kEps ? kEps : n;
}

inline Eigen::Vector3d unit_vec(const Eigen::Vector3d& v)
{
    return v / safe_norm(v);
}

inline Eigen::Vector3d vertex_position(const PotentialCollisionMesh& mesh, const int idx)
{
    return mesh.rest_positions().row(idx).transpose();
}

EdgeProjection edge_projection(const Eigen::Vector3d& q, const Eigen::Vector3d& p0, const Eigen::Vector3d& p1)
{
    const Eigen::Vector3d d = p1 - p0;
    const double d_norm = safe_norm(d);
    const Eigen::Vector3d d_unit = d / d_norm;
    const double t = (q - p0).dot(d_unit);
    const Eigen::Vector3d P_e = p0 + t * d_unit;
    const Eigen::Vector3d diff = q - P_e;
    const double r_e = diff.norm();
    const Eigen::Vector3d unit_edge = diff / safe_norm(diff);
    return EdgeProjection{ P_e, r_e, d_unit, unit_edge };
}

inline double H(double z)
{
    if (z < -1.0) {
        return 0.0;
    }
    if (z > 1.0) {
        return 1.0;
    }
    return ((2.0 - z) * (z + 1.0) * (z + 1.0)) / 4.0;
}

inline double H_alpha(double t, double alpha)
{
    return H(t / alpha);
}

inline double h_local(double z)
{
    return (2.0 * z + 1.0) * (z - 1.0) * (z - 1.0);
}

inline double h_epsilon(double z, double epsilon)
{
    return h_local(z / epsilon);
}

int find_local_edge(const PotentialCollisionMesh& mesh, const int fidx, const int edge_idx)
{
    const Eigen::MatrixXi& face_edges = mesh.faces_to_edges();
    for (int i = 0; i < face_edges.cols(); i++) {
        if (face_edges(fidx, i) == edge_idx) {
            return i;
        }
    }
    throw std::runtime_error("Edge not found in face.");
}

bool outside_face(const double r_f)
{
    return r_f > 0.0;
}

double phi_ef(
    const Eigen::Vector3d& q,
    const PotentialCollisionMesh& mesh,
    const int fidx,
    const int local_edge)
{
    const Eigen::MatrixXi& F = mesh.faces();
    const int v0 = F(fidx, local_edge);
    const int v1 = F(fidx, (local_edge + 1) % 3);
    const Eigen::Vector3d p0 = vertex_position(mesh, v0);
    const Eigen::Vector3d p1 = vertex_position(mesh, v1);

    const EdgeProjection proj = edge_projection(q, p0, p1);
    return proj.unit_edge.dot(mesh.edge_inward(fidx)[static_cast<size_t>(local_edge)]);
}

std::pair<double, double> phi_ve(
    const Eigen::Vector3d& q,
    const Eigen::Vector3d& p0,
    const Eigen::Vector3d& p1,
    const Eigen::Vector3d& d_unit)
{
    const Eigen::Vector3d unit0 = unit_vec(q - p0);
    const Eigen::Vector3d unit1 = unit_vec(q - p1);
    const double phi0 = unit0.dot(d_unit);
    const double phi1 = unit1.dot(-d_unit);
    return std::make_pair(phi0, phi1);
}

bool outside_edge(
    const Eigen::Vector3d& q,
    const double r_e,
    const Eigen::Vector3d& P_e,
    const Eigen::Vector3d& edge_normal,
    const double r_f0,
    const double phi_ef0,
    const bool has_f1,
    const double r_f1,
    const double phi_ef1)
{
    double r_min = r_e;
    int closest_elt = 2;

    if (phi_ef0 > 0.0) {
        const double r0 = std::abs(r_f0);
        if (r0 < r_min) {
            r_min = r0;
            closest_elt = 0;
        }
    }
    if (has_f1 && phi_ef1 > 0.0) {
        const double r1 = std::abs(r_f1);
        if (r1 < r_min) {
            r_min = r1;
            closest_elt = 1;
        }
    }

    if (closest_elt == 0) {
        return r_f0 > 0.0;
    }
    if (closest_elt == 1) {
        return r_f1 > 0.0;
    }

    return (q - P_e).dot(edge_normal) > 0.0;
}

bool outside_vertex(
    const Eigen::Vector3d& q,
    const int v_idx,
    const PotentialCollisionMesh& mesh,
    const double r_v,
    const double r_f_min_signed,
    const int face_min,
    const double r_e_min,
    const int edge_min,
    const Eigen::Vector3d& P_e_min)
{
    double r_f_min_abs = 1e30;
    if (face_min >= 0) {
        r_f_min_abs = std::abs(r_f_min_signed);
    }
    double r_min_fe = r_f_min_abs;
    if (r_e_min < r_min_fe) {
        r_min_fe = r_e_min;
    }

    const bool use_vertex = r_v < r_min_fe;
    const bool use_face = (face_min >= 0) && (r_f_min_abs <= r_e_min) && (!use_vertex);
    const bool use_edge = (edge_min >= 0) && (!use_face) && (!use_vertex);

    if (use_face) {
        return r_f_min_signed > 0.0;
    }
    if (use_edge) {
        return (q - P_e_min).dot(mesh.edge_normals().row(edge_min)) > 0.0;
    }

    return mesh.pointed_vertices()[static_cast<size_t>(v_idx)] != 0;
}

double potential_face(
    const Eigen::Vector3d& q,
    const int fidx,
    const PotentialCollisionMesh& mesh,
    const double alpha,
    const double p,
    const double epsilon,
    const bool localized,
    const bool one_sided)
{
    const Eigen::MatrixXi& F = mesh.faces();
    const int v0 = F(fidx, 0);
    const Eigen::Vector3d p0 = vertex_position(mesh, v0);
    const Eigen::Vector3d n = mesh.normals().row(fidx);

    const double r_f = (q - p0).dot(n);
    const double r_f_abs = std::abs(r_f);

    double B = 1.0;
    for (int local_edge = 0; local_edge < 3; local_edge++) {
        const double phi = phi_ef(q, mesh, fidx, local_edge);
        B *= H_alpha(phi, alpha);
    }

    const double denom = std::pow(r_f_abs, p);
    double I_f = (denom <= kEps) ? kSingularValue : (B / denom);
    if (one_sided && !outside_face(r_f)) {
        I_f = 0.0;
    }
    if (localized) {
        I_f *= h_epsilon(r_f_abs, epsilon);
    }
    return I_f;
}

double potential_edge(
    const Eigen::Vector3d& q,
    const int edge_idx,
    const PotentialCollisionMesh& mesh,
    const double alpha,
    const double p,
    const double epsilon,
    const bool localized,
    const bool one_sided)
{
    const int a = mesh.edges()(edge_idx, 0);
    const int b = mesh.edges()(edge_idx, 1);
    const Eigen::Vector3d p0 = vertex_position(mesh, a);
    const Eigen::Vector3d p1 = vertex_position(mesh, b);

    const EdgeProjection proj = edge_projection(q, p0, p1);
    const auto phi_pair = phi_ve(q, p0, p1, proj.d_unit);
    const double phi0 = phi_pair.first;
    const double phi1 = phi_pair.second;

    const int f0 = mesh.edges_to_faces()(edge_idx, 0);
    const int f1 = mesh.edges_to_faces()(edge_idx, 1);
    const bool has_f1 = f1 >= 0;

    double phi_ef0 = 0.0;
    double r_f0 = 0.0;
    double h_face_0 = 0.0;
    if (f0 >= 0) {
        const int local0 = find_local_edge(mesh, f0, edge_idx);
        phi_ef0 = proj.unit_edge.dot(mesh.edge_inward(f0)[static_cast<size_t>(local0)]);
        h_face_0 = H_alpha(phi_ef0, alpha);
        const int v0 = mesh.faces()(f0, 0);
        r_f0 = (q - vertex_position(mesh, v0)).dot(mesh.normals().row(f0));
        if (one_sided) {
            h_face_0 *= outside_face(r_f0);
        }
    }

    double phi_ef1 = 0.0;
    double r_f1 = 0.0;
    double h_face_1 = 0.0;
    if (has_f1) {
        const int local1 = find_local_edge(mesh, f1, edge_idx);
        phi_ef1 = proj.unit_edge.dot(mesh.edge_inward(f1)[static_cast<size_t>(local1)]);
        h_face_1 = H_alpha(phi_ef1, alpha);
        const int v1 = mesh.faces()(f1, 0);
        r_f1 = (q - vertex_position(mesh, v1)).dot(mesh.normals().row(f1));
        if (one_sided) {
            h_face_1 *= outside_face(r_f1);
        }
    }

    const double B_edge = (1.0 - h_face_0 - h_face_1)
        * H_alpha(phi0, alpha) * H_alpha(phi1, alpha);

    const double denom = std::pow(proj.r_e, p);
    double I_e = (denom <= kEps) ? kSingularValue : (B_edge / denom);

    if (one_sided) {
        const bool outside = outside_edge(
            q,
            proj.r_e,
            proj.P_e,
            mesh.edge_normals().row(edge_idx),
            r_f0,
            phi_ef0,
            has_f1,
            r_f1,
            phi_ef1);
        if (!outside) {
            I_e = 0.0;
        }
    }
    if (localized) {
        I_e *= h_epsilon(proj.r_e, epsilon);
    }
    return I_e;
}

std::tuple<double, double, int> vertex_face_term(
    const Eigen::Vector3d& q,
    const int v_idx,
    const PotentialCollisionMesh& mesh,
    const double alpha,
    const bool one_sided)
{
    double face_term = 0.0;
    double r_f_min_signed = 0.0;
    int face_min = -1;
    double r_f_min = 1e30;

    const auto& face_list = mesh.vertices_to_faces()[static_cast<size_t>(v_idx)];
    for (const int f : face_list) {
        const int v0 = mesh.faces()(f, 0);
        const int v1 = mesh.faces()(f, 1);
        const int v2 = mesh.faces()(f, 2);

        int e0 = 0;
        int e1 = 0;
        if (v_idx == v0) {
            e0 = 0;
            e1 = 2;
        } else if (v_idx == v1) {
            e0 = 0;
            e1 = 1;
        } else if (v_idx == v2) {
            e0 = 1;
            e1 = 2;
        } else {
            continue;
        }

        const double phi0 = phi_ef(q, mesh, f, e0);
        const double phi1 = phi_ef(q, mesh, f, e1);
        double h0 = H_alpha(phi0, alpha);
        double h1 = H_alpha(phi1, alpha);

        const int v_face = mesh.faces()(f, 0);
        const double r_f = (q - vertex_position(mesh, v_face)).dot(mesh.normals().row(f));
        if (one_sided) {
            const bool outside = outside_face(r_f);
            h0 *= outside;
            h1 *= outside;
        }

        face_term += h0 * h1;
        if (phi0 > 0.0 && phi1 > 0.0) {
            const double r_abs = std::abs(r_f);
            if (r_abs < r_f_min) {
                r_f_min = r_abs;
                face_min = f;
                r_f_min_signed = r_f;
            }
        }
    }

    return std::make_tuple(face_term, r_f_min_signed, face_min);
}

std::tuple<double, double, int, Eigen::Vector3d> vertex_edge_term(
    const Eigen::Vector3d& q,
    const int v_idx,
    const PotentialCollisionMesh& mesh,
    const double alpha,
    const bool one_sided)
{
    double edge_term = 0.0;
    double r_e_min = 1e30;
    int edge_min = -1;
    Eigen::Vector3d P_e_min(0.0, 0.0, 0.0);

    const auto& edge_list = mesh.vertices_to_edges()[static_cast<size_t>(v_idx)];
    for (const int edge_idx : edge_list) {
        const int a = mesh.edges()(edge_idx, 0);
        const int b = mesh.edges()(edge_idx, 1);
        const Eigen::Vector3d p0 = vertex_position(mesh, a);
        const Eigen::Vector3d p1 = vertex_position(mesh, b);
        const EdgeProjection proj = edge_projection(q, p0, p1);

        double phi_v = 0.0;
        if (v_idx == a) {
            phi_v = phi_ve(q, p0, p1, proj.d_unit).first;
        } else {
            phi_v = phi_ve(q, p0, p1, proj.d_unit).second;
        }
        double h_v = H_alpha(phi_v, alpha);

        const int f0 = mesh.edges_to_faces()(edge_idx, 0);
        const int f1 = mesh.edges_to_faces()(edge_idx, 1);
        const bool has_f1 = f1 >= 0;

        double phi_ef0 = 0.0;
        double r_f0 = 0.0;
        double h_face_0 = 0.0;
        if (f0 >= 0) {
            const int local0 = find_local_edge(mesh, f0, edge_idx);
            phi_ef0 = proj.unit_edge.dot(mesh.edge_inward(f0)[static_cast<size_t>(local0)]);
            h_face_0 = H_alpha(phi_ef0, alpha);
            const int v0 = mesh.faces()(f0, 0);
            r_f0 = (q - vertex_position(mesh, v0)).dot(mesh.normals().row(f0));
            if (one_sided) {
                h_face_0 *= outside_face(r_f0);
            }
        }

        double phi_ef1 = 0.0;
        double r_f1 = 0.0;
        double h_face_1 = 0.0;
        if (has_f1) {
            const int local1 = find_local_edge(mesh, f1, edge_idx);
            phi_ef1 = proj.unit_edge.dot(mesh.edge_inward(f1)[static_cast<size_t>(local1)]);
            h_face_1 = H_alpha(phi_ef1, alpha);
            const int v1 = mesh.faces()(f1, 0);
            r_f1 = (q - vertex_position(mesh, v1)).dot(mesh.normals().row(f1));
            if (one_sided) {
                h_face_1 *= outside_face(r_f1);
            }
        }

        if (one_sided) {
            const bool outside = outside_edge(
                q,
                proj.r_e,
                proj.P_e,
                mesh.edge_normals().row(edge_idx),
                r_f0,
                phi_ef0,
                has_f1,
                r_f1,
                phi_ef1);
            if (!outside) {
                h_v = 0.0;
            }
        }

        edge_term += (1.0 - h_face_0 - h_face_1) * h_v;

        if (phi_v > 0.0 && proj.r_e < r_e_min) {
            r_e_min = proj.r_e;
            edge_min = edge_idx;
            P_e_min = proj.P_e;
        }
    }

    return std::make_tuple(edge_term, r_e_min, edge_min, P_e_min);
}

double potential_vertex(
    const Eigen::Vector3d& q,
    const int v_idx,
    const PotentialCollisionMesh& mesh,
    const double alpha,
    const double p,
    const double epsilon,
    const bool localized,
    const bool one_sided)
{
    const double r_v = (q - vertex_position(mesh, v_idx)).norm();

    const auto face_info = vertex_face_term(q, v_idx, mesh, alpha, one_sided);
    const auto edge_info = vertex_edge_term(q, v_idx, mesh, alpha, one_sided);
    const double face_term = std::get<0>(face_info);
    const double r_f_min_signed = std::get<1>(face_info);
    const int face_min = std::get<2>(face_info);
    const double edge_term = std::get<0>(edge_info);
    const double r_e_min = std::get<1>(edge_info);
    const int edge_min = std::get<2>(edge_info);
    const Eigen::Vector3d P_e_min = std::get<3>(edge_info);

    if (one_sided) {
        if (!outside_vertex(
                q,
                v_idx,
                mesh,
                r_v,
                r_f_min_signed,
                face_min,
                r_e_min,
                edge_min,
                P_e_min)) {
            return 0.0;
        }
    }

    const double denom = std::pow(r_v, p);
    double I_v = (denom <= kEps) ? kSingularValue
                                 : (1.0 - face_term - edge_term) / denom;
    if (localized) {
        I_v *= h_epsilon(r_v, epsilon);
    }
    return I_v;
}

void get_vertices_and_edges(
    const std::vector<int>& face_indices,
    const PotentialCollisionMesh& mesh,
    std::vector<int>& edge_list,
    std::vector<int>& vertex_list)
{
    std::vector<char> edge_mark(mesh.num_edges(), 0);
    std::vector<char> vertex_mark(mesh.num_vertices(), 0);

    edge_list.clear();
    vertex_list.clear();

    for (const int fidx : face_indices) {
        for (int i = 0; i < 3; i++) {
            const int edge_idx = mesh.faces_to_edges()(fidx, i);
            if (!edge_mark[edge_idx]) {
                edge_mark[edge_idx] = 1;
                edge_list.push_back(edge_idx);
            }
        }
        const int v0 = mesh.faces()(fidx, 0);
        const int v1 = mesh.faces()(fidx, 1);
        const int v2 = mesh.faces()(fidx, 2);
        if (!vertex_mark[v0]) {
            vertex_mark[v0] = 1;
            vertex_list.push_back(v0);
        }
        if (!vertex_mark[v1]) {
            vertex_mark[v1] = 1;
            vertex_list.push_back(v1);
        }
        if (!vertex_mark[v2]) {
            vertex_mark[v2] = 1;
            vertex_list.push_back(v2);
        }
    }
}

double smoothed_offset_potential_point(
    const Eigen::Vector3d& q,
    const std::vector<int>& face_indices,
    const PotentialCollisionMesh& mesh,
    const double alpha,
    const double p,
    const double epsilon,
    const bool include_faces,
    const bool include_edges,
    const bool include_vertices,
    const bool localized,
    const bool one_sided)
{
    if (!(include_faces || include_edges || include_vertices)) {
        return 0.0;
    }

    double face_sum = 0.0;
    double edge_sum = 0.0;
    double vertex_sum = 0.0;

    std::vector<int> edge_list;
    std::vector<int> vertex_list;
    get_vertices_and_edges(face_indices, mesh, edge_list, vertex_list);

    for (const int fidx : face_indices) {
        if (include_faces) {
            face_sum += potential_face(
                q,
                fidx,
                mesh,
                alpha,
                p,
                epsilon,
                localized,
                one_sided);
        }
    }

    if (include_edges) {
        for (const int edge_idx : edge_list) {
            edge_sum += potential_edge(
                q,
                edge_idx,
                mesh,
                alpha,
                p,
                epsilon,
                localized,
                one_sided);
        }
    }

    if (include_vertices) {
        for (const int v_idx : vertex_list) {
            vertex_sum += potential_vertex(
                q,
                v_idx,
                mesh,
                alpha,
                p,
                epsilon,
                localized,
                one_sided);
        }
    }

    return face_sum + edge_sum + vertex_sum;
}

} // namespace

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
    bool one_sided)
{
    if (q.cols() != 3) {
        throw std::runtime_error("q must have shape (nq, 3).");
    }

    std::vector<int> face_indices(static_cast<size_t>(mesh.num_faces()));
    for (int i = 0; i < mesh.faces().rows(); i++) {
        face_indices[static_cast<size_t>(i)] = i;
    }

    Eigen::VectorXd out(q.rows());
    for (int i = 0; i < q.rows(); i++) {
        const Eigen::Vector3d qi = q.row(i);
        out[i] = smoothed_offset_potential_point(
            qi,
            face_indices,
            mesh,
            alpha,
            p,
            epsilon,
            include_faces,
            include_edges,
            include_vertices,
            localized,
            one_sided);
    }

    return out;
}

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
    bool one_sided)
{
    PotentialCollisionMesh mesh(V, F);
    return smoothed_offset_potential(
        q,
        mesh,
        alpha,
        p,
        epsilon,
        include_faces,
        include_edges,
        include_vertices,
        localized,
        one_sided);
}

} // namespace ipc

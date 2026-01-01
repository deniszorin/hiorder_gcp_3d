#include "potential_cpp.hpp"

#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace ipc {

namespace {

constexpr double kSingularValue = 1e12;
constexpr double kEps = 1e-12;

struct PotentialParameters {
    double alpha;
    double p;
    double epsilon;
    bool localized;
    bool one_sided;
};

using FacePoints = std::array<Eigen::Vector3d, 3>;
using FacePointsPair = std::array<FacePoints, 2>;

// ****************************************************************************
// Basic geometry functions for numba to avoid function calls

inline double safe_norm(const Eigen::Vector3d& v)
{
    const double n = v.norm();
    return n < kEps ? kEps : n;
}

inline Eigen::Vector3d unit_vec(const Eigen::Vector3d& v)
{
    return v / safe_norm(v);
}

inline Eigen::Vector3d unit_dir(const Eigen::Vector3d& p0, const Eigen::Vector3d& p1)
{
    return unit_vec(p1 - p0);
}

inline Eigen::Vector3d vertex_position(const PotentialCollisionMesh& mesh, const int idx)
{
    return mesh.rest_positions().row(idx).transpose();
}

inline Eigen::Vector3d face_normal(
    const Eigen::Vector3d& p0, const Eigen::Vector3d& p1, const Eigen::Vector3d& p2)
{
    Eigen::Vector3d n = (p1 - p0).cross(p2 - p0);
    return n / safe_norm(n);
}

inline void face_edge_endpoints(
    const Eigen::Vector3d& p0, const Eigen::Vector3d& p1, const Eigen::Vector3d& p2,
    const int local_edge, Eigen::Vector3d& edge_p0, Eigen::Vector3d& edge_p1)
{
    if (local_edge == 0) {
        edge_p0 = p0;
        edge_p1 = p1;
    } else if (local_edge == 1) {
        edge_p0 = p1;
        edge_p1 = p2;
    } else {
        edge_p0 = p2;
        edge_p1 = p0;
    }
}

inline Eigen::Vector3d face_edge_inward(
    const Eigen::Vector3d& n,
    const Eigen::Vector3d& p0, const Eigen::Vector3d& p1, const Eigen::Vector3d& p2,
    const int local_edge)
{
    Eigen::Vector3d edge_p0;
    Eigen::Vector3d edge_p1;
    face_edge_endpoints(p0, p1, p2, local_edge, edge_p0, edge_p1);
    const Eigen::Vector3d d_e = unit_dir(edge_p0, edge_p1);
    return n.cross(d_e);
}

inline void edge_projection(
    const Eigen::Vector3d& q, const Eigen::Vector3d& p0, const Eigen::Vector3d& d_unit,
    Eigen::Vector3d& P_e, double& r_e, Eigen::Vector3d& unit_Pe_to_q)
{
    // the choice of the direction on the edge does not affect
    // projected position P_e can use any
    const double t = (q - p0).dot(d_unit);
    P_e = p0 + t * d_unit;
    // P_e: projection of q to each edge line.
    // r_e: distance from q to each edge line.
    const Eigen::Vector3d diff = q - P_e;
    r_e = diff.norm();
    unit_Pe_to_q = diff / safe_norm(diff);
}

// ****************************************************************************
// Potential blending/localization functions

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

// ****************************************************************************
// Helpers for local to global and back vertex/edge index conversion

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


// ****************************************************************************
//  Potential directional terms Phi^{e,f}, Phi^{v,e} 

double phi_ef(
    const Eigen::Vector3d& q,
    const Eigen::Vector3d& n,
    const Eigen::Vector3d& p0, const Eigen::Vector3d& p1, const Eigen::Vector3d& p2,
    const int local_edge)
{
    Eigen::Vector3d edge_p0;
    Eigen::Vector3d edge_p1;
    face_edge_endpoints(p0, p1, p2, local_edge, edge_p0, edge_p1);

    const Eigen::Vector3d d_unit = unit_dir(edge_p0, edge_p1);
    Eigen::Vector3d P_e;
    double r_e;
    Eigen::Vector3d unit_Pe_to_q;
    edge_projection(q, edge_p0, d_unit, P_e, r_e, unit_Pe_to_q);

    // Phi^{e,f} := (q - P_e)_+ dot (n x d_e[i]) per face
    const Eigen::Vector3d edge_inward = face_edge_inward(n, p0, p1, p2, local_edge);
    return unit_Pe_to_q.dot(edge_inward);
}

std::pair<double, double> phi_ve(
    const Eigen::Vector3d& q,
    const Eigen::Vector3d& p0, const Eigen::Vector3d& p1,
    const Eigen::Vector3d& d_unit)
{
    const Eigen::Vector3d unit0 = unit_dir(p0, q);
    const Eigen::Vector3d unit1 = unit_dir(p1, q);
    
    // Phi^{i,e}, i=0,1 factors per edge
    const double phi0 = unit0.dot(d_unit);
    const double phi1 = unit1.dot(-d_unit);
    return std::make_pair(phi0, phi1);
}

Eigen::Vector3d edge_normal_from_faces(
    const FacePointsPair& face_points, const bool has_f1)
{
    const auto& f0 = face_points[0];
    Eigen::Vector3d n_avg = face_normal(f0[0], f0[1], f0[2]);
    if (has_f1) {
        const auto& f1 = face_points[1];
        n_avg += face_normal(f1[0], f1[1], f1[2]);
    }
    const double n_norm = n_avg.norm();
    if (n_norm > kEps) {
        n_avg /= n_norm;
    }
    return n_avg;
}

// ****************************************************************************
// Functions to check if a point is outside a given face, edge, vertex

bool outside_face(const double r_f)
{
    // Assumes that signed distance to the face is given (computed elsewhere) and returns if it is positive
    return r_f > 0.0;
}

bool outside_edge(
    const Eigen::Vector3d& q,
    const double r_e, const Eigen::Vector3d& P_e, const Eigen::Vector3d& edge_normal,
    const double r_f0, const double phi_ef0, const bool has_f1, const double r_f1, const double phi_ef1)
{
    // Assumes that that the distance to the edge r_e, projection P_e,
    // as well as  signed distances to faces f0, f1, and directional factors Phi^{e,fi}, i=0,1
    // are given, implements the logic of the local outside test:
    // determine if one of the halfplanes of f0, f1, or the edge iself is closest,
    // use the test for the closest element: signed distance for face, and dot product with
    // the edge normal (this works because the volume where the edge is closest is within pi/2 of the normal)
    // initialize r_min with the distance to the edge
    double r_min = r_e;
    int closest_elt = 2;

    // check where the distance to f0 is less than edge
    // if the projection is within the halfplane (Phi^{e,f0} > 0)
    if (phi_ef0 > 0.0) {
        const double r0 = std::abs(r_f0);
        if (r0 < r_min) {
            r_min = r0;
            closest_elt = 0;
        }
    }
    // same for f1
    if (has_f1 && phi_ef1 > 0.0) {
        const double r1 = std::abs(r_f1);
        if (r1 < r_min) {
            r_min = r1;
            closest_elt = 1;
        }
    }

    // if f0 is closest, check if  signed distance to it  r_{f_0} is positive
    if (closest_elt == 0) {
        return r_f0 > 0.0;
    }
    // if f1 is closest, check if  signed distance to it  r_{f_1} is positive
    if (closest_elt == 1) {
        return r_f1 > 0.0;
    }

    // if the edge itself is closest,  check dot product with the average normal
    // it always points outside, and the sector where edge is closest is within pi/2
    // of the average normal.
    return (q - P_e).dot(edge_normal) > 0.0;
}

bool outside_vertex(
    const Eigen::Vector3d& q,
    const double r_v,
    const double r_f_min_signed, const int face_min,
    const double r_e_min, const int edge_min,
    const Eigen::Vector3d& P_e_min, const Eigen::Vector3d& edge_normal_min,
    const bool pointed_vertex)
{
    // Assumes distance to the vertex,
    // signed distance to the closest face,
    // distance and projection to the closest edge are given.
    // Determines which element (closest face, closest edge or vertex) is closest, and
    // then does the outside check based on the element.
    // get closest face sector and edge ray if any
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
        return (q - P_e_min).dot(edge_normal_min) > 0.0;
    }

    // if any points left unassigned after a pass over all edges, use
    // the pointed-vertex flag for those vertex-closest queries.
    // the reason for this is that if the vertex is closest, this means q
    // is in the polar cone and the pointed-vertex flag indicates if this
    // cones is inside or outside (the whole cone has to be on one side)
    return pointed_vertex;
}

// ****************************************************************************
// Potential evaluation, face, edge, vertex components

double potential_face(
    const Eigen::Vector3d& q, const FacePoints& face_points,
    const PotentialParameters& params)
{
    const auto& p0 = face_points[0];
    const auto& p1 = face_points[1];
    const auto& p2 = face_points[2];
    const Eigen::Vector3d n = face_normal(p0, p1, p2);

    // signed distance to the face plane.
    const double r_f = (q - p0).dot(n);
    const double r_f_abs = std::abs(r_f);

    double B = 1.0;
    for (int local_edge = 0; local_edge < 3; local_edge++) {
        const double phi = phi_ef(q, n, p0, p1, p2, local_edge);
        B *= H_alpha(phi, params.alpha);
    }

    const double denom = std::pow(r_f_abs, params.p);
    double I_f = (denom <= kEps) ? kSingularValue : (B / denom);
    if (params.one_sided && !outside_face(r_f)) {
        I_f = 0.0;
    }
    if (params.localized) {
        I_f *= h_epsilon(r_f_abs, params.epsilon);
    }
    return I_f;
}

double potential_edge(
    const Eigen::Vector3d& q, const FacePointsPair& face_points,
    const int local0, const int local1, const bool has_f1,
    const PotentialParameters& params)
{
    const auto& f0 = face_points[0];
    Eigen::Vector3d edge_p0;
    Eigen::Vector3d edge_p1;
    face_edge_endpoints(f0[0], f0[1], f0[2], local0, edge_p0, edge_p1);

    const Eigen::Vector3d d_unit = unit_dir(edge_p0, edge_p1);
    Eigen::Vector3d P_e;
    double r_e;
    Eigen::Vector3d unit_Pe_to_q;
    edge_projection(q, edge_p0, d_unit, P_e, r_e, unit_Pe_to_q);
    const auto phi_pair = phi_ve(q, edge_p0, edge_p1, d_unit);
    const double phi0 = phi_pair.first;
    const double phi1 = phi_pair.second;

    const Eigen::Vector3d n0 = face_normal(f0[0], f0[1], f0[2]);
    const Eigen::Vector3d edge_inward_0 =
        face_edge_inward(n0, f0[0], f0[1], f0[2], local0);
    const double phi_ef0 = unit_Pe_to_q.dot(edge_inward_0);
    double h_face_0 = H_alpha(phi_ef0, params.alpha);
    const double r_f0 = (q - f0[0]).dot(n0);
    if (params.one_sided) {
        h_face_0 *= outside_face(r_f0);
    }

    double h_face_1 = 0.0;
    double phi_ef1 = 0.0;
    double r_f1 = 0.0;
    if (has_f1) {
        const auto& f1 = face_points[1];
        const Eigen::Vector3d n1 = face_normal(f1[0], f1[1], f1[2]);
        const Eigen::Vector3d edge_inward_1 =
            face_edge_inward(n1, f1[0], f1[1], f1[2], local1);
        phi_ef1 = unit_Pe_to_q.dot(edge_inward_1);
        h_face_1 = H_alpha(phi_ef1, params.alpha);
        r_f1 = (q - f1[0]).dot(n1);
        if (params.one_sided) {
            h_face_1 *= outside_face(r_f1);
        }
    }

    const double B_edge = (1.0 - h_face_0 - h_face_1)
        * H_alpha(phi0, params.alpha) * H_alpha(phi1, params.alpha);

    // distance to edge r_e already computed per edge
    const double denom = std::pow(r_e, params.p);
    double I_e = (denom <= kEps) ? kSingularValue : (B_edge / denom);

    if (params.one_sided) {
        const Eigen::Vector3d edge_n = edge_normal_from_faces(face_points, has_f1);
        const bool outside = outside_edge(
            q,
            r_e, P_e, edge_n,
            r_f0, phi_ef0, has_f1, r_f1, phi_ef1);
        if (!outside) {
            I_e = 0.0;
        }
    }
    if (params.localized) {
        I_e *= h_epsilon(r_e, params.epsilon);
    }
    return I_e;
}

void vertex_face_term(
    const Eigen::Vector3d& q, const Eigen::Vector3d& p_v,
    const Eigen::Vector3d* neighbor_points, const int neighbor_count, const bool is_boundary,
    const double alpha, const bool one_sided,
    double& face_term, double& r_f_min_signed, int& face_min)
{
    // get closest face sector and edge ray if any
    // initialize to inf
    // local index of closest face
    face_term = 0.0;
    r_f_min_signed = 0.0;
    face_min = -1;
    double r_f_min = 1e30;

    const int k = neighbor_count;
    if (k < 2) {
        return;
    }

    const int limit = is_boundary ? k - 1 : k;
    for (int i = 0; i < limit; i++) {
        const Eigen::Vector3d& p_prev = neighbor_points[i];
        const Eigen::Vector3d& p_next = neighbor_points[(i + 1) % k];
        const Eigen::Vector3d p0 = p_next;
        const Eigen::Vector3d p1 = p_v;
        const Eigen::Vector3d p2 = p_prev;
        const Eigen::Vector3d n = face_normal(p0, p1, p2);

        // face directional factor affecting the vertex (incident edges only)
        const double phi0 = phi_ef(q, n, p0, p1, p2, 0);
        const double phi1 = phi_ef(q, n, p0, p1, p2, 1);
        double h0 = H_alpha(phi0, alpha);
        double h1 = H_alpha(phi1, alpha);

        const double r_f = (q - p0).dot(n);
        if (one_sided) {
            const bool outside = outside_face(r_f);
            h0 *= outside;
            h1 *= outside;
        }

        face_term += h0 * h1;
        // is the projection inside the face, determined by Phi^{e_i,f} signs, i= 0,1
        if (phi0 > 0.0 && phi1 > 0.0) {
            const double r_abs = std::abs(r_f);
            // if it is, then compare to the current min distance, and replace if less
            if (r_abs < r_f_min) {
                r_f_min = r_abs;
                face_min = i;
                r_f_min_signed = r_f;
            }
        }
    }

    return;
}

void vertex_edge_term(
    const Eigen::Vector3d& q, const Eigen::Vector3d& p_v,
    const Eigen::Vector3d* neighbor_points, const int neighbor_count, const bool is_boundary,
    const double alpha, const bool one_sided,
    double& edge_term, double& r_e_min, int& edge_min,
    Eigen::Vector3d& P_e_min, Eigen::Vector3d& edge_normal_min)
{
    // This function does two things at once: computes the sum of edge directional factors for the vertex,
    // and along the way computes the closest edge distance and edge  r_e_min, edge_min,  and projection on closest edge
    edge_term = 0.0;
    r_e_min = 1e30;
    edge_min = -1;
    P_e_min = Eigen::Vector3d(0.0, 0.0, 0.0);
    edge_normal_min = Eigen::Vector3d(0.0, 0.0, 0.0);

    const int k = neighbor_count;
    if (k == 0) {
        return;
    }

    for (int i = 0; i < k; i++) {
        const Eigen::Vector3d d_unit = unit_dir(p_v, neighbor_points[i]);
        Eigen::Vector3d P_e;
        double r_e;
        Eigen::Vector3d unit_Pe_to_q;
        edge_projection(q, p_v, d_unit, P_e, r_e, unit_Pe_to_q);

        //  Phi^{v,e} terms
        const double phi_v = phi_ve(q, p_v, neighbor_points[i], d_unit).first;
        double h_v = H_alpha(phi_v, alpha);

        const bool has_prev = (i > 0) || (!is_boundary);
        const bool has_next = (i < k - 1) || (!is_boundary);
        const int prev_idx = (i > 0) ? (i - 1) : (k - 1);
        const int next_idx = (i < k - 1) ? (i + 1) : 0;

        FacePointsPair face_points;
        std::array<int, 2> local_edges;
        int face_count = 0;
        if (has_prev) {
            face_points[face_count] = { neighbor_points[i], p_v, neighbor_points[prev_idx] };
            local_edges[face_count] = 0;
            face_count++;
        }
        if (has_next) {
            face_points[face_count] = {neighbor_points[next_idx], p_v, neighbor_points[i] };
            local_edges[face_count] = 1;
            face_count++;
        }

        std::array<double, 2> phi_ef = { 0.0, 0.0 };
        std::array<double, 2> h_face = { 0.0, 0.0 };
        std::array<double, 2> r_f = { 0.0, 0.0 };
        std::array<Eigen::Vector3d, 2> n_face = {
            Eigen::Vector3d(0.0, 0.0, 0.0),
            Eigen::Vector3d(0.0, 0.0, 0.0),
        };

        for (int j = 0; j < face_count; j++) {
            const auto& face = face_points[j];
            const Eigen::Vector3d n = face_normal(face[0], face[1], face[2]);
            const Eigen::Vector3d edge_inward =
                face_edge_inward(n, face[0], face[1], face[2], local_edges[j]);
            phi_ef[j] = unit_Pe_to_q.dot(edge_inward);
            h_face[j] = H_alpha(phi_ef[j], alpha);
            r_f[j] = (q - face[0]).dot(n);
            if (one_sided) {
                h_face[j] *= outside_face(r_f[j]);
            }
            n_face[j] = n;
        }

        const bool has_f1 = face_count > 1;

        Eigen::Vector3d n_avg = n_face[0];
        if (has_f1) n_avg += n_face[1];
        if (one_sided) {
            const double n_norm = n_avg.norm();
            if (n_norm > kEps) {
                n_avg = n_avg / n_norm;
            }

            const bool outside = outside_edge(
                q, r_e, P_e, n_avg,
                r_f[0], phi_ef[0], has_f1, r_f[1], phi_ef[1]);
            if (!outside) {
                h_v = 0.0;
            }
        }

        // complete part of the edge directional factor to be used for the vertex
        // it is different from the complete factor as it only uses h_v = H^alpha(Phi^{v,e})
        // for this vertex, not both
        edge_term += (1.0 - h_face[0] - h_face[1]) * h_v;

        // is the projection of q to the ray starting at vertex along the edge inside the ray
        if (phi_v > 0.0 && r_e < r_e_min) {
            // replace the distance if projection is inside and the distance is less
            r_e_min = r_e;
            edge_min = i;
            P_e_min = P_e;
            if (one_sided) edge_normal_min = n_avg;
        }
    }

    return;
}

double potential_vertex(
    const Eigen::Vector3d& q, const Eigen::Vector3d& p_v,
    const Eigen::Vector3d* neighbor_points, const int neighbor_count, const bool is_boundary,
    const bool pointed_vertex,
    const PotentialParameters& params)
{
    // potential due to a vertex at point q
    const double r_v = (q - p_v).norm();

    // denominator of the potential has a sum over directional terms over faces and edges computed here
    // these are also needed to determine local sidedeness
    double face_term;
    double r_f_min_signed;
    int face_min;
    double edge_term;
    double r_e_min;
    int edge_min;
    Eigen::Vector3d P_e_min;
    Eigen::Vector3d edge_normal_min;
    vertex_face_term(
        q, p_v, neighbor_points, neighbor_count, is_boundary,
        params.alpha, params.one_sided,
        face_term, r_f_min_signed, face_min);
    vertex_edge_term(
        q, p_v, neighbor_points, neighbor_count, is_boundary,
        params.alpha, params.one_sided,
        edge_term, r_e_min, edge_min, P_e_min, edge_normal_min);

    if (params.one_sided) {
        if (!outside_vertex(
                q,
                r_v,
                r_f_min_signed, face_min,
                r_e_min, edge_min, P_e_min,
                edge_normal_min, pointed_vertex)) {
            return 0.0;
        }
    }

    const double denom = std::pow(r_v, params.p);
    double I_v = (denom <= kEps) ? kSingularValue
                                 : (1.0 - face_term - edge_term) / denom;
    if (params.localized) {
        I_v *= h_epsilon(r_v, params.epsilon);
    }
    return I_v;
}

// ****************************************************************************
// Helper to extract unique edge and vertex lists from a face list

void get_vertices_and_edges(
    const std::vector<int>& face_indices, const PotentialCollisionMesh& mesh,
    std::vector<int>& edge_list, std::vector<int>& vertex_list)
{
    // Go over the list of faces, extract edges and faces,
    // place them in lists of unique edges and faces; not using sets to keep this numba-compatible
    std::vector<char> edge_mark(mesh.num_edges(), 0);
    std::vector<char> vertex_mark(mesh.num_vertices(), 0);

    edge_list.clear();
    vertex_list.clear();
    edge_list.reserve(mesh.num_edges());
    vertex_list.reserve(mesh.num_vertices());

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

double smoothed_offset_potential_point_impl(
    const Eigen::Vector3d& q, const std::vector<int>& face_indices,
    const std::vector<int>& edge_list, const std::vector<int>& vertex_list,
    const PotentialCollisionMesh& mesh,
    const PotentialParameters& params,
    const bool include_faces, const bool include_edges, const bool include_vertices)
{
    // Compute potential from faces,edges and vertices given by the face list face_indices at point q.
    // See smoothed_offset_potential for arguments.
    if (!(include_faces || include_edges || include_vertices)) {
        return 0.0;
    }

    double face_sum = 0.0;
    double edge_sum = 0.0;
    double vertex_sum = 0.0;

    for (const int fidx : face_indices) {
        if (include_faces) {
            const int v0 = mesh.faces()(fidx, 0);
            const int v1 = mesh.faces()(fidx, 1);
            const int v2 = mesh.faces()(fidx, 2);
            const FacePoints face_points = {
                vertex_position(mesh, v0),
                vertex_position(mesh, v1),
                vertex_position(mesh, v2),
            };
            face_sum += potential_face(
                q, face_points,
                params);
        }
    }

    if (include_edges) {
        for (const int edge_idx : edge_list) {
            int f0 = mesh.edges_to_faces()(edge_idx, 0);
            int f1 = mesh.edges_to_faces()(edge_idx, 1);
            if (f0 < 0 && f1 >= 0) {
                std::swap(f0, f1);
            }
            const bool has_f1 = f1 >= 0;
            const int local0 = find_local_edge(mesh, f0, edge_idx);
            const int local1 = has_f1 ? find_local_edge(mesh, f1, edge_idx) : -1;
            FacePointsPair face_points;
            face_points[0] = {
                vertex_position(mesh, mesh.faces()(f0, 0)),
                vertex_position(mesh, mesh.faces()(f0, 1)),
                vertex_position(mesh, mesh.faces()(f0, 2)),
            };
            if (has_f1) {
                face_points[1] = {
                    vertex_position(mesh, mesh.faces()(f1, 0)),
                    vertex_position(mesh, mesh.faces()(f1, 1)),
                    vertex_position(mesh, mesh.faces()(f1, 2)),
                };
            }
            edge_sum += potential_edge(
                q, face_points,
                local0, local1, has_f1,
                params);
        }
    }

    if (include_vertices) {
        for (const int v_idx : vertex_list) {
            const Eigen::Vector3d p_v = vertex_position(mesh, v_idx);
            const auto& edge_list_v = mesh.vertices_to_edges()[static_cast<size_t>(v_idx)];
            if (edge_list_v.size() > 50) {
                throw std::runtime_error("Vertex has more than 50 incident edges.");
            }
            std::array<Eigen::Vector3d, 50> neighbor_points;
            int neighbor_count = 0;
            int boundary_count = 0;
            for (const int edge_idx : edge_list_v) {
                const int a = mesh.edges()(edge_idx, 0);
                const int b = mesh.edges()(edge_idx, 1);
                const int neighbor_idx = (a == v_idx) ? b : a;
                neighbor_points[neighbor_count] = vertex_position(mesh, neighbor_idx);
                neighbor_count++;
                if (mesh.edges_to_faces()(edge_idx, 0) < 0
                    || mesh.edges_to_faces()(edge_idx, 1) < 0) {
                    boundary_count++;
                }
            }
            const bool is_boundary = boundary_count == 2;
            const bool pointed_vertex = mesh.pointed_vertices()[static_cast<size_t>(v_idx)] != 0;
            vertex_sum += potential_vertex(
                q, p_v,
                neighbor_points.data(), neighbor_count, is_boundary,
                pointed_vertex,
                params);
        }
    }

    return face_sum + edge_sum + vertex_sum;
}

// ****************************************************************************
// Main potential calls

double smoothed_offset_potential_point(
    const Eigen::Vector3d& q, const std::vector<int>& face_indices,
    const PotentialCollisionMesh& mesh,
    const double alpha, const double p, const double epsilon,
    const bool include_faces, const bool include_edges, const bool include_vertices,
    const bool localized, const bool one_sided)
{
    const PotentialParameters params{
        alpha,
        p,
        epsilon,
        localized,
        one_sided,
    };

    std::vector<int> edge_list;
    std::vector<int> vertex_list;
    get_vertices_and_edges(
        face_indices, mesh,
        edge_list, vertex_list);
    return smoothed_offset_potential_point_impl(
        q, face_indices,
        edge_list, vertex_list,
        mesh,
        params,
        include_faces, include_edges, include_vertices);
}

} // namespace

Eigen::VectorXd smoothed_offset_potential(
    Eigen::ConstRef<Eigen::MatrixXd> q,
    const PotentialCollisionMesh& mesh,
    double alpha, double p, double epsilon,
    bool include_faces, bool include_edges, bool include_vertices,
    bool localized, bool one_sided)
{
    if (q.cols() != 3) {
        throw std::runtime_error("q must have shape (nq, 3).");
    }

    std::vector<int> face_indices(static_cast<size_t>(mesh.num_faces()));
    for (int i = 0; i < mesh.faces().rows(); i++) {
        face_indices[static_cast<size_t>(i)] = i;
    }
    std::vector<int> edge_list;
    std::vector<int> vertex_list;
    get_vertices_and_edges(
        face_indices, mesh,
        edge_list, vertex_list);
    const PotentialParameters params{
        alpha,
        p,
        epsilon,
        localized,
        one_sided,
    };

    Eigen::VectorXd out(q.rows());
    for (int i = 0; i < q.rows(); i++) {
        const Eigen::Vector3d qi = q.row(i).transpose();
        out[i] = smoothed_offset_potential_point_impl(
            qi, face_indices,
            edge_list, vertex_list,
            mesh,
            params,
            include_faces, include_edges, include_vertices);
    }

    return out;
}

Eigen::VectorXd smoothed_offset_potential_cpp(
    Eigen::ConstRef<Eigen::MatrixXd> q,
    Eigen::ConstRef<Eigen::MatrixXd> V,
    Eigen::ConstRef<Eigen::MatrixXi> F,
    double alpha, double p, double epsilon,
    bool include_faces, bool include_edges, bool include_vertices,
    bool localized, bool one_sided)
{
    PotentialCollisionMesh mesh(V, F);
    return smoothed_offset_potential(
        q,
        mesh,
        alpha, p, epsilon,
        include_faces, include_edges, include_vertices,
        localized, one_sided);
}

} // namespace ipc

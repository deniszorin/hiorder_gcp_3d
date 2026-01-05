#pragma once

#include "potential_collision_mesh.hpp"

#include <Eigen/Dense>

#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ipc {

constexpr double kSingularValue = 1e12;
constexpr double kEps = 1e-12;

template <typename F>
using Vector3 = Eigen::Matrix<F, 3, 1>;
template <typename F>
using FacePointsT = std::array<Vector3<F>, 3>;
template <typename F>
using EdgePointsT = std::array<Vector3<F>, 4>;

using FacePoints = FacePointsT<double>;
using EdgePoints = EdgePointsT<double>;

inline FacePoints face_points_from_rows(const Eigen::Matrix<double, 3, 3>& face_points)
{
    return {
        face_points.row(0).transpose(),
        face_points.row(1).transpose(),
        face_points.row(2).transpose(),
    };
}

inline EdgePoints edge_points_from_rows(const Eigen::Matrix<double, 4, 3>& edge_points)
{
    return {
        edge_points.row(0).transpose(),
        edge_points.row(1).transpose(),
        edge_points.row(2).transpose(),
        edge_points.row(3).transpose(),
    };
}

inline double to_passive_value(const double value) { return value; }

template <typename F>
struct PassiveValue {
    static double get(const F& value)
    {
        return static_cast<double>(value);
    }
};

template <typename F>
inline double to_passive_value(const F& value)
{
    return PassiveValue<F>::get(value);
}

template <typename F>
inline F make_constant(const double value, const F& ref)
{
    return ref * 0.0 + value;
}

template <typename F>
inline Vector3<F> make_zero_vec(const F& ref)
{
    const F zero = make_constant(0.0, ref);
    Vector3<F> out;
    out[0] = zero;
    out[1] = zero;
    out[2] = zero;
    return out;
}

template <typename F>
inline Eigen::Vector3d to_passive_vec(const Vector3<F>& vec)
{
    return Eigen::Vector3d(
        to_passive_value(vec[0]),
        to_passive_value(vec[1]),
        to_passive_value(vec[2]));
}

template <typename DerivedA, typename DerivedB>
inline auto dot3(const Eigen::MatrixBase<DerivedA>& a, const Eigen::MatrixBase<DerivedB>& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template <typename DerivedA, typename DerivedB>
inline auto cross3(const Eigen::MatrixBase<DerivedA>& a, const Eigen::MatrixBase<DerivedB>& b)
{
    using Scalar = decltype(a[0] * b[0]);
    Vector3<Scalar> out;
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
    return out;
}

// ****************************************************************************
// Basic geometry functions

template <typename Derived>
inline auto norm3(const Eigen::MatrixBase<Derived>& v)
{
    using std::sqrt;
    return sqrt(dot3(v, v));
}

template <typename F>
inline F safe_norm(const Vector3<F>& v)
{
    const F n = norm3(v);
    return n < kEps ? make_constant(kEps, n) : n;
}

template <typename F>
inline Vector3<F> unit_vec(const Vector3<F>& v)
{
    return v / safe_norm(v);
}

template <typename F>
inline Vector3<F> unit_dir(const Vector3<F>& p0, const Vector3<F>& p1)
{
    const Vector3<F> diff = p1 - p0;
    return unit_vec(diff);
}

template <typename F>
inline Vector3<F> face_normal(
    const Vector3<F>& p0, const Vector3<F>& p1, const Vector3<F>& p2)
{
    const Vector3<F> n = cross3(p1 - p0, p2 - p0);
    return n / safe_norm(n);
}

template <typename F>
inline void face_edge_endpoints(
    const Vector3<F>& p0, const Vector3<F>& p1, const Vector3<F>& p2,
    const int local_edge, Vector3<F>& edge_p0, Vector3<F>& edge_p1)
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

template <typename F>
inline Vector3<F> face_edge_inward(
    const Vector3<F>& n,
    const Vector3<F>& p0, const Vector3<F>& p1, const Vector3<F>& p2,
    const int local_edge)
{
    Vector3<F> edge_p0;
    Vector3<F> edge_p1;
    face_edge_endpoints<F>(p0, p1, p2, local_edge, edge_p0, edge_p1);
    const Vector3<F> d_e = unit_dir(edge_p0, edge_p1);
    return cross3(n, d_e);
}

template <typename F>
inline void edge_projection(
    const Vector3<F>& q, const Vector3<F>& p0, const Vector3<F>& d_unit,
    Vector3<F>& P_e, F& r_e, Vector3<F>& unit_Pe_to_q)
{
    // the choice of the direction on the edge does not affect
    // projected position P_e can use any
    const F t = dot3(q - p0, d_unit);
    P_e = p0 + t * d_unit;
    // P_e: projection of q to each edge line.
    // r_e: distance from q to each edge line.
    const Vector3<F> diff = q - P_e;
    r_e = norm3(diff);
    unit_Pe_to_q = diff / safe_norm(diff);
}

template <typename F>
inline void project_point_to_segment(
    const Vector3<F>& q, const Vector3<F>& p0, const Vector3<F>& p1,
    Vector3<F>& proj, F& r, bool& inside)
{
    const Vector3<F> d = p1 - p0;
    const F denom = dot3(d, d);
    const F t = dot3(q - p0, d) / denom;
    const double t_passive = to_passive_value(t);
    if (t_passive <= 0.0) {
        proj = p0;
        inside = false;
    } else if (t_passive >= 1.0) {
        proj = p1;
        inside = false;
    } else {
        proj = p0 + t * d;
        inside = true;
    }
    r = norm3(q - proj);
}

template <typename DerivedA, typename DerivedB>
inline auto distance_to_edge_line(
    const Eigen::MatrixBase<DerivedA>& q_minus_p,
    const Eigen::MatrixBase<DerivedB>& edge)
{
    const auto t = dot3(q_minus_p, edge) / dot3(edge, edge);
    return norm3(q_minus_p - t * edge);
}

template <typename F>
inline F triangle_distance(
    const Vector3<F>& q,
    const Vector3<F>& p0, const Vector3<F>& p1, const Vector3<F>& p2)
{
    const Vector3<F> e0 = p1 - p0;
    const Vector3<F> e1 = p2 - p1;
    const Vector3<F> e2 = p0 - p2;
    const Vector3<F> n = cross3(e0, -e2);
    const F n_norm = norm3(n);
    if (to_passive_value(n_norm) <= kEps) {
        const F d0 = norm3(q - p0);
        const F d1 = norm3(q - p1);
        const F d2 = norm3(q - p2);
        F d_min = d0;
        if (to_passive_value(d1) < to_passive_value(d_min)) {
            d_min = d1;
        }
        if (to_passive_value(d2) < to_passive_value(d_min)) {
            d_min = d2;
        }
        return d_min;
    } else {
        const F signed_dist = dot3(n, q - p0);
        const Vector3<F> q_proj = q - (signed_dist / (n_norm * n_norm)) * n;
        using std::abs;
        const F r_f = abs(signed_dist) / n_norm;
        const Vector3<F> qp0 = q_proj - p0;
        const Vector3<F> qp1 = q_proj - p1;
        const Vector3<F> qp2 = q_proj - p2;

        // side tests for edge halfplanes, > 0 = inside the triangle
        const F s0 = dot3(qp0, cross3(n, e0));
        const F s1 = dot3(qp1, cross3(n, e1));
        const F s2 = dot3(qp2, cross3(n, e2));

        // "slab" tests for each edge, is a point on the same side as the edge
        // with respect to perpendicular trough the edge endpoints
        const F t0_0 = dot3(qp0, e0);
        const F t0_1 = dot3(qp1, e1);
        const F t0_2 = dot3(qp2, e2);
        const F t1_0 = dot3(qp1, -e0);
        const F t1_1 = dot3(qp2, -e1);
        const F t1_2 = dot3(qp0, -e2);

        // 7 regions, triangle interior, closest to one of edges, closest to one of vertices
        if (to_passive_value(s0) >= -kEps
            && to_passive_value(s1) >= -kEps
            && to_passive_value(s2) >= -kEps) {
            return r_f;
        }

        if (to_passive_value(s0) < 0.0
            && to_passive_value(t0_0) >= -kEps
            && to_passive_value(t1_0) >= -kEps) {
            return distance_to_edge_line(q - p0, e0);
        }
        if (to_passive_value(s1) < 0.0
            && to_passive_value(t0_1) >= -kEps
            && to_passive_value(t1_1) >= -kEps) {
            return distance_to_edge_line(q - p1, e1);
        }
        if (to_passive_value(s2) < 0.0
            && to_passive_value(t0_2) >= -kEps
            && to_passive_value(t1_2) >= -kEps) {
            return distance_to_edge_line(q - p2, e2);
        }

        if (to_passive_value(t0_0) < 0.0 && to_passive_value(t1_2) < 0.0) {
            return norm3(q - p0);
        }
        if (to_passive_value(t0_1) < 0.0 && to_passive_value(t1_0) < 0.0) {
            return norm3(q - p1);
        }
        if (to_passive_value(t0_2) < 0.0 && to_passive_value(t1_1) < 0.0) {
            return norm3(q - p2);
        }

        throw std::runtime_error("triangle_distance: region classification failed.");
    }
}

template <typename F>
inline Vector3<F> edge_normal_from_faces(
    const FacePointsT<F>& f0, const FacePointsT<F>& f1, const bool has_f1)
{
    Vector3<F> n_avg = face_normal(f0[0], f0[1], f0[2]);
    if (has_f1) {
        n_avg += face_normal(f1[0], f1[1], f1[2]);
    }
    const F n_norm = norm3(n_avg);
    if (n_norm > kEps) {
        n_avg = n_avg / n_norm;
    }
    return n_avg;
}

// ****************************************************************************
// Functions to check if a point is outside a given face, edge, vertex

inline bool outside_face(const double r_f)
{
    // Assumes that signed distance to the face is given (computed elsewhere) and returns if it is positive
    return r_f > 0.0;
}

inline bool outside_edge(
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

inline bool outside_vertex(
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
// Helpers for local to global and back vertex/edge index conversion

inline Eigen::Vector3d vertex_position(const PotentialCollisionMesh& mesh, const int idx)
{
    return mesh.rest_positions().row(idx).transpose();
}

template <typename F>
inline Vector3<F> vertex_position(const PotentialCollisionMesh& mesh, const int idx)
{
    return mesh.rest_positions().row(idx).transpose().template cast<F>();
}

inline int find_local_edge(const PotentialCollisionMesh& mesh, const int fidx, const int edge_idx)
{
    const Eigen::MatrixXi& face_edges = mesh.faces_to_edges();
    for (int i = 0; i < face_edges.cols(); i++) {
        if (face_edges(fidx, i) == edge_idx) {
            return i;
        }
    }
    throw std::runtime_error("Edge not found in face.");
}

inline void get_vertices_and_edges(
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

inline std::vector<int> build_edge_valence(const PotentialCollisionMesh& mesh)
{
    std::vector<int> edge_valence(mesh.num_edges(), 0);
    for (int e = 0; e < mesh.edges_to_faces().rows(); e++) {
        edge_valence[static_cast<size_t>(e)] += mesh.edges_to_faces()(e, 0) >= 0;
        edge_valence[static_cast<size_t>(e)] += mesh.edges_to_faces()(e, 1) >= 0;
    }
    return edge_valence;
}

inline std::vector<char> build_vertex_internal(const PotentialCollisionMesh& mesh)
{
    std::vector<char> vertex_internal(mesh.num_vertices(), 1);
    for (int e = 0; e < mesh.edges().rows(); e++) {
        if (mesh.edges_to_faces()(e, 0) < 0 || mesh.edges_to_faces()(e, 1) < 0) {
            const int v0 = mesh.edges()(e, 0);
            const int v1 = mesh.edges()(e, 1);
            vertex_internal[static_cast<size_t>(v0)] = 0;
            vertex_internal[static_cast<size_t>(v1)] = 0;
        }
    }
    return vertex_internal;
}

} // namespace ipc

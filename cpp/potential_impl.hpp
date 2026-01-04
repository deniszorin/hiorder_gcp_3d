#pragma once

#include "potential_cpp.hpp"

#include <Eigen/Dense>

#include <array>
#include <cmath>
#include <utility>

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
// Basic geometry functions for numba to avoid function calls

template <typename F>
inline F safe_norm(const Vector3<F>& v)
{
    using std::sqrt;
    const F n = sqrt(dot3(v, v));
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
    using std::sqrt;
    r_e = sqrt(dot3(diff, diff));
    unit_Pe_to_q = diff / safe_norm(diff);
}

// ****************************************************************************
// Potential blending/localization functions

template <typename F>
inline F H(const F& z)
{
    if (z < -1.0) {
        return make_constant(0.0, z);
    }
    if (z > 1.0) {
        return make_constant(1.0, z);
    }
    return ((2.0 - z) * (z + 1.0) * (z + 1.0)) / 4.0;
}

template <typename F>
inline F H_alpha(const F& t, const double alpha)
{
    return H(t / alpha);
}

template <typename F>
inline F h_local(const F& z)
{
    if (z > 1.0) {
        return make_constant(0.0, z);
    }
    return (2.0 * z + 1.0) * (z - 1.0) * (z - 1.0);
}

template <typename F>
inline F h_epsilon(const F& z, const double epsilon)
{
    return h_local(z / epsilon);
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
//  Potential directional terms Phi^{e,f}, Phi^{v,e}

template <typename F>
inline F phi_ef(
    const Vector3<F>& q,
    const Vector3<F>& n,
    const Vector3<F>& p0, const Vector3<F>& p1, const Vector3<F>& p2,
    const int local_edge)
{
    Vector3<F> edge_p0;
    Vector3<F> edge_p1;
    face_edge_endpoints<F>(p0, p1, p2, local_edge, edge_p0, edge_p1);

    const Vector3<F> d_unit = unit_dir(edge_p0, edge_p1);
    Vector3<F> P_e;
    F r_e;
    Vector3<F> unit_Pe_to_q;
    edge_projection(q, edge_p0, d_unit, P_e, r_e, unit_Pe_to_q);

    // Phi^{e,f} := (q - P_e)_+ dot (n x d_e[i]) per face
    const Vector3<F> edge_inward = face_edge_inward(n, p0, p1, p2, local_edge);
    return dot3(unit_Pe_to_q, edge_inward);
}

template <typename F>
inline std::pair<F, F> phi_ve(
    const Vector3<F>& q,
    const Vector3<F>& p0, const Vector3<F>& p1,
    const Vector3<F>& d_unit)
{
    const Vector3<F> unit0 = unit_dir(p0, q);
    const Vector3<F> unit1 = unit_dir(p1, q);

    // Phi^{i,e}, i=0,1 factors per edge
    const F phi0 = dot3(unit0, d_unit);
    const F phi1 = dot3(unit1, -d_unit);
    return std::make_pair(phi0, phi1);
}

template <typename F>
inline Vector3<F> edge_normal_from_faces(
    const FacePointsT<F>& f0, const FacePointsT<F>& f1, const bool has_f1)
{
    Vector3<F> n_avg = face_normal(f0[0], f0[1], f0[2]);
    if (has_f1) {
        n_avg += face_normal(f1[0], f1[1], f1[2]);
    }
    using std::sqrt;
    const F n_norm = sqrt(dot3(n_avg, n_avg));
    if (n_norm > kEps) {
        n_avg = n_avg / n_norm;
    }
    return n_avg;
}

// ****************************************************************************
// Potential evaluation, face, edge, vertex components

template <typename F>
inline F potential_face(
    const Vector3<F>& q, const FacePointsT<F>& face_points,
    const PotentialParameters& params)
{
    const auto& p0 = face_points[0];
    const auto& p1 = face_points[1];
    const auto& p2 = face_points[2];
    const Vector3<F> n = face_normal(p0, p1, p2);

    // signed distance to the face plane.
    const F r_f = dot3(q - p0, n);
    using std::abs;
    const F r_f_abs = abs(r_f);

    F B = make_constant(1.0, r_f);
    for (int local_edge = 0; local_edge < 3; local_edge++) {
        const F phi = phi_ef(q, n, p0, p1, p2, local_edge);
        B *= H_alpha(phi, params.alpha);
    }

    using std::pow;
    const F denom = pow(r_f_abs, params.p);
    F I_f = (denom <= kEps) ? make_constant(kSingularValue, denom) : (B / denom);
    if (params.one_sided && !outside_face(to_passive_value(r_f))) {
        I_f = make_constant(0.0, I_f);
    }
    if (params.localized) {
        I_f *= h_epsilon(r_f_abs, params.epsilon);
    }
    return I_f;
}

template <typename F>
inline F potential_edge(
    const Vector3<F>& q, const EdgePointsT<F>& edge_points, const bool has_f1,
    const PotentialParameters& params)
{
    const Vector3<F> edge_p0 = edge_points[0];
    const Vector3<F> edge_p1 = edge_points[1];
    const Vector3<F> f0_p0 = edge_p0;
    const Vector3<F> f0_p1 = edge_p1;
    const Vector3<F> f0_p2 = edge_points[2];
    const Vector3<F> d_unit = unit_dir(edge_p0, edge_p1);
    Vector3<F> P_e;
    F r_e;
    Vector3<F> unit_Pe_to_q;
    edge_projection(q, edge_p0, d_unit, P_e, r_e, unit_Pe_to_q);
    const auto phi_pair = phi_ve(q, edge_p0, edge_p1, d_unit);
    const F phi0 = phi_pair.first;
    const F phi1 = phi_pair.second;

    const Vector3<F> n0 = face_normal(f0_p0, f0_p1, f0_p2);
    const Vector3<F> edge_inward_0 =
        face_edge_inward(n0, f0_p0, f0_p1, f0_p2, 0);
    const F phi_ef0 = dot3(unit_Pe_to_q, edge_inward_0);
    F h_face_0 = H_alpha(phi_ef0, params.alpha);
    const F r_f0 = dot3(q - f0_p0, n0);
    if (params.one_sided) {
        h_face_0 *= outside_face(to_passive_value(r_f0));
    }

    const F zero = make_constant(0.0, r_e);
    F h_face_1 = zero;
    F phi_ef1 = zero;
    F r_f1 = zero;
    if (has_f1) {
        const Vector3<F> f1_p0 = edge_p1;
        const Vector3<F> f1_p1 = edge_p0;
        const Vector3<F> f1_p2 = edge_points[3];
        const Vector3<F> n1 = face_normal(f1_p0, f1_p1, f1_p2);
        const Vector3<F> edge_inward_1 =
            face_edge_inward(n1, f1_p0, f1_p1, f1_p2, 0);
        phi_ef1 = dot3(unit_Pe_to_q, edge_inward_1);
        h_face_1 = H_alpha(phi_ef1, params.alpha);
        r_f1 = dot3(q - f1_p0, n1);
        if (params.one_sided) {
            h_face_1 *= outside_face(to_passive_value(r_f1));
        }
    }

    const F one = make_constant(1.0, h_face_0);
    const F B_edge = (one - h_face_0 - h_face_1)
        * H_alpha(phi0, params.alpha) * H_alpha(phi1, params.alpha);

    // distance to edge r_e already computed per edge
    using std::pow;
    const F denom = pow(r_e, params.p);
    F I_e = (denom <= kEps) ? make_constant(kSingularValue, denom) : (B_edge / denom);

    if (params.one_sided) {
        FacePointsT<F> f0 = { f0_p0, f0_p1, f0_p2 };
        FacePointsT<F> f1 = { f0_p0, f0_p1, f0_p2 };
        if (has_f1) {
            f1 = { edge_p1, edge_p0, edge_points[3] };
        }
        const Vector3<F> edge_n = edge_normal_from_faces(f0, f1, has_f1);
        const double r_f0_passive = to_passive_value(r_f0);
        const double phi_ef0_passive = to_passive_value(phi_ef0);
        const double r_f1_passive = to_passive_value(r_f1);
        const double phi_ef1_passive = to_passive_value(phi_ef1);
        const bool outside = outside_edge(
            to_passive_vec(q),
            to_passive_value(r_e), to_passive_vec(P_e), to_passive_vec(edge_n),
            r_f0_passive, phi_ef0_passive, has_f1, r_f1_passive, phi_ef1_passive);
        if (!outside) {
            I_e = make_constant(0.0, I_e);
        }
    }
    if (params.localized) {
        I_e *= h_epsilon(r_e, params.epsilon);
    }
    return I_e;
}

template <typename F>
inline void vertex_face_term(
    const Vector3<F>& q, const Vector3<F>& p_v,
    const Vector3<F>* neighbor_points, const int neighbor_count, const bool is_boundary,
    const double alpha, const bool one_sided,
    F& face_term, F& r_f_min_signed, int& face_min)
{
    // get closest face sector and edge ray if any
    // initialize to inf
    // local index of closest face
    const F zero = make_constant(0.0, q[0]);
    const F big = make_constant(1e30, q[0]);
    face_term = zero;
    r_f_min_signed = zero;
    face_min = -1;
    F r_f_min = big;

    const int k = neighbor_count;
    if (k < 2) {
        return;
    }

    const int limit = is_boundary ? k - 1 : k;
    for (int i = 0; i < limit; i++) {
        const Vector3<F>& p_prev = neighbor_points[i];
        const Vector3<F>& p_next = neighbor_points[(i + 1) % k];
        const Vector3<F> p0 = p_next;
        const Vector3<F> p1 = p_v;
        const Vector3<F> p2 = p_prev;
        const Vector3<F> n = face_normal(p0, p1, p2);

        // face directional factor affecting the vertex (incident edges only)
        const F phi0 = phi_ef(q, n, p0, p1, p2, 0);
        const F phi1 = phi_ef(q, n, p0, p1, p2, 1);
        F h0 = H_alpha(phi0, alpha);
        F h1 = H_alpha(phi1, alpha);

        const F r_f = dot3(q - p0, n);
        if (one_sided) {
            const bool outside = outside_face(to_passive_value(r_f));
            h0 *= outside;
            h1 *= outside;
        }

        face_term += h0 * h1;
        // is the projection inside the face, determined by Phi^{e_i,f} signs, i= 0,1
        if (phi0 > 0.0 && phi1 > 0.0) {
            using std::abs;
            const F r_abs = abs(r_f);
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

template <typename F>
inline void vertex_edge_term(
    const Vector3<F>& q, const Vector3<F>& p_v,
    const Vector3<F>* neighbor_points, const int neighbor_count, const bool is_boundary,
    const double alpha, const bool one_sided,
    F& edge_term, F& r_e_min, int& edge_min,
    Vector3<F>& P_e_min, Vector3<F>& edge_normal_min)
{
    // This function does two things at once: computes the sum of edge directional factors for the vertex,
    // and along the way computes the closest edge distance and edge  r_e_min, edge_min,  and projection on closest edge
    const F zero = make_constant(0.0, q[0]);
    const F big = make_constant(1e30, q[0]);
    edge_term = zero;
    r_e_min = big;
    edge_min = -1;
    P_e_min = make_zero_vec(q[0]);
    edge_normal_min = make_zero_vec(q[0]);

    const int k = neighbor_count;
    if (k == 0) {
        return;
    }

    for (int i = 0; i < k; i++) {
        const Vector3<F> d_unit = unit_dir(p_v, neighbor_points[i]);
        Vector3<F> P_e;
        F r_e;
        Vector3<F> unit_Pe_to_q;
        edge_projection(q, p_v, d_unit, P_e, r_e, unit_Pe_to_q);

        //  Phi^{v,e} terms
        const F phi_v = phi_ve(q, p_v, neighbor_points[i], d_unit).first;
        F h_v = H_alpha(phi_v, alpha);

        const bool has_prev = (i > 0) || (!is_boundary);
        const bool has_next = (i < k - 1) || (!is_boundary);
        const int prev_idx = (i > 0) ? (i - 1) : (k - 1);
        const int next_idx = (i < k - 1) ? (i + 1) : 0;

        std::array<FacePointsT<F>, 2> face_points;
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

        std::array<F, 2> phi_ef = { zero, zero };
        std::array<F, 2> h_face = { zero, zero };
        std::array<F, 2> r_f = { zero, zero };
        std::array<Vector3<F>, 2> n_face = {
            make_zero_vec(q[0]),
            make_zero_vec(q[0]),
        };

        for (int j = 0; j < face_count; j++) {
            const auto& face = face_points[j];
            const Vector3<F> n = face_normal(face[0], face[1], face[2]);
            const Vector3<F> edge_inward =
                face_edge_inward(n, face[0], face[1], face[2], local_edges[j]);
            phi_ef[j] = dot3(unit_Pe_to_q, edge_inward);
            h_face[j] = H_alpha(phi_ef[j], alpha);
            r_f[j] = dot3(q - face[0], n);
            if (one_sided) {
                h_face[j] *= outside_face(to_passive_value(r_f[j]));
            }
            n_face[j] = n;
        }

        const bool has_f1 = face_count > 1;

        Vector3<F> n_avg = n_face[0];
        if (has_f1) {
            n_avg += n_face[1];
        }
        if (one_sided) {
            using std::sqrt;
            const F n_norm = sqrt(dot3(n_avg, n_avg));
            if (n_norm > kEps) {
                n_avg = n_avg / n_norm;
            }

            const bool outside = outside_edge(
                to_passive_vec(q),
                to_passive_value(r_e), to_passive_vec(P_e), to_passive_vec(n_avg),
                to_passive_value(r_f[0]), to_passive_value(phi_ef[0]), has_f1,
                to_passive_value(r_f[1]), to_passive_value(phi_ef[1]));
            if (!outside) {
                h_v = zero;
            }
        }

        // complete part of the edge directional factor to be used for the vertex
        // it is different from the complete factor as it only uses h_v = H^alpha(Phi^{v,e})
        // for this vertex, not both
        edge_term += (make_constant(1.0, h_v) - h_face[0] - h_face[1]) * h_v;

        // is the projection of q to the ray starting at vertex along the edge inside the ray
        if (phi_v > 0.0 && r_e < r_e_min) {
            // replace the distance if projection is inside and the distance is less
            r_e_min = r_e;
            edge_min = i;
            P_e_min = P_e;
            if (one_sided) {
                edge_normal_min = n_avg;
            }
        }
    }

    return;
}

template <typename F>
inline F potential_vertex(
    const Vector3<F>& q, const Vector3<F>& p_v,
    const Vector3<F>* neighbor_points, const int neighbor_count, const bool is_boundary,
    const bool pointed_vertex,
    const PotentialParameters& params)
{
    // potential due to a vertex at point q
    using std::sqrt;
    const F r_v = sqrt(dot3(q - p_v, q - p_v));

    // denominator of the potential has a sum over directional terms over faces and edges computed here
    // these are also needed to determine local sidedeness
    F face_term;
    F r_f_min_signed;
    int face_min;
    F edge_term;
    F r_e_min;
    int edge_min;
    Vector3<F> P_e_min;
    Vector3<F> edge_normal_min;
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
                to_passive_vec(q),
                to_passive_value(r_v),
                to_passive_value(r_f_min_signed), face_min,
                to_passive_value(r_e_min), edge_min, to_passive_vec(P_e_min),
                to_passive_vec(edge_normal_min), pointed_vertex)) {
            return make_constant(0.0, r_v);
        }
    }

    using std::pow;
    const F denom = pow(r_v, params.p);
    const F one = make_constant(1.0, denom);
    F I_v = (denom <= kEps) ? make_constant(kSingularValue, denom)
                            : (one - face_term - edge_term) / denom;
    if (params.localized) {
        I_v *= h_epsilon(r_v, params.epsilon);
    }
    return I_v;
}

} // namespace ipc

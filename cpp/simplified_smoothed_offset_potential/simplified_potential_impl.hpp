#pragma once

#include "geometry_connectivity.hpp"
#include "potential_parameters.hpp"

#include <vector>

namespace ipc {

// ****************************************************************************
// Simplified potential evaluation, face, edge, vertex components

template <typename F>
inline F simplified_potential_face(
    const Vector3<F>& q, const FacePointsT<F>& face_points,
    const PotentialParameters& params)
{
    const F r_f = triangle_distance(q, face_points[0], face_points[1], face_points[2]);
    using std::pow;
    const F denom = pow(r_f, params.p);
    const F one = make_constant(1.0, denom);
    F I_f = (denom <= kEps) ? make_constant(kSingularValue, denom) : (one / denom);
    if (params.localized) {
        I_f *= h_epsilon(r_f, params.epsilon);
    }
    return I_f;
}

template <typename F>
inline F simplified_potential_edge(
    const Vector3<F>& q, const Vector3<F>& p0, const Vector3<F>& p1,
    const PotentialParameters& params)
{
    Vector3<F> proj;
    F r_e;
    bool inside = false;
    project_point_to_segment(q, p0, p1, proj, r_e, inside);
    using std::pow;
    const F denom = pow(r_e, params.p);
    const F one = make_constant(1.0, denom);
    F I_e = (denom <= kEps) ? make_constant(kSingularValue, denom) : (one / denom);
    if (params.localized) {
        I_e *= h_epsilon(r_e, params.epsilon);
    }
    return I_e;
}

template <typename F>
inline F simplified_potential_vertex(
    const Vector3<F>& q, const Vector3<F>& p_v,
    const PotentialParameters& params)
{
    const F r_v = norm3(q - p_v);
    using std::pow;
    const F denom = pow(r_v, params.p);
    const F one = make_constant(1.0, denom);
    F I_v = (denom <= kEps) ? make_constant(kSingularValue, denom) : (one / denom);
    if (params.localized) {
        I_v *= h_epsilon(r_v, params.epsilon);
    }
    return I_v;
}

template <typename F>
inline F simplified_smoothed_offset_potential_point_impl(
    const Vector3<F>& q,
    const std::vector<int>& face_indices,
    const std::vector<int>& edge_list,
    const std::vector<int>& vertex_list,
    const PotentialCollisionMesh& mesh,
    const PotentialParameters& params,
    const std::vector<int>& edge_valence,
    const std::vector<char>& vertex_internal,
    const bool include_faces, const bool include_edges, const bool include_vertices)
{
    if (!(include_faces || include_edges || include_vertices)) {
        return make_constant(0.0, q[0]);
    }

    F total = make_constant(0.0, q[0]);

    if (include_faces) {
        for (const int fidx : face_indices) {
            const int v0 = mesh.faces()(fidx, 0);
            const int v1 = mesh.faces()(fidx, 1);
            const int v2 = mesh.faces()(fidx, 2);
            const FacePointsT<F> face_points = {
                vertex_position<F>(mesh, v0),
                vertex_position<F>(mesh, v1),
                vertex_position<F>(mesh, v2),
            };
            total += simplified_potential_face(q, face_points, params);
        }
    }

    if (include_edges) {
        for (const int edge_idx : edge_list) {
            const int weight = edge_valence[static_cast<size_t>(edge_idx)] - 1;
            if (weight == 0) {
                continue;
            }
            const int v0 = mesh.edges()(edge_idx, 0);
            const int v1 = mesh.edges()(edge_idx, 1);
            const Vector3<F> p0 = vertex_position<F>(mesh, v0);
            const Vector3<F> p1 = vertex_position<F>(mesh, v1);
            total -= static_cast<double>(weight)
                * simplified_potential_edge(q, p0, p1, params);
        }
    }

    if (include_vertices) {
        for (const int v_idx : vertex_list) {
            if (vertex_internal[static_cast<size_t>(v_idx)] != 0) {
                total += simplified_potential_vertex(
                    q, vertex_position<F>(mesh, v_idx), params);
            }
        }
    }

    return total;
}

} // namespace ipc

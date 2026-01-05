#include "potential_impl.hpp"

#include <TinyAD/Scalar.hh>

#include <stdexcept>
#include <vector>

namespace ipc {

template <
    int k, typename PassiveT, bool with_hessian,
    int hess_row_start, int hess_col_start, int hess_rows, int hess_cols>
struct PassiveValue<
    TinyAD::Scalar<k, PassiveT, with_hessian, hess_row_start, hess_col_start, hess_rows, hess_cols>> {
    static double get(
        const TinyAD::Scalar<k, PassiveT, with_hessian, hess_row_start, hess_col_start, hess_rows, hess_cols>& value)
    {
        return TinyAD::to_passive(value);
    }
};

namespace {

template <typename F>
inline Vector3<F> cast_vec3(const Eigen::Vector3d& vec)
{
    return vec.template cast<F>();
}

template <typename F>
inline FacePointsT<F> cast_face_points(const FacePoints& face_points)
{
    return {
        cast_vec3<F>(face_points[0]),
        cast_vec3<F>(face_points[1]),
        cast_vec3<F>(face_points[2]),
    };
}

template <typename F>
inline EdgePointsT<F> cast_edge_points(const EdgePoints& edge_points)
{
    return {
        cast_vec3<F>(edge_points[0]),
        cast_vec3<F>(edge_points[1]),
        cast_vec3<F>(edge_points[2]),
        cast_vec3<F>(edge_points[3]),
    };
}

} // namespace

std::pair<double, Eigen::Vector3d> potential_face_cpp_tinyad(
    const Eigen::Vector3d& q,
    const Eigen::Matrix<double, 3, 3>& face_points,
    const PotentialParameters& params)
{
    using AD = TinyAD::Scalar<3, double, false>;
    const FacePoints face = face_points_from_rows(face_points);
    const FacePointsT<AD> face_ad = cast_face_points<AD>(face);
    const Vector3<AD> q_ad = AD::make_active({q[0], q[1], q[2]});
    const AD value_ad = potential_face(q_ad, face_ad, params);
    return std::make_pair(TinyAD::to_passive(value_ad), value_ad.grad);
}

std::pair<double, Eigen::Vector3d> potential_edge_cpp_tinyad(
    const Eigen::Vector3d& q,
    const Eigen::Matrix<double, 4, 3>& edge_points,
    const bool has_f1,
    const PotentialParameters& params)
{
    using AD = TinyAD::Scalar<3, double, false>;
    const EdgePoints edge_points_arr = edge_points_from_rows(edge_points);
    const EdgePointsT<AD> edge_ad = cast_edge_points<AD>(edge_points_arr);
    const Vector3<AD> q_ad = AD::make_active({q[0], q[1], q[2]});
    const AD value_ad =
        potential_edge(q_ad, edge_ad, has_f1, params);
    return std::make_pair(TinyAD::to_passive(value_ad), value_ad.grad);
}

std::pair<double, Eigen::Vector3d> potential_vertex_cpp_tinyad(
    const Eigen::Vector3d& q, const Eigen::Vector3d& p_v,
    Eigen::ConstRef<Eigen::MatrixXd> neighbor_points, const bool is_boundary,
    const bool pointed_vertex,
    const PotentialParameters& params)
{
    if (neighbor_points.cols() != 3) {
        throw std::runtime_error("neighbor_points must have shape (k, 3).");
    }
    if (neighbor_points.rows() > 50) {
        throw std::runtime_error("Vertex has more than 50 incident edges.");
    }
    using AD = TinyAD::Scalar<3, double, false>;
    std::vector<Vector3<AD>> neighbor_list;
    neighbor_list.reserve(neighbor_points.rows());
    for (int i = 0; i < neighbor_points.rows(); i++) {
        neighbor_list.push_back(cast_vec3<AD>(neighbor_points.row(i).transpose()));
    }
    const Vector3<AD> q_ad = AD::make_active({q[0], q[1], q[2]});
    const Vector3<AD> p_v_ad = cast_vec3<AD>(p_v);
    const AD value_ad = potential_vertex(
        q_ad, p_v_ad,
        neighbor_list.data(), static_cast<int>(neighbor_list.size()), is_boundary,
        pointed_vertex,
        params);
    return std::make_pair(TinyAD::to_passive(value_ad), value_ad.grad);
}

} // namespace ipc

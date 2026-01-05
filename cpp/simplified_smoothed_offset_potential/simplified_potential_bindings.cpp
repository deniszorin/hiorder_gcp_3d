#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "simplified_potential.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_simplified_potential_cpp, m)
{
    const py::module_ core = py::module_::import("_potential_cpp");
    m.attr("PotentialParameters") = core.attr("PotentialParameters");

    m.def(
        "simplified_smoothed_offset_potential_cpp",
        [](
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
            bool one_sided) {
            ipc::PotentialCollisionMesh mesh(V, F);
            const ipc::PotentialParameters params{
                alpha,
                p,
                epsilon,
                localized,
                one_sided,
            };
            return ipc::simplified_smoothed_offset_potential(
                q,
                mesh,
                params,
                include_faces,
                include_edges,
                include_vertices);
        },
        py::arg("q"),
        py::arg("V"),
        py::arg("F"),
        py::arg("alpha") = 0.1,
        py::arg("p") = 2.0,
        py::arg("epsilon") = 0.1,
        py::arg("include_faces") = true,
        py::arg("include_edges") = true,
        py::arg("include_vertices") = true,
        py::arg("localized") = false,
        py::arg("one_sided") = false,
        "Compute simplified potential using the C++ implementation.");

    m.def(
        "simplified_smoothed_offset_potential_accelerated_cpp",
        [](
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
            bool one_sided) {
            return ipc::simplified_smoothed_offset_potential_accelerated_cpp(
                q,
                V,
                F,
                alpha,
                p,
                epsilon,
                include_faces,
                include_edges,
                include_vertices,
                localized,
                one_sided);
        },
        py::arg("q"),
        py::arg("V"),
        py::arg("F"),
        py::arg("alpha") = 0.1,
        py::arg("p") = 2.0,
        py::arg("epsilon") = 0.1,
        py::arg("include_faces") = true,
        py::arg("include_edges") = true,
        py::arg("include_vertices") = true,
        py::arg("localized") = false,
        py::arg("one_sided") = false,
        "Compute simplified potential using the C++ accelerated implementation.");

    m.def(
        "simplified_smoothed_offset_potential_cpp_tinyad",
        [](
            const Eigen::Vector3d& q,
            Eigen::ConstRef<Eigen::MatrixXd> V,
            Eigen::ConstRef<Eigen::MatrixXi> F,
            double alpha,
            double p,
            double epsilon,
            bool include_faces,
            bool include_edges,
            bool include_vertices,
            bool localized,
            bool one_sided) {
            const auto result = ipc::simplified_smoothed_offset_potential_cpp_tinyad(
                q,
                V,
                F,
                alpha,
                p,
                epsilon,
                include_faces,
                include_edges,
                include_vertices,
                localized,
                one_sided);
            return py::make_tuple(result.first, result.second);
        },
        py::arg("q"),
        py::arg("V"),
        py::arg("F"),
        py::arg("alpha") = 0.1,
        py::arg("p") = 2.0,
        py::arg("epsilon") = 0.1,
        py::arg("include_faces") = true,
        py::arg("include_edges") = true,
        py::arg("include_vertices") = true,
        py::arg("localized") = false,
        py::arg("one_sided") = false,
        "Compute simplified potential and gradient using TinyAD.");

    m.def(
        "simplified_potential_face",
        [](
            const Eigen::Vector3d& q,
            const Eigen::Matrix<double, 3, 3>& face_points,
            const ipc::PotentialParameters& params) {
            return ipc::simplified_potential_face(q, face_points, params);
        },
        py::arg("q"),
        py::arg("face_points"),
        py::arg("params"),
        "Compute simplified face potential using the C++ implementation.");

    m.def(
        "simplified_potential_face_cpp_tinyad",
        [](
            const Eigen::Vector3d& q,
            const Eigen::Matrix<double, 3, 3>& face_points,
            const ipc::PotentialParameters& params) {
            const auto result = ipc::simplified_potential_face_cpp_tinyad(
                q,
                face_points,
                params);
            return py::make_tuple(result.first, result.second);
        },
        py::arg("q"),
        py::arg("face_points"),
        py::arg("params"),
        "Compute simplified face potential and gradient using TinyAD.");

    m.def(
        "simplified_potential_edge",
        [](
            const Eigen::Vector3d& q,
            const Eigen::Matrix<double, 2, 3>& edge_points,
            const ipc::PotentialParameters& params) {
            return ipc::simplified_potential_edge(q, edge_points, params);
        },
        py::arg("q"),
        py::arg("edge_points"),
        py::arg("params"),
        "Compute simplified edge potential using the C++ implementation.");

    m.def(
        "simplified_potential_edge_cpp_tinyad",
        [](
            const Eigen::Vector3d& q,
            const Eigen::Matrix<double, 2, 3>& edge_points,
            const ipc::PotentialParameters& params) {
            const auto result = ipc::simplified_potential_edge_cpp_tinyad(
                q,
                edge_points,
                params);
            return py::make_tuple(result.first, result.second);
        },
        py::arg("q"),
        py::arg("edge_points"),
        py::arg("params"),
        "Compute simplified edge potential and gradient using TinyAD.");

    m.def(
        "simplified_potential_vertex",
        [](
            const Eigen::Vector3d& q,
            const Eigen::Vector3d& p_v,
            const ipc::PotentialParameters& params) {
            return ipc::simplified_potential_vertex(
                q, p_v, params);
        },
        py::arg("q"),
        py::arg("p_v"),
        py::arg("params"),
        "Compute simplified vertex potential using the C++ implementation.");

    m.def(
        "simplified_potential_vertex_cpp_tinyad",
        [](
            const Eigen::Vector3d& q,
            const Eigen::Vector3d& p_v,
            const ipc::PotentialParameters& params) {
            const auto result = ipc::simplified_potential_vertex_cpp_tinyad(
                q,
                p_v,
                params);
            return py::make_tuple(result.first, result.second);
        },
        py::arg("q"),
        py::arg("p_v"),
        py::arg("params"),
        "Compute simplified vertex potential and gradient using TinyAD.");

    m.def(
        "simplified_potential_face_grad_hess",
        [](
            const Eigen::Vector3d& q,
            const Eigen::Matrix<double, 3, 3>& face_points,
            const ipc::PotentialParameters& params) {
            Eigen::VectorXd grad;
            Eigen::MatrixXd hess;
            ipc::simplified_potential_face_grad_hess(
                q, face_points, params, grad, hess);
            return py::make_tuple(grad, hess);
        },
        py::arg("q"),
        py::arg("face_points"),
        py::arg("params"),
        "Compute simplified face potential gradient and Hessian using TinyAD.");

    m.def(
        "simplified_potential_edge_grad_hess",
        [](
            const Eigen::Vector3d& q,
            const Eigen::Matrix<double, 2, 3>& edge_points,
            const ipc::PotentialParameters& params) {
            Eigen::VectorXd grad;
            Eigen::MatrixXd hess;
            ipc::simplified_potential_edge_grad_hess(
                q, edge_points, params, grad, hess);
            return py::make_tuple(grad, hess);
        },
        py::arg("q"),
        py::arg("edge_points"),
        py::arg("params"),
        "Compute simplified edge potential gradient and Hessian using TinyAD.");

    m.def(
        "simplified_potential_vertex_grad_hess",
        [](
            const Eigen::Vector3d& q,
            const Eigen::Vector3d& p_v,
            const ipc::PotentialParameters& params) {
            Eigen::VectorXd grad;
            Eigen::MatrixXd hess;
            ipc::simplified_potential_vertex_grad_hess(
                q, p_v, params, grad, hess);
            return py::make_tuple(grad, hess);
        },
        py::arg("q"),
        py::arg("p_v"),
        py::arg("params"),
        "Compute simplified vertex potential gradient and Hessian using TinyAD.");
}

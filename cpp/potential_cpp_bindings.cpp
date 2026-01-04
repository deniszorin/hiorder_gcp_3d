#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "potential_cpp.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_potential_cpp, m)
{
    py::class_<ipc::PotentialParameters>(m, "PotentialParameters")
        .def(
            py::init<double, double, double, bool, bool>(),
            py::arg("alpha") = 0.1,
            py::arg("p") = 2.0,
            py::arg("epsilon") = 0.1,
            py::arg("localized") = false,
            py::arg("one_sided") = false)
        .def_readwrite("alpha", &ipc::PotentialParameters::alpha)
        .def_readwrite("p", &ipc::PotentialParameters::p)
        .def_readwrite("epsilon", &ipc::PotentialParameters::epsilon)
        .def_readwrite("localized", &ipc::PotentialParameters::localized)
        .def_readwrite("one_sided", &ipc::PotentialParameters::one_sided);

    m.def(
        "smoothed_offset_potential_cpp",
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
            return ipc::smoothed_offset_potential(
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
        "Compute smoothed offset potential using the C++ implementation.");

    m.def(
        "smoothed_offset_potential_accelerated_cpp",
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
            return ipc::smoothed_offset_potential_accelerated_cpp(
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
        "Compute smoothed offset potential using the C++ accelerated implementation.");

    m.def(
        "pointed_vertices_cpp",
        [](
            Eigen::ConstRef<Eigen::MatrixXd> V,
            Eigen::ConstRef<Eigen::MatrixXi> F) {
            ipc::PotentialCollisionMesh mesh(V, F);
            const auto& flags = mesh.pointed_vertices();
            py::array_t<bool> out(flags.size());
            auto buf = out.mutable_unchecked<1>();
            for (ssize_t i = 0; i < static_cast<ssize_t>(flags.size()); i++) {
                buf(i) = flags[static_cast<size_t>(i)] != 0;
            }
            return out;
        },
        py::arg("V"),
        py::arg("F"),
        "Compute pointed vertices using the C++ implementation.");

    m.def(
        "potential_face",
        [](
            const Eigen::Vector3d& q,
            const Eigen::Matrix<double, 3, 3>& face_points,
            const ipc::PotentialParameters& params) {
            return ipc::potential_face(q, face_points, params);
        },
        py::arg("q"),
        py::arg("face_points"),
        py::arg("params"),
        "Compute face potential using the C++ implementation.");

    m.def(
        "potential_face_cpp_tinyad",
        [](
            const Eigen::Vector3d& q,
            const Eigen::Matrix<double, 3, 3>& face_points,
            const ipc::PotentialParameters& params) {
            const auto result = ipc::potential_face_cpp_tinyad(
                q,
                face_points,
                params);
            return py::make_tuple(result.first, result.second);
        },
        py::arg("q"),
        py::arg("face_points"),
        py::arg("params"),
        "Compute face potential and gradient using TinyAD.");

    m.def(
        "potential_edge",
        [](
            const Eigen::Vector3d& q,
            const Eigen::Matrix<double, 4, 3>& edge_points,
            bool has_f1,
            const ipc::PotentialParameters& params) {
            return ipc::potential_edge(q, edge_points, has_f1, params);
        },
        py::arg("q"),
        py::arg("edge_points"),
        py::arg("has_f1"),
        py::arg("params"),
        "Compute edge potential using the C++ implementation.");

    m.def(
        "potential_edge_cpp_tinyad",
        [](
            const Eigen::Vector3d& q,
            const Eigen::Matrix<double, 4, 3>& edge_points,
            bool has_f1,
            const ipc::PotentialParameters& params) {
            const auto result = ipc::potential_edge_cpp_tinyad(
                q,
                edge_points,
                has_f1,
                params);
            return py::make_tuple(result.first, result.second);
        },
        py::arg("q"),
        py::arg("edge_points"),
        py::arg("has_f1"),
        py::arg("params"),
        "Compute edge potential and gradient using TinyAD.");

    m.def(
        "potential_vertex",
        [](
            const Eigen::Vector3d& q,
            const Eigen::Vector3d& p_v,
            Eigen::ConstRef<Eigen::MatrixXd> neighbor_points,
            bool is_boundary,
            bool pointed_vertex,
            const ipc::PotentialParameters& params) {
            return ipc::potential_vertex(
                q, p_v, neighbor_points, is_boundary, pointed_vertex, params);
        },
        py::arg("q"),
        py::arg("p_v"),
        py::arg("neighbor_points"),
        py::arg("is_boundary"),
        py::arg("pointed_vertex"),
        py::arg("params"),
        "Compute vertex potential using the C++ implementation.");

    m.def(
        "potential_vertex_cpp_tinyad",
        [](
            const Eigen::Vector3d& q,
            const Eigen::Vector3d& p_v,
            Eigen::ConstRef<Eigen::MatrixXd> neighbor_points,
            bool is_boundary,
            bool pointed_vertex,
            const ipc::PotentialParameters& params) {
            const auto result = ipc::potential_vertex_cpp_tinyad(
                q,
                p_v,
                neighbor_points,
                is_boundary,
                pointed_vertex,
                params);
            return py::make_tuple(result.first, result.second);
        },
        py::arg("q"),
        py::arg("p_v"),
        py::arg("neighbor_points"),
        py::arg("is_boundary"),
        py::arg("pointed_vertex"),
        py::arg("params"),
        "Compute vertex potential and gradient using TinyAD.");

    m.def(
        "potential_face_grad_hess",
        [](
            const Eigen::Vector3d& q,
            const Eigen::Matrix<double, 3, 3>& face_points,
            const ipc::PotentialParameters& params) {
            Eigen::VectorXd grad;
            Eigen::MatrixXd hess;
            ipc::potential_face_grad_hess(q, face_points, params, grad, hess);
            return py::make_tuple(grad, hess);
        },
        py::arg("q"),
        py::arg("face_points"),
        py::arg("params"),
        "Compute face potential gradient and Hessian using TinyAD.");

    m.def(
        "potential_edge_grad_hess",
        [](
            const Eigen::Vector3d& q,
            const Eigen::Matrix<double, 4, 3>& edge_points,
            bool has_f1,
            const ipc::PotentialParameters& params) {
            Eigen::VectorXd grad;
            Eigen::MatrixXd hess;
            ipc::potential_edge_grad_hess(q, edge_points, has_f1, params, grad, hess);
            return py::make_tuple(grad, hess);
        },
        py::arg("q"),
        py::arg("edge_points"),
        py::arg("has_f1"),
        py::arg("params"),
        "Compute edge potential gradient and Hessian using TinyAD.");

    m.def(
        "potential_vertex_grad_hess",
        [](
            const Eigen::Vector3d& q,
            const Eigen::Vector3d& p_v,
            Eigen::ConstRef<Eigen::MatrixXd> neighbor_points,
            bool is_boundary,
            bool pointed_vertex,
            const ipc::PotentialParameters& params) {
            Eigen::VectorXd grad;
            Eigen::MatrixXd hess;
            ipc::potential_vertex_grad_hess(
                q, p_v, neighbor_points, is_boundary, pointed_vertex, params, grad, hess);
            return py::make_tuple(grad, hess);
        },
        py::arg("q"),
        py::arg("p_v"),
        py::arg("neighbor_points"),
        py::arg("is_boundary"),
        py::arg("pointed_vertex"),
        py::arg("params"),
        "Compute vertex potential gradient and Hessian using TinyAD.");
}

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "potential_cpp.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_potential_cpp, m)
{
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
            return ipc::smoothed_offset_potential(
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
}

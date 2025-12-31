#include "cone_convex_hull.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

std::vector<Eigen::Vector3d> to_vec3_list(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& e)
{
    if (e.ndim() != 2 || e.shape(1) != 3) {
        throw std::runtime_error("e must be (n, 3).");
    }
    const ssize_t n = e.shape(0);
    std::vector<Eigen::Vector3d> out;
    out.reserve(static_cast<size_t>(n));
    auto buf = e.unchecked<2>();
    for (ssize_t i = 0; i < n; i++) {
        out.emplace_back(buf(i, 0), buf(i, 1), buf(i, 2));
    }
    return out;
}

} // namespace

PYBIND11_MODULE(_cone_convex_hull_cpp, m)
{
    m.def(
        "cone_convex_hull_cpp",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& e,
           double eps) {
            const std::vector<Eigen::Vector3d> edges = to_vec3_list(e);
            const ipc::ConeHullResult result = ipc::cone_convex_hull(edges, eps);
            py::array_t<int> indices(result.indices.size());
            auto idx_buf = indices.mutable_unchecked<1>();
            for (ssize_t i = 0; i < static_cast<ssize_t>(result.indices.size()); i++) {
                idx_buf(i) = result.indices[static_cast<size_t>(i)];
            }
            return py::make_tuple(indices, result.coplanar, result.fullspace);
        },
        py::arg("e"),
        py::arg("eps") = 1e-12,
        "Compute the convex hull of a polygonal cone.");
}

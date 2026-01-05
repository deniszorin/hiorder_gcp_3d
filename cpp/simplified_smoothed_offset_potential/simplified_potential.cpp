#include "simplified_potential_impl.hpp"
#include "simplified_potential.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

#include <vtkCellArray.h>
#include <vtkIdList.h>
#include <vtkIdTypeArray.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkStaticCellLocator.h>

namespace ipc {

double simplified_smoothed_offset_potential_point(
    const Eigen::Vector3d& q, const std::vector<int>& face_indices,
    const PotentialCollisionMesh& mesh,
    const PotentialParameters& params,
    const std::vector<int>& edge_valence,
    const std::vector<char>& vertex_internal,
    const bool include_faces, const bool include_edges, const bool include_vertices)
{
    std::vector<int> edge_list;
    std::vector<int> vertex_list;
    get_vertices_and_edges(
        face_indices, mesh,
        edge_list, vertex_list);
    return simplified_smoothed_offset_potential_point_impl<double>(
        q, face_indices,
        edge_list, vertex_list,
        mesh,
        params,
        edge_valence,
        vertex_internal,
        include_faces, include_edges, include_vertices);
}

Eigen::VectorXd simplified_smoothed_offset_potential(
    Eigen::ConstRef<Eigen::MatrixXd> q,
    const PotentialCollisionMesh& mesh,
    const PotentialParameters& params,
    bool include_faces, bool include_edges, bool include_vertices)
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

    const std::vector<int> edge_valence = build_edge_valence(mesh);
    const std::vector<char> vertex_internal = build_vertex_internal(mesh);

    Eigen::VectorXd out(q.rows());
    for (int i = 0; i < q.rows(); i++) {
        const Eigen::Vector3d qi = q.row(i).transpose();
        out[i] = simplified_smoothed_offset_potential_point_impl<double>(
            qi, face_indices,
            edge_list, vertex_list,
            mesh,
            params,
            edge_valence,
            vertex_internal,
            include_faces, include_edges, include_vertices);
    }

    return out;
}

Eigen::VectorXd simplified_smoothed_offset_potential_accelerated(
    Eigen::ConstRef<Eigen::MatrixXd> q,
    const PotentialCollisionMesh& mesh,
    const PotentialParameters& params,
    bool include_faces, bool include_edges, bool include_vertices)
{
    if (!params.localized) {
        throw std::runtime_error("Accelerated simplified potential requires localized=true.");
    }
    if (q.cols() != 3) {
        throw std::runtime_error("q must have shape (nq, 3).");
    }
    if (!(include_faces || include_edges || include_vertices)) {
        return Eigen::VectorXd::Zero(q.rows());
    }

    const auto& V = mesh.rest_positions();
    const auto& F = mesh.faces();
    const vtkIdType n_points = static_cast<vtkIdType>(V.rows());
    const vtkIdType n_faces = static_cast<vtkIdType>(F.rows());

    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    points->SetNumberOfPoints(n_points);
    for (vtkIdType i = 0; i < n_points; i++) {
        points->SetPoint(i, V(i, 0), V(i, 1), V(i, 2));
    }

    vtkSmartPointer<vtkIdTypeArray> cell_ids = vtkSmartPointer<vtkIdTypeArray>::New();
    cell_ids->SetNumberOfValues(n_faces * 4);
    for (vtkIdType i = 0; i < n_faces; i++) {
        const vtkIdType base = i * 4;
        cell_ids->SetValue(base, 3);
        cell_ids->SetValue(base + 1, static_cast<vtkIdType>(F(i, 0)));
        cell_ids->SetValue(base + 2, static_cast<vtkIdType>(F(i, 1)));
        cell_ids->SetValue(base + 3, static_cast<vtkIdType>(F(i, 2)));
    }

    vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
    cells->SetCells(n_faces, cell_ids);

    vtkSmartPointer<vtkPolyData> poly = vtkSmartPointer<vtkPolyData>::New();
    poly->SetPoints(points);
    poly->SetPolys(cells);

    vtkSmartPointer<vtkStaticCellLocator> locator = vtkSmartPointer<vtkStaticCellLocator>::New();
    locator->SetDataSet(poly);
    locator->BuildLocator();

    vtkSmartPointer<vtkIdList> id_list = vtkSmartPointer<vtkIdList>::New();
    id_list->Allocate(n_faces);

    double closest_point[3] = {0.0, 0.0, 0.0};
    vtkIdType cell_id = -1;
    int sub_id = 0;
    double dist2 = 0.0;

    const std::vector<int> edge_valence = build_edge_valence(mesh);
    const std::vector<char> vertex_internal = build_vertex_internal(mesh);

    Eigen::VectorXd out(q.rows());
    for (int i = 0; i < q.rows(); i++) {
        const Eigen::Vector3d qi = q.row(i).transpose();
        double qpt[3] = { qi[0], qi[1], qi[2] };
        if (!locator->FindClosestPointWithinRadius(
                qpt, params.epsilon, closest_point, cell_id, sub_id, dist2)) {
            out[i] = 0.0;
            continue;
        }
        double bounds[6] = {
            qpt[0] - params.epsilon, qpt[0] + params.epsilon,
            qpt[1] - params.epsilon, qpt[1] + params.epsilon,
            qpt[2] - params.epsilon, qpt[2] + params.epsilon,
        };
        id_list->Reset();
        locator->FindCellsWithinBounds(bounds, id_list);
        const vtkIdType count = id_list->GetNumberOfIds();
        if (count == 0) {
            out[i] = 0.0;
            continue;
        }
        std::vector<int> face_indices;
        face_indices.reserve(static_cast<size_t>(count));
        for (vtkIdType j = 0; j < count; j++) {
            face_indices.push_back(static_cast<int>(id_list->GetId(j)));
        }
        out[i] = simplified_smoothed_offset_potential_point(
            qi, face_indices,
            mesh,
            params,
            edge_valence,
            vertex_internal,
            include_faces, include_edges, include_vertices);
    }

    return out;
}

Eigen::VectorXd simplified_smoothed_offset_potential_cpp(
    Eigen::ConstRef<Eigen::MatrixXd> q,
    Eigen::ConstRef<Eigen::MatrixXd> V,
    Eigen::ConstRef<Eigen::MatrixXi> F,
    double alpha, double p, double epsilon,
    bool include_faces, bool include_edges, bool include_vertices,
    bool localized, bool one_sided)
{
    PotentialCollisionMesh mesh(V, F);
    const PotentialParameters params{
        alpha,
        p,
        epsilon,
        localized,
        one_sided,
    };
    return simplified_smoothed_offset_potential(
        q,
        mesh,
        params,
        include_faces, include_edges, include_vertices);
}

Eigen::VectorXd simplified_smoothed_offset_potential_accelerated_cpp(
    Eigen::ConstRef<Eigen::MatrixXd> q,
    Eigen::ConstRef<Eigen::MatrixXd> V,
    Eigen::ConstRef<Eigen::MatrixXi> F,
    double alpha, double p, double epsilon,
    bool include_faces, bool include_edges, bool include_vertices,
    bool localized, bool one_sided)
{
    PotentialCollisionMesh mesh(V, F);
    const PotentialParameters params{
        alpha,
        p,
        epsilon,
        localized,
        one_sided,
    };
    return simplified_smoothed_offset_potential_accelerated(
        q,
        mesh,
        params,
        include_faces, include_edges, include_vertices);
}

double simplified_potential_face(
    const Eigen::Vector3d& q,
    const Eigen::Matrix<double, 3, 3>& face_points,
    const PotentialParameters& params)
{
    const FacePoints face = face_points_from_rows(face_points);
    return simplified_potential_face<double>(q, face, params);
}

double simplified_potential_edge(
    const Eigen::Vector3d& q,
    const Eigen::Matrix<double, 2, 3>& edge_points,
    const PotentialParameters& params)
{
    const Vector3<double> p0 = edge_points.row(0).transpose();
    const Vector3<double> p1 = edge_points.row(1).transpose();
    return simplified_potential_edge<double>(q, p0, p1, params);
}

double simplified_potential_vertex(
    const Eigen::Vector3d& q, const Eigen::Vector3d& p_v,
    const PotentialParameters& params)
{
    return simplified_potential_vertex<double>(q, p_v, params);
}

} // namespace ipc

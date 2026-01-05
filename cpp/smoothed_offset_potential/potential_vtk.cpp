#include "potential.hpp"

#include <stdexcept>
#include <vector>

#include <vtkCellArray.h>
#include <vtkIdList.h>
#include <vtkIdTypeArray.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkStaticCellLocator.h>

namespace ipc {

Eigen::VectorXd smoothed_offset_potential_accelerated(
    Eigen::ConstRef<Eigen::MatrixXd> q,
    const PotentialCollisionMesh& mesh,
    const PotentialParameters& params,
    bool include_faces, bool include_edges, bool include_vertices)
{
    if (!params.localized) {
        throw std::runtime_error("Accelerated potential requires localized=true.");
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
        out[i] = smoothed_offset_potential_point(
            qi, face_indices,
            mesh,
            params,
            include_faces, include_edges, include_vertices);
    }

    return out;
}

Eigen::VectorXd smoothed_offset_potential_accelerated_cpp(
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
    return smoothed_offset_potential_accelerated(
        q,
        mesh,
        params,
        include_faces, include_edges, include_vertices);
}

} // namespace ipc

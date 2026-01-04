#include "potential_cpp.hpp"
#include "potential_impl.hpp"

#include <array>
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

namespace {

inline Eigen::Vector3d vertex_position(const PotentialCollisionMesh& mesh, const int idx)
{
    return mesh.rest_positions().row(idx).transpose();
}

// ****************************************************************************
// Helpers for local to global and back vertex/edge index conversion

int find_local_edge(const PotentialCollisionMesh& mesh, const int fidx, const int edge_idx)
{
    const Eigen::MatrixXi& face_edges = mesh.faces_to_edges();
    for (int i = 0; i < face_edges.cols(); i++) {
        if (face_edges(fidx, i) == edge_idx) {
            return i;
        }
    }
    throw std::runtime_error("Edge not found in face.");
}


// ****************************************************************************
// Helper to extract unique edge and vertex lists from a face list

void get_vertices_and_edges(
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

double smoothed_offset_potential_point_impl(
    const Eigen::Vector3d& q, const std::vector<int>& face_indices,
    const std::vector<int>& edge_list, const std::vector<int>& vertex_list,
    const PotentialCollisionMesh& mesh,
    const PotentialParameters& params,
    const bool include_faces, const bool include_edges, const bool include_vertices)
{
    // Compute potential from faces,edges and vertices given by the face list face_indices at point q.
    // See smoothed_offset_potential for arguments.
    if (!(include_faces || include_edges || include_vertices)) {
        return 0.0;
    }

    double face_sum = 0.0;
    double edge_sum = 0.0;
    double vertex_sum = 0.0;

    for (const int fidx : face_indices) {
        if (include_faces) {
            const int v0 = mesh.faces()(fidx, 0);
            const int v1 = mesh.faces()(fidx, 1);
            const int v2 = mesh.faces()(fidx, 2);
            const FacePoints face_points = {
                vertex_position(mesh, v0),
                vertex_position(mesh, v1),
                vertex_position(mesh, v2),
            };
            face_sum += potential_face(
                q, face_points,
                params);
        }
    }

    if (include_edges) {
        for (const int edge_idx : edge_list) {
            int f0 = mesh.edges_to_faces()(edge_idx, 0);
            int f1 = mesh.edges_to_faces()(edge_idx, 1);
            if (f0 < 0 && f1 >= 0) {
                std::swap(f0, f1);
            }
            const bool has_f1 = f1 >= 0;
            const int local0 = find_local_edge(mesh, f0, edge_idx);
            const int edge_v0 = mesh.faces()(f0, local0);
            const int edge_v1 = mesh.faces()(f0, (local0 + 1) % 3);
            const int other_v0 = mesh.faces()(f0, (local0 + 2) % 3);

            EdgePoints edge_points;
            edge_points[0] = vertex_position(mesh, edge_v0);
            edge_points[1] = vertex_position(mesh, edge_v1);
            edge_points[2] = vertex_position(mesh, other_v0);
            edge_points[3] = Eigen::Vector3d::Zero();
            if (has_f1) {
                const int local1 = find_local_edge(mesh, f1, edge_idx);
                const int other_v1 = mesh.faces()(f1, (local1 + 2) % 3);
                edge_points[3] = vertex_position(mesh, other_v1);
            }
            edge_sum += potential_edge(
                q, edge_points,
                has_f1,
                params);
        }
    }

    if (include_vertices) {
        for (const int v_idx : vertex_list) {
            const Eigen::Vector3d p_v = vertex_position(mesh, v_idx);
            const auto& edge_list_v = mesh.vertices_to_edges()[static_cast<size_t>(v_idx)];
            if (edge_list_v.size() > 50) {
                throw std::runtime_error("Vertex has more than 50 incident edges.");
            }
            std::array<Eigen::Vector3d, 50> neighbor_points;
            int neighbor_count = 0;
            int boundary_count = 0;
            for (const int edge_idx : edge_list_v) {
                const int a = mesh.edges()(edge_idx, 0);
                const int b = mesh.edges()(edge_idx, 1);
                const int neighbor_idx = (a == v_idx) ? b : a;
                neighbor_points[neighbor_count] = vertex_position(mesh, neighbor_idx);
                neighbor_count++;
                if (mesh.edges_to_faces()(edge_idx, 0) < 0
                    || mesh.edges_to_faces()(edge_idx, 1) < 0) {
                    boundary_count++;
                }
            }
            const bool is_boundary = boundary_count == 2;
            const bool pointed_vertex = mesh.pointed_vertices()[static_cast<size_t>(v_idx)] != 0;
            vertex_sum += potential_vertex(
                q, p_v,
                neighbor_points.data(), neighbor_count, is_boundary,
                pointed_vertex,
                params);
        }
    }

    return face_sum + edge_sum + vertex_sum;
}

// ****************************************************************************
// Main potential calls

double smoothed_offset_potential_point(
    const Eigen::Vector3d& q, const std::vector<int>& face_indices,
    const PotentialCollisionMesh& mesh,
    const PotentialParameters& params,
    const bool include_faces, const bool include_edges, const bool include_vertices)
{
    std::vector<int> edge_list;
    std::vector<int> vertex_list;
    get_vertices_and_edges(
        face_indices, mesh,
        edge_list, vertex_list);
    return smoothed_offset_potential_point_impl(
        q, face_indices,
        edge_list, vertex_list,
        mesh,
        params,
        include_faces, include_edges, include_vertices);
}

} // namespace

Eigen::VectorXd smoothed_offset_potential(
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

    Eigen::VectorXd out(q.rows());
    for (int i = 0; i < q.rows(); i++) {
        const Eigen::Vector3d qi = q.row(i).transpose();
        out[i] = smoothed_offset_potential_point_impl(
            qi, face_indices,
            edge_list, vertex_list,
            mesh,
            params,
            include_faces, include_edges, include_vertices);
    }

    return out;
}

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

Eigen::VectorXd smoothed_offset_potential_cpp(
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
    return smoothed_offset_potential(
        q,
        mesh,
        params,
        include_faces, include_edges, include_vertices);
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

double potential_face(
    const Eigen::Vector3d& q,
    const Eigen::Matrix<double, 3, 3>& face_points,
    const PotentialParameters& params)
{
    const FacePoints face = face_points_from_rows(face_points);
    return potential_face(q, face, params);
}

double potential_edge(
    const Eigen::Vector3d& q,
    const Eigen::Matrix<double, 4, 3>& edge_points,
    const bool has_f1,
    const PotentialParameters& params)
{
    const EdgePoints edge_points_arr = edge_points_from_rows(edge_points);
    return potential_edge(q, edge_points_arr, has_f1, params);
}

double potential_vertex(
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
    std::vector<Eigen::Vector3d> neighbor_list;
    neighbor_list.reserve(neighbor_points.rows());
    for (int i = 0; i < neighbor_points.rows(); i++) {
        neighbor_list.push_back(neighbor_points.row(i).transpose());
    }
    return potential_vertex(
        q, p_v,
        neighbor_list.data(), static_cast<int>(neighbor_list.size()), is_boundary,
        pointed_vertex,
        params);
}

} // namespace ipc

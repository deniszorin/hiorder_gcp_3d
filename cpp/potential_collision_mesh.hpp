#pragma once

#include "collision_mesh.hpp"

#include <array>
#include <vector>

namespace ipc {

class PotentialCollisionMesh : public CollisionMesh {
public:
    PotentialCollisionMesh(
        Eigen::ConstRef<Eigen::MatrixXd> rest_positions,
        Eigen::ConstRef<Eigen::MatrixXi> faces = Eigen::MatrixXi());

    const Eigen::MatrixXd& normals() const { return m_normals; }
    const Eigen::MatrixXd& vertex_normals() const { return m_vertex_normals; }
    const Eigen::MatrixXd& edge_normals() const { return m_edge_normals; }
    const std::vector<char>& pointed_vertices() const { return m_pointed_vertices; }
    void set_pointed_vertices(const std::vector<char>& pointed_vertices);
    const std::array<Eigen::Vector3d, 3>& edge_inward(const int f) const
    {
        return m_edge_inward[f];
    }

private:
    void compute_face_geometry();
    void compute_edge_normals();
    void compute_vertex_normals();
    void compute_pointed_vertices();

    Eigen::MatrixXd m_normals;
    Eigen::MatrixXd m_vertex_normals;
    std::vector<std::array<Eigen::Vector3d, 3>> m_edge_inward;
    Eigen::MatrixXd m_edge_normals;
    std::vector<char> m_pointed_vertices;
};

} // namespace ipc

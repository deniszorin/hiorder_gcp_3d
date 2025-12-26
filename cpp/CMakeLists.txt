cmake_minimum_required(VERSION 3.16)

project(hiorder_collision_mesh LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED NO_MODULE)

add_library(collision_mesh STATIC
    collision_mesh.cpp
)

target_include_directories(collision_mesh PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

add_executable(collision_mesh_test
    tests/collision_mesh_test.cpp
)

target_link_libraries(collision_mesh_test PRIVATE collision_mesh)

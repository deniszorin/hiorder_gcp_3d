cmake_minimum_required(VERSION 3.16)

project(hiorder_collision_mesh LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(VTK QUIET COMPONENTS CommonCore CommonDataModel)

set(COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)
set(SMOOTHED_DIR ${CMAKE_CURRENT_SOURCE_DIR}/smoothed_offset_potential)
set(SIMPLIFIED_DIR ${CMAKE_CURRENT_SOURCE_DIR}/simplified_smoothed_offset_potential)

set(VTK_SOURCES "")
if(VTK_FOUND)
    list(APPEND VTK_SOURCES
        ${SMOOTHED_DIR}/potential_vtk.cpp
        ${SIMPLIFIED_DIR}/simplified_potential_vtk.cpp
    )
endif()

execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    RESULT_VARIABLE PYBIND11_CMAKE_RESULT
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(PYBIND11_CMAKE_RESULT EQUAL 0 AND PYBIND11_CMAKE_DIR)
    list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})
endif()
find_package(pybind11 CONFIG REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
    RESULT_VARIABLE NUMPY_RESULT
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT NUMPY_RESULT EQUAL 0 OR NOT NUMPY_INCLUDE_DIR)
    message(FATAL_ERROR "NumPy include directory not found.")
endif()

add_library(predicates STATIC
    predicates.c
)

add_library(collision_mesh STATIC
    ${COMMON_DIR}/collision_mesh.cpp
)

target_include_directories(collision_mesh PUBLIC
    ${COMMON_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(collision_mesh PUBLIC Eigen3::Eigen)

add_library(cone_convex_hull STATIC
    ${COMMON_DIR}/cone_convex_hull.cpp
)

target_include_directories(cone_convex_hull PUBLIC
    ${COMMON_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(cone_convex_hull PRIVATE predicates)
target_link_libraries(cone_convex_hull PUBLIC Eigen3::Eigen)

add_library(potential_core STATIC
    ${COMMON_DIR}/potential_collision_mesh.cpp
    ${SMOOTHED_DIR}/potential.cpp
    ${SMOOTHED_DIR}/potential_tinyad.cpp
    ${SIMPLIFIED_DIR}/simplified_potential.cpp
    ${SIMPLIFIED_DIR}/simplified_potential_tinyad.cpp
    ${VTK_SOURCES}
)

target_include_directories(potential_core PUBLIC
    ${COMMON_DIR}
    ${SMOOTHED_DIR}
    ${SIMPLIFIED_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/TinyAD/include
)

target_link_libraries(potential_core PUBLIC
    collision_mesh
    cone_convex_hull
    Eigen3::Eigen
)

if(VTK_FOUND)
    target_link_libraries(potential_core PUBLIC
        VTK::CommonCore
        VTK::CommonDataModel
    )
    target_compile_definitions(potential_core PUBLIC IPC_HAS_VTK)
endif()

add_library(_potential_cpp MODULE
    ${SMOOTHED_DIR}/potential_bindings.cpp
)

target_include_directories(_potential_cpp PRIVATE
    ${NUMPY_INCLUDE_DIR}
)

target_link_libraries(_potential_cpp PRIVATE
    potential_core
    pybind11::module
    Python3::Module
)

set_target_properties(_potential_cpp PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_potential_cpp"
)

if(Python3_EXTENSION_SUFFIX)
    set_target_properties(_potential_cpp PROPERTIES SUFFIX "${Python3_EXTENSION_SUFFIX}")
elseif(PYTHON_MODULE_EXTENSION)
    set_target_properties(_potential_cpp PROPERTIES SUFFIX "${PYTHON_MODULE_EXTENSION}")
endif()

if(APPLE)
    target_link_options(_potential_cpp PRIVATE "-undefined" "dynamic_lookup")
endif()

add_library(_simplified_potential_cpp MODULE
    ${SIMPLIFIED_DIR}/simplified_potential_bindings.cpp
)

target_include_directories(_simplified_potential_cpp PRIVATE
    ${NUMPY_INCLUDE_DIR}
)

target_link_libraries(_simplified_potential_cpp PRIVATE
    potential_core
    pybind11::module
    Python3::Module
)

set_target_properties(_simplified_potential_cpp PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_simplified_potential_cpp"
)

if(Python3_EXTENSION_SUFFIX)
    set_target_properties(_simplified_potential_cpp PROPERTIES SUFFIX "${Python3_EXTENSION_SUFFIX}")
elseif(PYTHON_MODULE_EXTENSION)
    set_target_properties(_simplified_potential_cpp PROPERTIES SUFFIX "${PYTHON_MODULE_EXTENSION}")
endif()

if(APPLE)
    target_link_options(_simplified_potential_cpp PRIVATE "-undefined" "dynamic_lookup")
endif()

add_executable(collision_mesh_test
    tests/collision_mesh_test.cpp
)

target_link_libraries(collision_mesh_test PRIVATE collision_mesh)

common
eigen_ext.hpp needed for collision_mesh
collision_mesh.{cpp, hpp}  A bare-bones version of the collision mesh from ipc-toolkit
cone_convex_hull_bindings.cpp  Python bindings for convex hull computation for a polygonal cone, used by outside_vertex function, through precomputed pointed_vertex flags (pointed_vertex = convex hull of the incident edges is not a full space). 
cone_convex_hull.{cpp,hpp}  computing a convex hull of a cone 
geometry_connectivity.hpp  various generic geometric computations needed for the potential evaluation and 
additional connectivity datastructures 
potential_collision_mesh.{cpp,hpp}   derived class of collision_mesh that adds more connectivity datastructures
potential_parameters.hpp a  struct to hold  potential paraders alpha, p, epsilon, and flags.


smoothed_offset_potential 
potential_impl.hpp: most smoothed offset potential implementation is here as templated functions for autodiff
potential.{hpp,cpp}   top level smoothed offset potential functions and wrappers for python binding 
potential_tinyad.cpp    instantiations/wrappers for autodiff computation of gradient and hessian for the potential components potential_face, edge, vertex. 
potential_bindings.cpp  Python bindings for smoothed offset potential functions
potential_vtk.cpp  VTK-accelerated localized evaluation helpers

simplified_smoothed_offset_potential
similar structure to smoothed contact potential

simplified_potential_bindings.cpp
simplified_potential_impl.hpp   most simplified smoothed offset potential implementation is here as templated functions for autodiff
simplified_potential.{cpp,hpp}   top level smoothed offset potential functions and wrappers for python binding 
simplified_potential_tinyad.cpp    instantiations/wrappers for autodiff computation of gradient and hessian for the potential components
simplified_potential_vtk.cpp  VTK-accelerated localized evaluation helpers

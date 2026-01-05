
This repository contains 3D potential evaluation, for meshes, with
Numba/Python implementations and C++/TinyAD equivalents. It includes:
* smoothed offset potential"
* simplified smoothed offset potential



## Layout

- `cpp/`: C++ implementation and bindings
  - `cpp/common/`: shared geometry/connectivity and mesh data structures
  - `cpp/smoothed_offset_potential/`: smoothed offset potential + TinyAD helpers
  - `cpp/simplified_smoothed_offset_potential/`: simplified potential + TinyAD helpers
  - `cpp/TinyAD/`: TinyAD headers
- `potential_numba.py`, `simplified_potential_numba.py`: Numba implementations
- `potential_cpp.py`: Python wrapper for C++ bindings
- `geometry.py`, `geometry_connectivity_numba.py`: geometry/connectivity helpers
- `tests/`: pytest coverage for Python/Numba/C++ agreement etc
- `scripts/`: visualization scripts

## Scripts

- `scripts/visualize_cone_convex_hull.py`: generate HTML cone/hull visualizations with PyVista
  for synthetic edge sets (writes to `visualizations/cone_convex_hull` by default).

- `scripts/generate_visualizations.py`: produce validation figures for smoothed offset potential,
  and export an OBJ for the `cube_face_centers` scene.
  - `--output-dir`: directory to write outputs (default: `visualizations`).
  - `--use-cpp`, switch potential computation: 
  - `--use-cpp` (default: False)  if True, use cpp implementation otherwise python
  - `--use-accelerated` (default: False) if True, use VTK acceleration for looking up triangles within eps (requires localized) works both for python and cpp
  - `--use-simplified` (default: False)  if True, use the simplified version of the potential, alpha ignored
  - `--localized`: enable localization (h_eps)   if True, use localization within epsilon, otherwise potential does not vanish
  - `--epsilon`: localization epsilon for isosurfaces (default: `0.1`).
  - `--alpha`: smoothing alpha (default: `0.1`). Only works for smoothed potential (default)
  - `--p`: potential exponent (default: `2.0`).

python scripts/generate_visualizations.py  --use-cpp --use-accelerated --use-simplified --localized

- `scripts/verify_outside_checks.py`: render outside-face/edge/vertex checks with PyVista, saving
  images/HTML to help validate outside tests.


## Dependencies
Python: numba, numpy, pyvista, pytest, pybind, setuptools, igl
C++:  vtk (used for accelerated lookup of triangles close to an eval point, optional)

## Build (C++)

cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j
```

VTK is optional. If found, accelerated VTK paths are compiled and exposed in the
Python bindings. If VTK is missing, accelerated functions are not exported.

## Tests

Test scripts are in tests/ run all with pytest, there are many.

## Notes

- The simplified potential matches the smoothed potential at `alpha = 0` 

## Environment
Use the Python virtual environment at `../hiorder_gcp_2d/.venv` for running all commands.

## Code style
- Follow these rules for function arguments, both defs and calls: 
  - For 5 arguments or less fitting on one line (79 characters), place these 
    on one line; 
  - do not use more than 4 positional arguments
  - for larger groups of arguments (more than 5), split into semantic groups, one per line
- Preserve existing hand-crafted grouping for function definitions and calls unless
  the argument list itself changes. Do not reflow existing argument groupings just
  to satisfy wrapping rules.
- Example of semantic grouping 
  def smoothed_offset_potential_numba(
    q: ArrayF,
    mesh, geom,
    alpha: float = 0.1, p: float = 2.0, epsilon: float = 0.1,
    include_faces: bool = True, include_edges: bool = True, include_vertices: bool = True, localized: bool = False, one_sided: bool = False,
) -> ArrayF:
by line semantic groups: 
line 1:  main argument q
line 2: mesh structures
line 3: potential parameters
line 4: flags to adjust potential 

## Comment management
Do not remove user comments unless the code these comments refer to is completely removed.
If comments do not reflect agent's code modifications, mark these comments as `AGENT: Comment possibly outdated`.

## Other
Do not update `tests/data/potential_regression.npz` unless explicitly requested.

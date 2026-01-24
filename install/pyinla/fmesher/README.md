# fmesher

Python wrapper for R's [fmesher](https://cran.r-project.org/package=fmesher) package for creating constrained Delaunay triangulations with mesh refinement.

## Installation

```bash
pip install fmesher
```

### R Dependencies

This package requires R and the fmesher R package. Install them:

1. **Install R**: https://cran.r-project.org/

2. **Install R packages** (run in R console):
```r
install.packages("fmesher")
install.packages("sf")  # Required for fm_nonconvex_hull
```

Or use the helper function after pip install:
```python
from fmesher import install_r_packages
install_r_packages()
```

## Quick Start

```python
import numpy as np
from fmesher import fm_mesh_2d, check_fmesher_available

# Check if everything is installed
available, error = check_fmesher_available()
if not available:
    print(f"Setup needed: {error}")

# Create mesh from points
locs = np.random.randn(100, 2)
mesh = fm_mesh_2d(
    loc=locs,
    max_edge=[0.5, 1.0],
    cutoff=0.1
)

print(f"Mesh: {mesh.n} vertices, {mesh.n_triangle} triangles")
```

## Features

- **fm_mesh_2d**: Create 2D triangular meshes with refinement
- **fm_nonconvex_hull**: Create non-convex hull boundaries
- **fm_segm**: Create mesh segments (boundaries/constraints)

## Example with Boundary

```python
from fmesher import fm_mesh_2d, fm_nonconvex_hull

# Create non-convex hull boundary
boundary = fm_nonconvex_hull(locs, convex=-0.15)

# Create mesh with boundary
mesh = fm_mesh_2d(
    loc=locs,
    boundary=boundary,
    max_edge=0.4
)
```

## API Reference

### fm_mesh_2d

```python
fm_mesh_2d(
    loc=None,           # Point locations (n, 2) array
    loc_domain=None,    # Domain extent points
    offset=None,        # Boundary extension distance
    n=None,             # Points on extended boundary
    boundary=None,      # Boundary segment(s)
    interior=None,      # Interior constraint(s)
    max_edge=None,      # Maximum triangle edge length
    min_angle=None,     # Minimum triangle angle
    cutoff=None,        # Minimum point separation
) -> FmMesh
```

### FmMesh attributes

- `loc`: Vertex coordinates (n, 2) array
- `tv`: Triangle vertex indices (n_triangles, 3) array
- `n`: Number of vertices
- `n_triangle`: Number of triangles
- `bbox`: Bounding box dict
- `summary()`: Print mesh summary

## License

MIT

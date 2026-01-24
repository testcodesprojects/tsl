#!/usr/bin/env python
"""Example usage of the fmesher Python wrapper.

Run this script to test the fmesher installation:
    python -m fmesher.example

Or from the spde directory:
    python fmesher/example.py
"""

import numpy as np


def main():
    print("=" * 60)
    print("fmesher Python Wrapper - Example")
    print("=" * 60)

    # Check availability first
    from fmesher import check_fmesher_available

    available, error = check_fmesher_available()
    if not available:
        print(f"\nfmesher is not available: {error}")
        print("\nTo install:")
        print("  1. pip install rpy2")
        print("  2. In R: install.packages('fmesher')")
        return

    print("\nfmesher is available!")

    from fmesher import fm_mesh_2d, fm_nonconvex_hull, fm_segm

    # Example 1: Simple mesh from random points
    print("\n" + "-" * 40)
    print("Example 1: Mesh from random points")
    print("-" * 40)

    np.random.seed(42)
    locs = np.random.randn(100, 2)

    mesh = fm_mesh_2d(
        loc=locs,
        max_edge=[0.5, 1.0],  # inner and outer max edge lengths
        cutoff=0.1,  # minimum point separation
    )

    print(mesh.summary())

    # Example 2: Mesh with automatic boundary extension
    print("\n" + "-" * 40)
    print("Example 2: Mesh with boundary extension")
    print("-" * 40)

    mesh_extended = fm_mesh_2d(
        loc=locs,
        offset=[-0.1, -0.3],  # negative = fraction of diameter
        max_edge=[0.3, 0.8],
        cutoff=0.1,
    )

    print(mesh_extended.summary())

    # Example 3: Mesh with non-convex hull boundary
    print("\n" + "-" * 40)
    print("Example 3: Mesh with non-convex hull")
    print("-" * 40)

    boundary = fm_nonconvex_hull(locs, convex=-0.15)
    print(f"Non-convex hull: {boundary.n_vertices} vertices, {boundary.n_edges} edges")

    mesh_hull = fm_mesh_2d(
        loc=locs,
        boundary=boundary,
        max_edge=0.4,
        cutoff=0.1,
    )

    print(mesh_hull.summary())

    # Example 4: Mesh with custom boundary
    print("\n" + "-" * 40)
    print("Example 4: Mesh with custom boundary")
    print("-" * 40)

    # Create a square boundary
    corners = np.array([[-2, -2], [2, -2], [2, 2], [-2, 2]])
    boundary = fm_segm(corners, is_bnd=True)

    mesh_custom = fm_mesh_2d(
        loc=locs,
        boundary=boundary,
        max_edge=0.3,
        cutoff=0.1,
    )

    print(mesh_custom.summary())

    # Show mesh data access
    print("\n" + "-" * 40)
    print("Accessing mesh data")
    print("-" * 40)
    print(f"Vertex locations shape: {mesh.loc.shape}")
    print(f"Triangle indices shape: {mesh.tv.shape}")
    print(f"Edge array shape: {mesh.vv.shape}")
    print(f"First 5 triangles:\n{mesh.tv[:5]}")
    print(f"Bounding box: {mesh.bbox}")

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

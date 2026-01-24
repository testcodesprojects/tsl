"""Python wrapper for R's fmesher package.

This package provides a Python interface to R's fmesher package for creating
constrained Delaunay triangulations with mesh refinement capabilities.

Installation:
    pip install pyinla[fmesher]

Requirements:
    - R: Must have R installed on the system
    - fmesher R package: install.packages("fmesher")
    - sf R package: install.packages("sf")  # for fm_nonconvex_hull

Quick setup:
    >>> from pyinla.fmesher import install_r_packages
    >>> install_r_packages()

Example:
    >>> from pyinla.fmesher import fm_mesh_2d
    >>> import numpy as np
    >>>
    >>> locs = np.random.randn(100, 2)
    >>> mesh = fm_mesh_2d(loc=locs, max_edge=[0.5, 1.0], cutoff=0.1)
    >>> print(f"Mesh: {mesh.n} vertices, {mesh.n_triangle} triangles")
"""

__version__ = "0.1.0"

# Lazy imports - only load when accessed to avoid rpy2 import errors
# when the fmesher extra is not installed

def __getattr__(name):
    """Lazy import fmesher components only when accessed."""

    _CORE_FUNCTIONS = {
        "fm_mesh_2d", "fm_nonconvex_hull", "fm_segm", "fm_hexagon_lattice",
        "fm_fem", "fm_evaluator", "fm_evaluate", "get_utm_crs",
        "check_fmesher_available",
    }

    _MESH_CLASSES = {"FmMesh", "FmSegment", "FmFEM", "FmEvaluator"}

    _SPDE_ITEMS = {
        "spde2_pcmatern", "spde_make_A", "spde_grid_projector",
        "SPDE2PcMatern", "SpdeGridProjector",
    }

    _EXCEPTIONS = {"FmesherError", "FmesherNotAvailableError", "Rpy2NotAvailableError"}

    _INSTALL_ITEMS = {"install_r_packages", "check_r_installation"}

    try:
        if name in _CORE_FUNCTIONS:
            from .core import (
                fm_mesh_2d, fm_nonconvex_hull, fm_segm, fm_hexagon_lattice,
                fm_fem, fm_evaluator, fm_evaluate, get_utm_crs,
                check_fmesher_available,
            )
            return locals()[name]

        elif name in _MESH_CLASSES:
            from .mesh import FmMesh, FmSegment, FmFEM, FmEvaluator
            return locals()[name]

        elif name in _SPDE_ITEMS:
            from .spde import (
                spde2_pcmatern, spde_make_A, spde_grid_projector,
                SPDE2PcMatern, SpdeGridProjector,
            )
            return locals()[name]

        elif name in _EXCEPTIONS:
            from .exceptions import (
                FmesherError, FmesherNotAvailableError, Rpy2NotAvailableError,
            )
            return locals()[name]

        elif name in _INSTALL_ITEMS:
            from .install import install_r_packages, check_r_installation
            return locals()[name]

    except ImportError as e:
        raise ImportError(
            f"fmesher requires rpy2, R, and R packages to be installed.\n\n"
            f"Setup steps:\n"
            f"  1. Install R on your system (https://cran.r-project.org/)\n"
            f"  2. pip install pyinla[fmesher]\n"
            f"  3. R -e 'install.packages(c(\"fmesher\", \"sf\"))'\n\n"
            f"Original error: {e}"
        ) from e

    raise AttributeError(f"module 'pyinla.fmesher' has no attribute '{name}'")


__all__ = [
    # Main functions
    "fm_mesh_2d",
    "fm_nonconvex_hull",
    "fm_segm",
    "fm_hexagon_lattice",
    "fm_fem",
    "fm_evaluator",
    "fm_evaluate",
    "get_utm_crs",
    "check_fmesher_available",
    # SPDE functions
    "spde2_pcmatern",
    "spde_make_A",
    "spde_grid_projector",
    # Installation helpers
    "install_r_packages",
    "check_r_installation",
    # Data classes
    "FmMesh",
    "FmSegment",
    "FmFEM",
    "FmEvaluator",
    "SPDE2PcMatern",
    "SpdeGridProjector",
    # Exceptions
    "FmesherError",
    "FmesherNotAvailableError",
    "Rpy2NotAvailableError",
]

"""Mesh data classes for fmesher results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import sparse


@dataclass
class FmSegment:
    """A mesh segment (boundary or interior constraint).

    Attributes
    ----------
    loc : np.ndarray
        Vertex locations, shape (n_vertices, 2) or (n_vertices, 3).
    idx : np.ndarray
        Segment edge indices (0-based), shape (n_edges, 2).
    is_bnd : bool
        True if this is a boundary segment, False for interior constraint.
    grp : np.ndarray, optional
        Group labels for each edge.
    crs : str, optional
        Coordinate reference system.
    """

    loc: np.ndarray
    idx: np.ndarray
    is_bnd: bool = True
    grp: Optional[np.ndarray] = None
    crs: Optional[str] = None

    def __post_init__(self) -> None:
        self.loc = np.asarray(self.loc, dtype=float)
        self.idx = np.asarray(self.idx, dtype=np.int64)

    @property
    def n_vertices(self) -> int:
        """Number of vertices in the segment."""
        return self.loc.shape[0]

    @property
    def n_edges(self) -> int:
        """Number of edges in the segment."""
        return self.idx.shape[0]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "loc": self.loc,
            "idx": self.idx,
            "is_bnd": self.is_bnd,
        }
        if self.grp is not None:
            result["grp"] = self.grp
        if self.crs is not None:
            result["crs"] = self.crs
        return result


@dataclass
class FmMesh:
    """A 2D triangular mesh from fmesher.

    Attributes
    ----------
    loc : np.ndarray
        Vertex locations, shape (n, 2) or (n, 3).
    tv : np.ndarray
        Triangle vertex indices (0-based), shape (n_triangles, 3).
    vv : np.ndarray
        Vertex-to-vertex adjacency edges, shape (n_edges, 2).
    tt : np.ndarray
        Triangle-to-triangle adjacency, shape (n_triangles, 3).
        Each row contains indices of adjacent triangles (-1 if no neighbor).
    segm_bnd : list of FmSegment
        Boundary segments.
    segm_int : list of FmSegment
        Interior constraint segments.
    manifold : str
        Manifold type (e.g., "R2" for plane, "S2" for sphere).
    crs : str, optional
        Coordinate reference system.
    meta : dict
        Additional metadata from the mesh creation.
    """

    loc: np.ndarray
    tv: np.ndarray
    vv: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.int64))
    tt: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.int64))
    segm_bnd: List[FmSegment] = field(default_factory=list)
    segm_int: List[FmSegment] = field(default_factory=list)
    manifold: str = "R2"
    crs: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    _r_mesh: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.loc = np.asarray(self.loc, dtype=float)
        self.tv = np.asarray(self.tv, dtype=np.int64)
        self.vv = np.asarray(self.vv, dtype=np.int64)
        self.tt = np.asarray(self.tt, dtype=np.int64)

    @property
    def n(self) -> int:
        """Number of vertices in the mesh."""
        return self.loc.shape[0]

    @property
    def n_triangle(self) -> int:
        """Number of triangles in the mesh."""
        return self.tv.shape[0]

    @property
    def n_edge(self) -> int:
        """Number of edges in the mesh."""
        return self.vv.shape[0]

    @property
    def dim(self) -> int:
        """Coordinate dimension (2 or 3)."""
        return self.loc.shape[1] if self.loc.ndim == 2 else 1

    @property
    def bbox(self) -> Dict[str, Tuple[float, float]]:
        """Bounding box of the mesh."""
        result = {
            "x": (float(self.loc[:, 0].min()), float(self.loc[:, 0].max())),
            "y": (float(self.loc[:, 1].min()), float(self.loc[:, 1].max())),
        }
        if self.dim >= 3:
            result["z"] = (float(self.loc[:, 2].min()), float(self.loc[:, 2].max()))
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert mesh to dictionary representation."""
        return {
            "loc": self.loc,
            "tv": self.tv,
            "vv": self.vv,
            "tt": self.tt,
            "n": self.n,
            "n_triangle": self.n_triangle,
            "n_edge": self.n_edge,
            "manifold": self.manifold,
            "crs": self.crs,
            "segm_bnd": [s.to_dict() for s in self.segm_bnd],
            "segm_int": [s.to_dict() for s in self.segm_int],
            "meta": self.meta,
        }

    def summary(self) -> str:
        """Return a summary string of the mesh."""
        lines = [
            f"FmMesh:",
            f"  Manifold: {self.manifold}",
            f"  Vertices: {self.n}",
            f"  Triangles: {self.n_triangle}",
            f"  Edges: {self.n_edge}",
        ]
        bbox = self.bbox
        lines.append(f"  x range: [{bbox['x'][0]:.4f}, {bbox['x'][1]:.4f}]")
        lines.append(f"  y range: [{bbox['y'][0]:.4f}, {bbox['y'][1]:.4f}]")
        if "z" in bbox:
            lines.append(f"  z range: [{bbox['z'][0]:.4f}, {bbox['z'][1]:.4f}]")
        if self.crs:
            lines.append(f"  CRS: {self.crs}")
        if self.segm_bnd:
            lines.append(f"  Boundary segments: {len(self.segm_bnd)}")
        if self.segm_int:
            lines.append(f"  Interior segments: {len(self.segm_int)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"FmMesh(n={self.n}, n_triangle={self.n_triangle}, manifold='{self.manifold}')"


@dataclass
class FmEvaluator:
    """Evaluator/projector for mapping mesh values to a regular grid.

    This class holds the projection matrix and grid information needed
    to interpolate values from mesh vertices onto a regular grid for
    visualization purposes.

    Attributes
    ----------
    x : np.ndarray
        X coordinates of the grid (1D array).
    y : np.ndarray
        Y coordinates of the grid (1D array).
    proj_A : sparse.csr_matrix
        Projection matrix of shape (n_grid_points, n_mesh_vertices).
        Multiplying this by mesh vertex values gives grid values.
    xlim : tuple of (float, float)
        X-axis limits (min, max).
    ylim : tuple of (float, float)
        Y-axis limits (min, max).
    dims : tuple of (int, int)
        Grid dimensions (nx, ny).
    lattice_loc : np.ndarray
        Grid point locations, shape (n_grid_points, 2).
    """

    x: np.ndarray
    y: np.ndarray
    proj_A: sparse.csr_matrix
    xlim: Tuple[float, float]
    ylim: Tuple[float, float]
    dims: Tuple[int, int]
    lattice_loc: np.ndarray

    @property
    def n_grid(self) -> int:
        """Total number of grid points."""
        return len(self.x) * len(self.y)

    @property
    def shape(self) -> Tuple[int, int]:
        """Grid shape (nx, ny)."""
        return (len(self.x), len(self.y))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "x": self.x,
            "y": self.y,
            "proj_A": self.proj_A,
            "xlim": self.xlim,
            "ylim": self.ylim,
            "dims": self.dims,
            "lattice_loc": self.lattice_loc,
        }

    def summary(self) -> str:
        """Return a summary string of the evaluator."""
        lines = [
            "FmEvaluator:",
            f"  Grid dimensions: {self.shape}",
            f"  X range: [{self.xlim[0]:.4f}, {self.xlim[1]:.4f}]",
            f"  Y range: [{self.ylim[0]:.4f}, {self.ylim[1]:.4f}]",
            f"  Projection matrix: {self.proj_A.shape}, nnz={self.proj_A.nnz}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"FmEvaluator(dims={self.shape}, xlim={self.xlim}, ylim={self.ylim})"


@dataclass
class FmFEM:
    """Finite Element Method matrices computed from a mesh.

    These matrices are used for SPDE-based spatial modeling.

    Attributes
    ----------
    c0 : np.ndarray
        Lumped mass matrix diagonal (Voronoi dual area per vertex), shape (n,).
    c1 : sparse.csr_matrix
        Full mass matrix, sparse (n, n).
    g1 : sparse.csr_matrix
        First-order stiffness matrix, sparse (n, n).
    g2 : sparse.csr_matrix, optional
        Second-order stiffness matrix, sparse (n, n). Only computed if order >= 2.
    va : np.ndarray
        Voronoi areas for each vertex, shape (n,).
    ta : np.ndarray
        Triangle areas, shape (n_triangles,).
    order : int
        FEM order used for computation.
    """

    c0: np.ndarray
    c1: sparse.csr_matrix
    g1: sparse.csr_matrix
    va: np.ndarray
    ta: np.ndarray
    order: int = 2
    g2: Optional[sparse.csr_matrix] = None

    @property
    def n(self) -> int:
        """Number of mesh vertices."""
        return len(self.c0)

    @property
    def n_triangle(self) -> int:
        """Number of triangles."""
        return len(self.ta)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "c0": self.c0,
            "c1": self.c1,
            "g1": self.g1,
            "va": self.va,
            "ta": self.ta,
            "order": self.order,
        }
        if self.g2 is not None:
            result["g2"] = self.g2
        return result

    def summary(self) -> str:
        """Return a summary string of the FEM matrices."""
        lines = [
            f"FmFEM:",
            f"  Vertices: {self.n}",
            f"  Triangles: {self.n_triangle}",
            f"  Order: {self.order}",
            f"  c0 (lumped mass): shape ({self.n},), sum={self.c0.sum():.4f}",
            f"  c1 (full mass): shape {self.c1.shape}, nnz={self.c1.nnz}",
            f"  g1 (stiffness): shape {self.g1.shape}, nnz={self.g1.nnz}",
        ]
        if self.g2 is not None:
            lines.append(f"  g2 (stiffness order 2): shape {self.g2.shape}, nnz={self.g2.nnz}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"FmFEM(n={self.n}, n_triangle={self.n_triangle}, order={self.order})"

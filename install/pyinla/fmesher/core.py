"""Core wrapper functions for R's fmesher package using rpy2.

This module provides Python wrappers for R's fmesher package functions,
primarily fm_mesh_2d for creating constrained Delaunay triangulations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import sparse

from .mesh import FmMesh, FmSegment, FmFEM, FmEvaluator
from .exceptions import FmesherNotAvailableError, Rpy2NotAvailableError

# Global flag for rpy2 availability
_HAS_RPY2 = False
_RPY2_ERROR: Optional[str] = None

try:
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.vectors import FloatVector, IntVector, ListVector, Matrix
    from rpy2.rinterface_lib.embedded import RRuntimeError

    _HAS_RPY2 = True
except ImportError as e:
    _RPY2_ERROR = str(e)

# Global reference to the fmesher R package
_fmesher_r: Optional[Any] = None

# Global counter for unique mesh IDs
_mesh_id_counter: int = 0


def _store_r_mesh(r_mesh: Any) -> str:
    """Store R mesh object in R's global environment and return its ID.

    This prevents R garbage collection by keeping the object referenced
    in R's global environment.
    """
    global _mesh_id_counter
    mesh_id = f".pyinla_mesh_{_mesh_id_counter}"
    _mesh_id_counter += 1
    ro.globalenv[mesh_id] = r_mesh
    return mesh_id


def _get_stored_r_mesh(mesh_id: str) -> Optional[Any]:
    """Retrieve stored R mesh object by ID from R's global environment."""
    try:
        return ro.globalenv[mesh_id]
    except Exception:
        return None


def _check_rpy2() -> None:
    """Check if rpy2 is available."""
    if not _HAS_RPY2:
        raise Rpy2NotAvailableError(
            f"rpy2 is not installed. Install with: pip install rpy2\n"
            f"Original error: {_RPY2_ERROR}"
        )


def _get_fmesher() -> Any:
    """Get or initialize the fmesher R package."""
    global _fmesher_r

    _check_rpy2()

    if _fmesher_r is not None:
        return _fmesher_r

    try:
        _fmesher_r = importr("fmesher")
        return _fmesher_r
    except RRuntimeError as e:
        raise FmesherNotAvailableError(
            f"fmesher R package is not installed. Install with R command:\n"
            f"  install.packages('fmesher')\n"
            f"Original error: {e}"
        ) from e


# =============================================================================
# Type conversion utilities
# =============================================================================


def _numpy_to_r_matrix(arr: np.ndarray) -> "Matrix":
    """Convert numpy array to R matrix."""
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    # Flatten column-major for R
    flat = arr.flatten(order="F")
    vec = FloatVector(flat.tolist())

    # Create matrix with correct dimensions
    r_matrix = ro.r["matrix"](vec, nrow=arr.shape[0], ncol=arr.shape[1])
    return r_matrix


def _r_matrix_to_numpy(r_mat: Any) -> np.ndarray:
    """Convert R matrix to numpy array."""
    dims = ro.r["dim"](r_mat)
    if dims == ro.NULL:
        # It's a vector
        return np.array(list(r_mat))

    nrow, ncol = int(dims[0]), int(dims[1])
    flat = np.array(list(r_mat))

    # R stores column-major
    return flat.reshape((nrow, ncol), order="F")


def _segment_to_r(segment: FmSegment) -> Any:
    """Convert FmSegment to R fm_segm object."""
    fmesher = _get_fmesher()

    loc_r = _numpy_to_r_matrix(segment.loc)
    # R uses 1-based indexing
    idx_r = _numpy_to_r_matrix((segment.idx + 1).astype(float))

    # Use R's fm_segm to create a proper segment object with correct class
    r_segm = fmesher.fm_segm(
        loc=loc_r,
        idx=idx_r,
        **{"is.bnd": ro.BoolVector([segment.is_bnd])}
    )

    return r_segm


def _r_segment_to_python(segm_r: Any, is_bnd: bool = True) -> Optional[FmSegment]:
    """Convert R segment object to Python FmSegment."""
    try:
        loc_r = segm_r.rx2("loc")
        idx_r = segm_r.rx2("idx")

        if loc_r == ro.NULL or idx_r == ro.NULL:
            return None

        loc = _r_matrix_to_numpy(loc_r)
        idx = _r_matrix_to_numpy(idx_r).astype(np.int64) - 1  # 0-based

        grp = None
        try:
            grp_r = segm_r.rx2("grp")
            if grp_r != ro.NULL:
                grp = np.array(list(grp_r), dtype=np.int64)
        except Exception:
            pass

        return FmSegment(loc=loc, idx=idx, is_bnd=is_bnd, grp=grp)
    except Exception:
        return None


def _r_mesh_to_fm_mesh(r_mesh: Any) -> FmMesh:
    """Convert R fm_mesh_2d result to Python FmMesh."""
    # Extract loc (vertex locations)
    loc_r = r_mesh.rx2("loc")
    loc = _r_matrix_to_numpy(loc_r)

    # Extract graph components
    graph = r_mesh.rx2("graph")

    # Triangle vertices - R uses 1-based indexing
    tv_r = graph.rx2("tv")
    tv = _r_matrix_to_numpy(tv_r).astype(np.int64) - 1

    # Vertex-vertex edges
    vv = np.empty((0, 2), dtype=np.int64)
    try:
        vv_r = graph.rx2("vv")
        if vv_r != ro.NULL:
            vv = _r_matrix_to_numpy(vv_r).astype(np.int64) - 1
    except Exception:
        pass

    # Triangle-triangle adjacency
    tt = np.empty((0, 3), dtype=np.int64)
    try:
        tt_r = graph.rx2("tt")
        if tt_r != ro.NULL:
            tt_raw = _r_matrix_to_numpy(tt_r).astype(np.int64)
            # Convert 0 to -1 (R uses 0 for no neighbor, we use -1)
            tt = np.where(tt_raw == 0, -1, tt_raw - 1)
    except Exception:
        pass

    # Extract manifold
    manifold = "R2"
    try:
        manifold_r = r_mesh.rx2("manifold")
        if manifold_r != ro.NULL:
            manifold = str(manifold_r[0])
    except Exception:
        pass

    # Extract CRS
    crs = None
    try:
        crs_r = r_mesh.rx2("crs")
        if crs_r != ro.NULL:
            crs = str(crs_r)
    except Exception:
        pass

    # Extract segments
    segm_bnd: List[FmSegment] = []
    segm_int: List[FmSegment] = []

    try:
        segm_r = r_mesh.rx2("segm")
        if segm_r != ro.NULL:
            # Boundary segments
            try:
                bnd_r = segm_r.rx2("bnd")
                if bnd_r != ro.NULL:
                    seg = _r_segment_to_python(bnd_r, is_bnd=True)
                    if seg is not None:
                        segm_bnd.append(seg)
            except Exception:
                pass

            # Interior segments
            try:
                int_r = segm_r.rx2("int")
                if int_r != ro.NULL:
                    seg = _r_segment_to_python(int_r, is_bnd=False)
                    if seg is not None:
                        segm_int.append(seg)
            except Exception:
                pass
    except Exception:
        pass

    # Store R mesh in global storage to prevent garbage collection
    mesh_id = _store_r_mesh(r_mesh)

    return FmMesh(
        loc=loc,
        tv=tv,
        vv=vv,
        tt=tt,
        segm_bnd=segm_bnd,
        segm_int=segm_int,
        manifold=manifold,
        crs=crs,
        _r_mesh=mesh_id,  # Store ID instead of direct reference
    )


# =============================================================================
# Public API functions
# =============================================================================


def fm_mesh_2d(
    loc: Optional[np.ndarray] = None,
    loc_domain: Optional[np.ndarray] = None,
    offset: Optional[Union[float, Sequence[float]]] = None,
    n: Optional[Union[int, Sequence[int]]] = None,
    boundary: Optional[Union[FmSegment, Sequence[FmSegment], np.ndarray]] = None,
    interior: Optional[Union[FmSegment, Sequence[FmSegment], np.ndarray]] = None,
    max_edge: Optional[Union[float, Sequence[float]]] = None,
    min_angle: Optional[Union[float, Sequence[float]]] = None,
    cutoff: Optional[float] = None,
    crs: Optional[str] = None,
) -> FmMesh:
    """Create a 2D triangular mesh using R's fmesher::fm_mesh_2d.

    This function wraps R's fmesher package to create constrained Delaunay
    triangulations with mesh refinement.

    Parameters
    ----------
    loc : np.ndarray, optional
        Matrix of point locations to include in the mesh. Shape (n, 2) or (n, 3).
    loc_domain : np.ndarray, optional
        Matrix of point locations used to determine the domain extent.
    offset : float or sequence of float, optional
        The automatic extension distance. If negative, interpreted as a factor
        of the approximate data diameter. Can be a 2-element sequence for
        inner and outer extension.
    n : int or sequence of int, optional
        Initial number of points on the extended boundary.
    boundary : FmSegment, sequence of FmSegment, or np.ndarray, optional
        Boundary specification. Can be a segment object or coordinate array.
    interior : FmSegment, sequence of FmSegment, or np.ndarray, optional
        Interior constraint specification.
    max_edge : float or sequence of float, optional
        Maximum allowed triangle edge length. Can be 2-element sequence for
        inner and outer regions.
    min_angle : float or sequence of float, optional
        Minimum allowed triangle angle in degrees.
    cutoff : float, optional
        Minimum distance allowed between points.
    crs : str, optional
        Coordinate reference system specification.

    Returns
    -------
    FmMesh
        A mesh object containing vertices, triangles, and adjacency information.

    Raises
    ------
    Rpy2NotAvailableError
        If rpy2 is not installed.
    FmesherNotAvailableError
        If the fmesher R package is not installed.

    Examples
    --------
    >>> import numpy as np
    >>> from fmesher import fm_mesh_2d
    >>>
    >>> # Create mesh from random points
    >>> locs = np.random.randn(100, 2)
    >>> mesh = fm_mesh_2d(loc=locs, max_edge=[0.5, 1.0], cutoff=0.1)
    >>> print(f"Mesh has {mesh.n} vertices and {mesh.n_triangle} triangles")

    >>> # Create mesh with automatic boundary extension
    >>> mesh = fm_mesh_2d(loc=locs, offset=[-0.1, -0.2], max_edge=[0.3, 0.6])
    """
    fmesher = _get_fmesher()

    # Build keyword arguments for R function
    kwargs: Dict[str, Any] = {}

    # Location matrices
    if loc is not None:
        loc_arr = np.asarray(loc, dtype=float)
        if loc_arr.ndim == 1:
            loc_arr = loc_arr.reshape(-1, 2)
        kwargs["loc"] = _numpy_to_r_matrix(loc_arr)

    if loc_domain is not None:
        loc_domain_arr = np.asarray(loc_domain, dtype=float)
        if loc_domain_arr.ndim == 1:
            loc_domain_arr = loc_domain_arr.reshape(-1, 2)
        kwargs["loc.domain"] = _numpy_to_r_matrix(loc_domain_arr)

    # Numeric parameters
    if offset is not None:
        if isinstance(offset, (int, float)):
            kwargs["offset"] = FloatVector([float(offset)])
        else:
            kwargs["offset"] = FloatVector([float(x) for x in offset])

    if n is not None:
        if isinstance(n, int):
            kwargs["n"] = IntVector([n])
        else:
            kwargs["n"] = IntVector(list(n))

    if max_edge is not None:
        if isinstance(max_edge, (int, float)):
            kwargs["max.edge"] = FloatVector([float(max_edge)])
        else:
            kwargs["max.edge"] = FloatVector([float(x) for x in max_edge])

    if min_angle is not None:
        if isinstance(min_angle, (int, float)):
            kwargs["min.angle"] = FloatVector([float(min_angle)])
        else:
            kwargs["min.angle"] = FloatVector([float(x) for x in min_angle])

    if cutoff is not None:
        kwargs["cutoff"] = float(cutoff)

    # Boundary and interior segments
    if boundary is not None:
        if isinstance(boundary, FmSegment):
            kwargs["boundary"] = _segment_to_r(boundary)
        elif isinstance(boundary, np.ndarray):
            # Create segment from coordinates
            seg = FmSegment(loc=boundary, idx=_make_closed_indices(len(boundary)))
            kwargs["boundary"] = _segment_to_r(seg)
        elif isinstance(boundary, (list, tuple)):
            # List of segments
            r_segments = [_segment_to_r(s) for s in boundary]
            kwargs["boundary"] = ro.ListVector(dict(enumerate(r_segments)))

    if interior is not None:
        if isinstance(interior, FmSegment):
            kwargs["interior"] = _segment_to_r(interior)
        elif isinstance(interior, np.ndarray):
            seg = FmSegment(
                loc=interior, idx=_make_closed_indices(len(interior)), is_bnd=False
            )
            kwargs["interior"] = _segment_to_r(seg)
        elif isinstance(interior, (list, tuple)):
            r_segments = [_segment_to_r(s) for s in interior]
            kwargs["interior"] = ro.ListVector(dict(enumerate(r_segments)))

    # Call R function
    r_mesh = fmesher.fm_mesh_2d(**kwargs)

    # Convert result to Python FmMesh
    return _r_mesh_to_fm_mesh(r_mesh)


def _make_closed_indices(n: int) -> np.ndarray:
    """Create closed polygon indices from n vertices."""
    if n < 2:
        return np.empty((0, 2), dtype=np.int64)
    indices = np.column_stack(
        [np.arange(n, dtype=np.int64), np.roll(np.arange(n, dtype=np.int64), -1)]
    )
    return indices


def fm_nonconvex_hull(
    loc: np.ndarray,
    convex: Union[float, Sequence[float]] = -0.15,
    concave: Optional[Union[float, Sequence[float]]] = None,
    resolution: int = 40,
    crs: Optional[str] = None,
) -> FmSegment:
    """Create a non-convex hull boundary using R's fmesher::fm_nonconvex_hull.

    This creates a boundary that follows the shape of the point cloud more
    closely than a convex hull.

    Parameters
    ----------
    loc : np.ndarray
        Matrix of point locations. Shape (n, 2).
    convex : float or sequence, optional
        Convexity parameter. Negative values are relative to data extent.
        Default is -0.15 (15% of extent).
    concave : float or sequence, optional
        Concavity parameter. If None, uses the convex value.
    resolution : int, optional
        Number of points for the hull approximation. Default is 40.
    crs : str, optional
        Coordinate reference system.

    Returns
    -------
    FmSegment
        A boundary segment representing the non-convex hull.

    Notes
    -----
    Requires the 'sf' R package: install.packages('sf')

    Examples
    --------
    >>> import numpy as np
    >>> from fmesher import fm_nonconvex_hull, fm_mesh_2d
    >>>
    >>> locs = np.random.randn(100, 2)
    >>> boundary = fm_nonconvex_hull(locs, convex=-0.1)
    >>> mesh = fm_mesh_2d(loc=locs, boundary=boundary, max_edge=0.5)
    """
    fmesher = _get_fmesher()

    loc_arr = np.asarray(loc, dtype=float)
    if loc_arr.ndim == 1:
        loc_arr = loc_arr.reshape(-1, 2)

    kwargs: Dict[str, Any] = {
        "x": _numpy_to_r_matrix(loc_arr),
        "resolution": resolution,
    }

    if isinstance(convex, (int, float)):
        kwargs["convex"] = float(convex)
    else:
        kwargs["convex"] = FloatVector([float(x) for x in convex])

    if concave is not None:
        if isinstance(concave, (int, float)):
            kwargs["concave"] = float(concave)
        else:
            kwargs["concave"] = FloatVector([float(x) for x in concave])

    r_hull = fmesher.fm_nonconvex_hull(**kwargs)

    # fm_nonconvex_hull returns an sf object - extract coordinates using sf package
    try:
        sf = importr("sf")
    except RRuntimeError as e:
        raise FmesherNotAvailableError(
            "fm_nonconvex_hull requires the 'sf' R package.\n"
            "Install with R command: install.packages('sf')"
        ) from e

    # Get coordinates from sf geometry
    coords_r = sf.st_coordinates(r_hull)
    loc_out = _r_matrix_to_numpy(coords_r)[:, :2]  # Take only X, Y columns

    # Remove last point if it duplicates first (closed polygon)
    n_pts = loc_out.shape[0]
    if n_pts > 1 and np.allclose(loc_out[0], loc_out[-1]):
        loc_out = loc_out[:-1]
        n_pts = loc_out.shape[0]

    idx_out = _make_closed_indices(n_pts)

    return FmSegment(loc=loc_out, idx=idx_out, is_bnd=True, crs=crs)


def fm_segm(
    loc: np.ndarray,
    idx: Optional[np.ndarray] = None,
    is_bnd: bool = True,
) -> FmSegment:
    """Create a mesh segment using R's fmesher::fm_segm.

    Parameters
    ----------
    loc : np.ndarray
        Matrix of point locations.
    idx : np.ndarray, optional
        Two-column matrix of segment indices (0-based). If not provided,
        creates a closed polygon connecting consecutive vertices.
    is_bnd : bool, optional
        Whether this is a boundary segment. Default is True.

    Returns
    -------
    FmSegment
        The created segment.

    Examples
    --------
    >>> import numpy as np
    >>> from fmesher import fm_segm, fm_mesh_2d
    >>>
    >>> # Create a square boundary
    >>> corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    >>> boundary = fm_segm(corners, is_bnd=True)
    >>> mesh = fm_mesh_2d(boundary=boundary, max_edge=0.2)
    """
    fmesher = _get_fmesher()

    loc_arr = np.asarray(loc, dtype=float)
    if loc_arr.ndim == 1:
        loc_arr = loc_arr.reshape(-1, 2)

    kwargs: Dict[str, Any] = {
        "loc": _numpy_to_r_matrix(loc_arr),
        "is.bnd": ro.BoolVector([is_bnd]),
    }

    if idx is not None:
        idx_arr = np.asarray(idx, dtype=np.int64) + 1  # 1-based for R
        kwargs["idx"] = _numpy_to_r_matrix(idx_arr.astype(float))

    r_segm = fmesher.fm_segm(**kwargs)

    # Extract result
    loc_r = r_segm.rx2("loc")
    idx_r = r_segm.rx2("idx")

    loc_out = _r_matrix_to_numpy(loc_r)
    idx_out = _r_matrix_to_numpy(idx_r).astype(np.int64) - 1

    return FmSegment(loc=loc_out, idx=idx_out, is_bnd=is_bnd)


def fm_hexagon_lattice(
    boundary: np.ndarray,
    edge_len: float,
    buffer: float = 0.0,
) -> np.ndarray:
    """Create a hexagonal lattice of points within a boundary.

    This wraps R's fmesher::fm_hexagon_lattice to generate evenly-spaced
    points in a hexagonal pattern within the specified boundary.

    Parameters
    ----------
    boundary : np.ndarray
        Matrix of boundary coordinates. Shape (n, 2).
    edge_len : float
        The edge length of the hexagons (spacing between points).
    buffer : float, optional
        Buffer distance to expand the boundary before generating points.
        Default is 0.0 (no buffer).

    Returns
    -------
    np.ndarray
        Array of (x, y) coordinates of the hexagon lattice points.

    Notes
    -----
    Requires the 'sf' R package: install.packages('sf')

    Examples
    --------
    >>> import numpy as np
    >>> from fmesher import fm_hexagon_lattice, fm_mesh_2d
    >>>
    >>> # Create boundary
    >>> boundary = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
    >>> # Generate hexagon lattice
    >>> points = fm_hexagon_lattice(boundary, edge_len=20)
    >>> # Use points to create mesh
    >>> mesh = fm_mesh_2d(loc=points, max_edge=50)
    """
    fmesher = _get_fmesher()

    try:
        sf = importr("sf")
    except RRuntimeError as e:
        raise FmesherNotAvailableError(
            "fm_hexagon_lattice requires the 'sf' R package.\n"
            "Install with R command: install.packages('sf')"
        ) from e

    # Convert boundary to numpy array
    bnd_arr = np.asarray(boundary, dtype=float)
    if bnd_arr.ndim == 1:
        bnd_arr = bnd_arr.reshape(-1, 2)

    # Close the polygon if not already closed
    if not np.allclose(bnd_arr[0], bnd_arr[-1]):
        bnd_arr = np.vstack([bnd_arr, bnd_arr[0]])

    # Create sf polygon from boundary coordinates
    # Convert to R matrix
    coords_r = _numpy_to_r_matrix(bnd_arr)

    # Create polygon: list(matrix) -> POLYGON -> sfc
    r_list = ro.r['list'](coords_r)
    r_polygon = sf.st_polygon(r_list)
    r_sfc = sf.st_sfc(r_polygon)

    # Apply buffer if specified
    if buffer > 0:
        r_sfc = sf.st_buffer(r_sfc, float(buffer))

    # Call fm_hexagon_lattice
    r_points = fmesher.fm_hexagon_lattice(r_sfc, edge_len=float(edge_len))

    # Extract coordinates from sf points
    coords_r = sf.st_coordinates(r_points)
    coords = _r_matrix_to_numpy(coords_r)[:, :2]  # Take only X, Y columns

    return coords


def get_utm_crs(
    gdf_or_coords: Union[Any, np.ndarray, Tuple[float, float]],
    units: str = "km",
) -> str:
    """Get the appropriate UTM CRS for a location or GeoDataFrame.

    Automatically determines the correct UTM zone based on the centroid
    of the input geometry or coordinates.

    Parameters
    ----------
    gdf_or_coords : GeoDataFrame, ndarray, or tuple
        Input can be:
        - A GeoDataFrame (uses centroid of all geometries)
        - A numpy array of shape (n, 2) with lon/lat coordinates
        - A tuple of (longitude, latitude) for a single point
    units : str, optional
        'km' for kilometers (default), 'm' for meters.
        Using 'km' is recommended for numerical stability in large regions.

    Returns
    -------
    str
        PROJ4 string for the appropriate UTM zone.

    Notes
    -----
    UTM (Universal Transverse Mercator) divides the world into 60 zones,
    each 6 degrees wide. The zone number is calculated from longitude:

        zone = floor((longitude + 180) / 6) + 1

    The hemisphere (north/south) is determined by latitude.

    Examples
    --------
    >>> import geopandas as gpd
    >>> from fmesher import get_utm_crs
    >>>
    >>> # From a GeoDataFrame
    >>> world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    >>> brazil = world[world['name'] == 'Brazil']
    >>> crs = get_utm_crs(brazil, units='km')
    >>> print(crs)
    +proj=utm +zone=23 +south +datum=WGS84 +units=km +no_defs
    >>>
    >>> # From coordinates (lon, lat)
    >>> crs = get_utm_crs((46.7, 24.7), units='km')  # Saudi Arabia
    >>> print(crs)
    +proj=utm +zone=38 +datum=WGS84 +units=km +no_defs
    >>>
    >>> # From numpy array of points
    >>> import numpy as np
    >>> points = np.array([[46.7, 24.7], [35.5, 33.9], [36.3, 33.5]])
    >>> crs = get_utm_crs(points, units='km')
    """
    # Determine centroid based on input type
    if hasattr(gdf_or_coords, "geometry"):
        # It's a GeoDataFrame
        centroid = gdf_or_coords.dissolve().centroid.iloc[0]
        lon, lat = centroid.x, centroid.y
    elif isinstance(gdf_or_coords, np.ndarray):
        # It's a numpy array of coordinates
        if gdf_or_coords.ndim == 1:
            lon, lat = gdf_or_coords[0], gdf_or_coords[1]
        else:
            lon = gdf_or_coords[:, 0].mean()
            lat = gdf_or_coords[:, 1].mean()
    elif isinstance(gdf_or_coords, (tuple, list)):
        # It's a (lon, lat) tuple
        lon, lat = gdf_or_coords[0], gdf_or_coords[1]
    else:
        raise TypeError(
            f"Expected GeoDataFrame, ndarray, or tuple, got {type(gdf_or_coords)}"
        )

    # Calculate UTM zone from longitude
    utm_zone = int((lon + 180) / 6) + 1

    # Determine hemisphere
    south_flag = "" if lat >= 0 else " +south"

    # Build PROJ4 string
    unit_str = "+units=km" if units == "km" else "+units=m"
    crs = f"+proj=utm +zone={utm_zone}{south_flag} +datum=WGS84 {unit_str} +no_defs"

    return crs


def check_fmesher_available() -> Tuple[bool, Optional[str]]:
    """Check if fmesher is available and return status.

    Returns
    -------
    tuple
        (is_available, error_message) - error_message is None if available.

    Examples
    --------
    >>> from fmesher import check_fmesher_available
    >>>
    >>> available, error = check_fmesher_available()
    >>> if available:
    ...     print("fmesher is ready to use!")
    ... else:
    ...     print(f"fmesher not available: {error}")
    """
    if not _HAS_RPY2:
        return False, f"rpy2 not installed: {_RPY2_ERROR}"

    try:
        _get_fmesher()
        return True, None
    except FmesherNotAvailableError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {e}"


def _r_sparse_to_scipy(r_sparse: Any) -> sparse.csr_matrix:
    """Convert R sparse matrix (dgCMatrix or dgTMatrix) to scipy sparse matrix."""
    # Get dimensions
    dims = ro.r["dim"](r_sparse)
    nrow, ncol = int(dims[0]), int(dims[1])

    # Check matrix class
    r_class = ro.r["class"](r_sparse)[0]

    if "dgCMatrix" in r_class:
        # Compressed sparse column format
        # Extract slots: @x (data), @i (row indices, 0-based), @p (column pointers)
        x = np.array(list(r_sparse.slots["x"]))
        i = np.array(list(r_sparse.slots["i"]), dtype=np.int32)
        p = np.array(list(r_sparse.slots["p"]), dtype=np.int32)

        mat = sparse.csc_matrix((x, i, p), shape=(nrow, ncol))
        return mat.tocsr()

    elif "dgTMatrix" in r_class:
        # Triplet format
        x = np.array(list(r_sparse.slots["x"]))
        i = np.array(list(r_sparse.slots["i"]), dtype=np.int32)
        j = np.array(list(r_sparse.slots["j"]), dtype=np.int32)

        mat = sparse.coo_matrix((x, (i, j)), shape=(nrow, ncol))
        return mat.tocsr()

    elif "ddiMatrix" in r_class:
        # Diagonal matrix
        diag = np.array(list(r_sparse.slots["x"]))
        return sparse.diags(diag, format="csr")

    else:
        # Try to convert via as.matrix (dense) - fallback
        r_dense = ro.r["as.matrix"](r_sparse)
        arr = _r_matrix_to_numpy(r_dense)
        return sparse.csr_matrix(arr)


def fm_fem(
    mesh: FmMesh,
    order: int = 2,
) -> FmFEM:
    """Compute Finite Element Method matrices from a mesh.

    This function wraps R's fmesher::fm_fem to compute the mass and stiffness
    matrices needed for SPDE-based spatial modeling.

    Parameters
    ----------
    mesh : FmMesh
        A mesh object created by fm_mesh_2d.
    order : int, optional
        The FEM order. Default is 2. Higher orders give more accurate
        approximations but slower computation.

    Returns
    -------
    FmFEM
        Object containing FEM matrices:
        - c0: Lumped mass matrix diagonal (n,)
        - c1: Full mass matrix, sparse (n, n)
        - g1: First-order stiffness matrix, sparse (n, n)
        - g2: Second-order stiffness matrix, sparse (n, n) if order >= 2
        - va: Voronoi areas for each vertex (n,)
        - ta: Triangle areas (n_triangles,)

    Raises
    ------
    Rpy2NotAvailableError
        If rpy2 is not installed.
    FmesherNotAvailableError
        If the fmesher R package is not installed.

    Notes
    -----
    The FEM matrices are used to construct the SPDE precision matrix Q.
    For the Matern SPDE with alpha=2:
        Q = kappa^4 * C + 2 * kappa^2 * G + G * C^{-1} * G

    where C is the mass matrix and G is the stiffness matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from fmesher import fm_mesh_2d, fm_fem
    >>>
    >>> # Create a mesh
    >>> locs = np.random.randn(100, 2)
    >>> mesh = fm_mesh_2d(loc=locs, max_edge=[0.5, 1.0], cutoff=0.1)
    >>>
    >>> # Compute FEM matrices
    >>> fem = fm_fem(mesh, order=2)
    >>> print(fem.summary())
    """
    fmesher = _get_fmesher()

    # Reconstruct R mesh object from Python FmMesh
    # We need to create an fm_mesh_2d object in R
    loc_r = _numpy_to_r_matrix(mesh.loc)
    tv_r = _numpy_to_r_matrix((mesh.tv + 1).astype(float))  # 1-based indexing

    # Create mesh in R using the graph structure
    r_mesh = ro.r(
        """
        function(loc, tv) {
            mesh <- list()
            mesh$loc <- loc
            mesh$graph <- list(tv = tv)
            mesh$n <- nrow(loc)
            mesh$manifold <- "R2"
            class(mesh) <- c("fm_mesh_2d", "inla.mesh")
            return(mesh)
        }
        """
    )(loc_r, tv_r)

    # Call fm_fem
    r_fem = fmesher.fm_fem(r_mesh, order=order)

    # Extract results
    # c0: lumped mass - sparse diagonal matrix, extract diagonal as vector
    c0_r = r_fem.rx2("c0")
    c0_sparse = _r_sparse_to_scipy(c0_r)
    c0 = np.asarray(c0_sparse.diagonal())

    # c1: full mass matrix (sparse)
    c1_r = r_fem.rx2("c1")
    c1 = _r_sparse_to_scipy(c1_r)

    # g1: first-order stiffness (sparse)
    g1_r = r_fem.rx2("g1")
    g1 = _r_sparse_to_scipy(g1_r)

    # g2: second-order stiffness (sparse) - only if order >= 2
    g2 = None
    if order >= 2:
        try:
            g2_r = r_fem.rx2("g2")
            if g2_r != ro.NULL:
                g2 = _r_sparse_to_scipy(g2_r)
        except Exception:
            pass

    # va: Voronoi areas - matrix, flatten to vector
    va_r = r_fem.rx2("va")
    va = _r_matrix_to_numpy(va_r).flatten()

    # ta: triangle areas - matrix, flatten to vector
    ta_r = r_fem.rx2("ta")
    ta = _r_matrix_to_numpy(ta_r).flatten()

    return FmFEM(
        c0=c0,
        c1=c1,
        g1=g1,
        va=va,
        ta=ta,
        order=order,
        g2=g2,
    )


def fm_evaluator(
    mesh: FmMesh,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    dims: Optional[Union[int, Tuple[int, int], Sequence[int]]] = None,
) -> FmEvaluator:
    """Create an evaluator/projector for mapping mesh values to a regular grid.

    This function wraps R's fmesher::fm_evaluator to create a projection
    matrix that can interpolate values from mesh vertices to a regular grid
    for visualization purposes.

    Parameters
    ----------
    mesh : FmMesh
        A mesh object created by fm_mesh_2d.
    xlim : tuple of (float, float), optional
        X-axis limits (min, max). If not provided, uses mesh bounding box.
    ylim : tuple of (float, float), optional
        Y-axis limits (min, max). If not provided, uses mesh bounding box.
    dims : int or tuple of (int, int), optional
        Grid dimensions. If a single int, uses the same for both dimensions.
        If not provided, defaults to (100, 100).

    Returns
    -------
    FmEvaluator
        An evaluator object containing:
        - x, y: Grid coordinates (1D arrays)
        - proj_A: Projection matrix (sparse)
        - lattice_loc: Grid point locations (n_points, 2)

    Raises
    ------
    Rpy2NotAvailableError
        If rpy2 is not installed.
    FmesherNotAvailableError
        If the fmesher R package is not installed.

    Examples
    --------
    >>> import numpy as np
    >>> from fmesher import fm_mesh_2d, fm_evaluator, fm_evaluate
    >>>
    >>> # Create a mesh
    >>> locs = np.random.randn(100, 2)
    >>> mesh = fm_mesh_2d(loc=locs, max_edge=[0.5, 1.0], cutoff=0.1)
    >>>
    >>> # Create evaluator for a grid
    >>> proj = fm_evaluator(mesh, xlim=(-3, 3), ylim=(-3, 3), dims=(50, 50))
    >>> print(proj.summary())
    >>>
    >>> # Project some values (e.g., from spatial effect)
    >>> field = np.random.randn(mesh.n)
    >>> grid_values = fm_evaluate(proj, field)
    """
    fmesher = _get_fmesher()

    # Set defaults from mesh bounding box
    bbox = mesh.bbox
    if xlim is None:
        xlim = bbox["x"]
    if ylim is None:
        ylim = bbox["y"]
    if dims is None:
        dims = (100, 100)
    elif isinstance(dims, int):
        dims = (dims, dims)
    else:
        dims = tuple(dims)

    # Use the cached R mesh if available (stored in R's global environment)
    # This avoids rpy2 weak reference issues when passing R objects
    mesh_id = mesh._r_mesh

    if mesh_id is not None and mesh_id in ro.globalenv.keys():
        # Call fm_evaluator entirely in R using the stored mesh by name
        # Extract all data within R to avoid rpy2 weak reference issues
        # Pass xlim/ylim as R variables to preserve full precision
        ro.globalenv[".pyinla_xlim"] = FloatVector([float(xlim[0]), float(xlim[1])])
        ro.globalenv[".pyinla_ylim"] = FloatVector([float(ylim[0]), float(ylim[1])])
        ro.globalenv[".pyinla_dims"] = IntVector([int(dims[0]), int(dims[1])])

        r_result = ro.r(
            f"""
            .pyinla_eval <- fm_evaluator(
                {mesh_id},
                xlim = .pyinla_xlim,
                ylim = .pyinla_ylim,
                dims = .pyinla_dims
            )
            .pyinla_A_t <- as(.pyinla_eval$proj$A, "TsparseMatrix")
            list(
                x = .pyinla_eval$x,
                y = .pyinla_eval$y,
                lattice_loc = .pyinla_eval$lattice$loc,
                A_i = .pyinla_A_t@i,
                A_j = .pyinla_A_t@j,
                A_x = .pyinla_A_t@x,
                A_nrow = nrow(.pyinla_eval$proj$A),
                A_ncol = ncol(.pyinla_eval$proj$A)
            )
            """
        )

        # Extract from list result (safer than accessing nested R objects)
        x = np.array(list(r_result.rx2("x")))
        y = np.array(list(r_result.rx2("y")))
        lattice_loc = _r_matrix_to_numpy(r_result.rx2("lattice_loc"))

        # Build sparse matrix from triplets
        A_i = np.array(list(r_result.rx2("A_i")), dtype=np.int64)
        A_j = np.array(list(r_result.rx2("A_j")), dtype=np.int64)
        A_x = np.array(list(r_result.rx2("A_x")))
        A_nrow = int(r_result.rx2("A_nrow")[0])
        A_ncol = int(r_result.rx2("A_ncol")[0])
        proj_A = sparse.csr_matrix((A_x, (A_i, A_j)), shape=(A_nrow, A_ncol))

    else:
        # Reconstruct R mesh object from Python FmMesh with complete graph structure
        loc_r = _numpy_to_r_matrix(mesh.loc)
        tv_r = _numpy_to_r_matrix((mesh.tv + 1).astype(float))  # 1-based indexing

        # Also pass vv and tt if available (for proper boundary handling)
        vv_r = _numpy_to_r_matrix((mesh.vv + 1).astype(float)) if mesh.vv.size > 0 else ro.NULL
        tt_r = _numpy_to_r_matrix((mesh.tt + 1).astype(float)) if mesh.tt.size > 0 else ro.NULL

        # Create mesh in R using the complete graph structure
        r_mesh = ro.r(
            """
            function(loc, tv, vv, tt) {
                mesh <- list()
                mesh$loc <- loc
                mesh$graph <- list(tv = tv)
                if (!is.null(vv)) mesh$graph$vv <- vv
                if (!is.null(tt)) mesh$graph$tt <- tt
                mesh$n <- nrow(loc)
                mesh$manifold <- "R2"
                class(mesh) <- c("fm_mesh_2d", "inla.mesh")
                return(mesh)
            }
            """
        )(loc_r, tv_r, vv_r, tt_r)

        # Call fm_evaluator
        r_evaluator = fmesher.fm_evaluator(
            r_mesh,
            xlim=FloatVector(list(xlim)),
            ylim=FloatVector(list(ylim)),
            dims=IntVector(list(dims)),
        )

        # Extract results from R evaluator object
        # x and y coordinates
        x = np.array(list(r_evaluator.rx2("x")))
        y = np.array(list(r_evaluator.rx2("y")))

        # Projection matrix from $proj$A
        proj_r = r_evaluator.rx2("proj")
        A_r = proj_r.rx2("A")
        proj_A = _r_sparse_to_scipy(A_r)

        # Lattice locations from $lattice$loc
        lattice_r = r_evaluator.rx2("lattice")
        lattice_loc_r = lattice_r.rx2("loc")
        lattice_loc = _r_matrix_to_numpy(lattice_loc_r)

    return FmEvaluator(
        x=x,
        y=y,
        proj_A=proj_A,
        xlim=xlim,
        ylim=ylim,
        dims=dims,
        lattice_loc=lattice_loc,
    )


def fm_evaluate(
    evaluator: FmEvaluator,
    field: np.ndarray,
    reshape: bool = True,
) -> np.ndarray:
    """Project mesh vertex values onto a regular grid.

    This function uses the projection matrix from fm_evaluator to interpolate
    values from mesh vertices to grid points.

    Parameters
    ----------
    evaluator : FmEvaluator
        An evaluator object created by fm_evaluator.
    field : np.ndarray
        Values at mesh vertices, shape (n_vertices,).
    reshape : bool, optional
        If True (default), reshape the result to a 2D grid (nx, ny).
        If False, return a 1D array of length (nx * ny).

    Returns
    -------
    np.ndarray
        Interpolated values at grid points.
        If reshape=True: shape (nx, ny)
        If reshape=False: shape (nx * ny,)

    Notes
    -----
    This function wraps R's fmesher::fm_evaluate. The interpolation is done
    using the sparse projection matrix A, where grid_values = A @ field.

    Points in the grid that are outside the mesh domain will have value 0
    (or NaN if the projection matrix has no non-zeros for that row).

    Examples
    --------
    >>> import numpy as np
    >>> from fmesher import fm_mesh_2d, fm_evaluator, fm_evaluate
    >>>
    >>> # Create mesh and evaluator
    >>> locs = np.random.randn(100, 2)
    >>> mesh = fm_mesh_2d(loc=locs, max_edge=[0.5, 1.0], cutoff=0.1)
    >>> proj = fm_evaluator(mesh, xlim=(-3, 3), ylim=(-3, 3), dims=(50, 50))
    >>>
    >>> # Create some field values (e.g., from SPDE spatial effect)
    >>> field = np.sin(mesh.loc[:, 0]) * np.cos(mesh.loc[:, 1])
    >>>
    >>> # Project to grid
    >>> grid_values = fm_evaluate(proj, field)
    >>> print(f"Grid shape: {grid_values.shape}")
    >>>
    >>> # Visualize with matplotlib
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(grid_values.T, origin='lower',
    ...            extent=[proj.xlim[0], proj.xlim[1], proj.ylim[0], proj.ylim[1]])
    >>> plt.colorbar()
    >>> plt.show()
    """
    field = np.asarray(field).flatten()

    if len(field) != evaluator.proj_A.shape[1]:
        raise ValueError(
            f"Field length ({len(field)}) does not match projection matrix "
            f"columns ({evaluator.proj_A.shape[1]})"
        )

    # Project using sparse matrix multiplication
    grid_values = evaluator.proj_A @ field

    # Set grid points outside the mesh to NaN (matching R's behavior)
    # These are rows where the projection matrix has all zeros (row sum = 0)
    row_sums = np.array(evaluator.proj_A.sum(axis=1)).flatten()
    outside_mask = row_sums == 0
    grid_values[outside_mask] = np.nan

    if reshape:
        # Reshape to 2D grid (nx, ny)
        # Note: R stores in column-major order, so we reshape accordingly
        grid_values = grid_values.reshape(evaluator.shape, order="F")

    return grid_values

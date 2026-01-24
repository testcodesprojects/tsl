# sm.py
"""
Sparse-matrix utilities compatible with INLA's R helpers.

This module mirrors key functions from sm.R:
- inla.as.sparse() / inla.as.dgTMatrix()
- inla.sparse.dim()
- inla.sparse.check()
- inla.sparse.get()
- inla.sm.write() / inla.sm.read()

SciPy COO (coordinate) format plays the role of R's Matrix::dgTMatrix.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple, Union, Optional

import numpy as np

try:
    import scipy.sparse as sp
except Exception as e:  # pragma: no cover
    raise ImportError(
        "scipy is required for sm.py (scipy.sparse). "
        "Install with `pip install scipy`."
    ) from e


__all__ = [
    "inla_as_sparse",
    "inla_as_dgTMatrix",
    "inla_sparse_dim",
    "inla_sparse_check",
    "inla_sparse_get",
    "inla_sm_write",
    "inla_sm_read",
]


ArrayLike = Union[np.ndarray, "sp.spmatrix", list]


def _to_coo(A: ArrayLike) -> "sp.coo_matrix":
    """Convert dense/sparse array-likes to COO with float64 dtype."""
    if sp.issparse(A):
        M = A.tocoo(copy=False)
        if M.dtype != np.float64:
            M = M.astype(np.float64)
        return M
    # Treat pandas DataFrame/Series or lists/ndarrays as dense
    arr = np.asarray(A, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D array; got shape {arr.shape}.")
    return sp.coo_matrix(arr)


def inla_as_sparse(
    A: ArrayLike,
    unique: bool = True,
    na_rm: bool = False,
    zeros_rm: bool = False,
) -> "sp.coo_matrix":
    """
    Convert a dense/sparse matrix into SciPy COO (dgTMatrix analogue).

    Parameters
    ----------
    A : array-like or scipy.sparse.spmatrix
        The matrix to convert.
    unique : bool
        If True, coalesce duplicate (i, j) entries (sum them).
    na_rm : bool
        Replace NaNs in the matrix values with zeros.
    zeros_rm : bool
        Remove explicitly stored zeros from the sparse representation.

    Returns
    -------
    coo_matrix
    """
    M = _to_coo(A)

    # Handle NaNs
    if na_rm:
        # NaNs only live in explicitly stored values (M.data) for sparse inputs
        if M.data.size:
            nan_mask = np.isnan(M.data)
            if nan_mask.any():
                M.data[nan_mask] = 0.0
        else:
            # no stored entries; nothing to do
            pass

    # Ensure uniqueness by summing duplicates
    if unique:
        M.sum_duplicates()

    # Optionally remove stored zeros
    if zeros_rm and M.data.size:
        nz_mask = M.data != 0.0
        if not nz_mask.all():
            M = sp.coo_matrix((M.data[nz_mask], (M.row[nz_mask], M.col[nz_mask])), shape=M.shape)

    # Ensure dtype and format
    if not isinstance(M, sp.coo_matrix):
        M = M.tocoo()
    if M.dtype != np.float64:
        M = M.astype(np.float64)

    return M


# Alias mirroring R's inla.as.dgTMatrix()
def inla_as_dgTMatrix(*args, **kwargs) -> "sp.coo_matrix":
    return inla_as_sparse(*args, **kwargs)


def _dim_from_triplet_text_file(path: str) -> Tuple[int, int]:
    """
    Replicate the R behavior in inla.sparse.dim() when A is a filename
    pointing to a text file with 3 columns: i j Aij.

    If min(i) == 0 or min(j) == 0 we treat indices as 0-based and add +1
    to get the dimension; else we assume 1-based indices.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    # Robust load: allow arbitrary whitespace; no header expected.
    data = np.loadtxt(path, dtype=float)
    if data.ndim == 1:
        if data.size != 3:
            raise ValueError(f"Expected 3 columns in {path}; got {data.size}.")
        data = data.reshape(1, 3)

    i = data[:, 0].astype(np.int64)
    j = data[:, 1].astype(np.int64)

    cindex = 1 if (i.min() == 0 or j.min() == 0) else 0
    nrow = int(i.max() + cindex)
    ncol = int(j.max() + cindex)
    return (nrow, ncol)


def inla_sparse_dim(A: Union[str, ArrayLike]) -> Tuple[int, int]:
    """
    Return dimensions of A. If A is a string, interpret as path to a
    whitespace-delimited (i j Aij) triplet text file as per R code.
    """
    A_checked = inla_sparse_check(A, must_be_squared=False)
    if isinstance(A_checked, str):
        return _dim_from_triplet_text_file(A_checked)
    else:
        return A_checked.shape  # type: ignore[return-value]


def inla_sparse_check(A: Union[str, ArrayLike], must_be_squared: bool = True) -> Union[str, "sp.coo_matrix"]:
    """
    Validate and normalize A into a COO matrix (or return filename).
    Mirrors R's inla.sparse.check().
    """
    if isinstance(A, str):
        if not os.path.exists(A):
            raise FileNotFoundError(f"File not found: {A}")
        return A

    if isinstance(A, (list, dict)):
        raise TypeError("Define matrix using scipy.sparse constructors instead; list/dict format is obsolete!")

    M = inla_as_dgTMatrix(A)
    if must_be_squared:
        if M.shape[0] != M.shape[1]:
            raise ValueError(f"Matrix is not square: {M.shape[0]} x {M.shape[1]}")
    return M


def inla_sparse_get(A: "sp.coo_matrix", row: Optional[int] = None, col: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Extract a specific row or column from a COO matrix, returning
    1-based indices like the R implementation.

    Returns a dict: {'i': ..., 'j': ..., 'values': ...}
    """
    if row is None and col is None:
        return {"i": np.array([], dtype=int), "j": np.array([], dtype=int), "values": np.array([], dtype=float)}

    if row is not None and col is not None:
        raise ValueError("Only one of 'row' and 'col' can be specified.")

    if not isinstance(A, sp.coo_matrix):
        # match R behavior: error and ask to convert up front
        raise TypeError("Matrix is not of type 'coo_matrix'; please convert it with inla_as_dgTMatrix().")

    nrow, ncol = A.shape

    if row is not None:
        if not (1 <= row <= nrow):
            raise IndexError(f"Row out of range: {row}")
        idx = np.flatnonzero(A.row == (row - 1))
        return {"i": np.full(idx.size, row, dtype=int), "j": A.col[idx] + 1, "values": A.data[idx].copy()}

    # column case
    if not (1 <= col <= ncol):
        raise IndexError(f"Column out of range: {col}")
    idx = np.flatnonzero(A.col == (col - 1))
    return {"i": A.row[idx] + 1, "j": np.full(idx.size, col, dtype=int), "values": A.data[idx].copy()}


def inla_sm_write(A: ArrayLike, filename: str = "SparseMatrix.dat") -> str:
    """
    Write sparse matrix to INLA-compatible binary format:
    [int32 nrow, int32 ncol, int32 nnz, int32 (i+1)*nnz, int32 (j+1)*nnz, float64 *nnz]

    This matches R's writeBin with native endianness.
    """
    M = inla_as_sparse(A, unique=True, na_rm=False, zeros_rm=False)
    # Ensure contiguous 1-based indices
    ii = (M.row.astype(np.int64) + 1).astype(np.int32)
    jj = (M.col.astype(np.int64) + 1).astype(np.int32)
    xx = M.data.astype(np.float64, copy=False)
    header = np.array([M.shape[0], M.shape[1], M.nnz], dtype=np.int32)

    with open(filename, "wb") as fp:
        header.tofile(fp)
        ii.tofile(fp)
        jj.tofile(fp)
        xx.tofile(fp)

    return filename


def inla_sm_read(filename: str = "SparseMatrix.dat") -> "sp.coo_matrix":
    """
    Read INLA-compatible binary sparse matrix written by inla_sm_write()
    or R's inla.sm.write().
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)

    with open(filename, "rb") as fp:
        nn = np.fromfile(fp, dtype=np.int32, count=3)
        if nn.size != 3:
            raise IOError("Could not read header (nrow, ncol, nnz).")
        nrow, ncol, nnz = int(nn[0]), int(nn[1]), int(nn[2])

        ii = np.fromfile(fp, dtype=np.int32, count=nnz)
        jj = np.fromfile(fp, dtype=np.int32, count=nnz)
        xx = np.fromfile(fp, dtype=np.float64, count=nnz)

    if not (ii.size == jj.size == xx.size == nnz):
        raise IOError("Unexpected end of file reading sparse matrix payload.")

    # Convert to 0-based
    i0 = (ii.astype(np.int64) - 1).astype(np.int64)
    j0 = (jj.astype(np.int64) - 1).astype(np.int64)
    M = sp.coo_matrix((xx, (i0, j0)), shape=(nrow, ncol))
    # Keep float64 (R writes doubles)
    if M.dtype != np.float64:
        M = M.astype(np.float64)
    return M

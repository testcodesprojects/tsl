# fmesher_io.py
"""
Python equivalents of fmesher_io.R:
- is_fmesher_file(filename)
- read_fmesher_file(filename, verbose=False, debug=False)
- write_fmesher_file(A, filename=..., verbose=False, debug=False, auto_convert=False)
- fmesher_make_dir(dir)
- fmesher_write(m, prefix, matrixname)
- fmesher_read(prefix, matrixname)

Format (all integers are int32 in native endianness)
----------------------------------------------------
header length (int32) = L
then L int32 header fields:
    [0] version
    [1] elems
    [2] nrow
    [3] ncol
    [4] datatype     (0 dense, 1 sparse)
    [5] valuetype    (0 integer, 1 double)
    [6] matrixtype   (0 general, 1 symmetric, 2 diagonal)
    [7] storagetype  (0 rowmajor, 1 columnmajor)

For dense:
    values: nrow*ncol entries, written by rows if rowmajor else by columns

For sparse (columnmajor here):
    i (int32)[elems]   # 0-based
    j (int32)[elems]   # 0-based
    values [elems]     # int32 or float64 depending on valuetype

Symmetric matrices are stored as a triangle (diagonal + strictly lower or
upper). On read, we reconstruct the full symmetric matrix.
"""

from __future__ import annotations

import os
import struct
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

try:
    import scipy.sparse as sp
except Exception:
    sp = None


def is_fmesher_file(filename: str) -> bool:
    if not (isinstance(filename, str) and os.path.exists(filename)):
        return False
    with open(filename, "rb") as fp:
        data = fp.read(4)
        if len(data) < 4:
            return False
        (len_h,) = struct.unpack("=i", data)
    return len_h == 8  # current expectation


def _valuetype_to_dtype(vtype: int):
    return (np.int32 if vtype == 0 else np.float64)


def read_fmesher_file(filename: str, verbose: bool = False, debug: bool = False):
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    if debug:
        verbose = True

    with open(filename, "rb") as fp:
        (len_h,) = struct.unpack("=i", fp.read(4))
        if len_h < 8:
            raise ValueError("Header too short.")
        h_raw = np.fromfile(fp, dtype=np.int32, count=len_h)
        if np.any(h_raw < 0):
            idx = int(np.where(h_raw < 0)[0][0])
            raise ValueError(f"Header entry {idx} is negative: {int(h_raw[idx])}. Unknown meaning.")

        h = {
            "version": int(h_raw[0]),
            "elems": int(h_raw[1]),
            "nrow": int(h_raw[2]),
            "ncol": int(h_raw[3]),
            "datatype": "dense" if h_raw[4] == 0 else "sparse",
            "valuetype": int(h_raw[5]),
            "matrixtype": {0: "general", 1: "symmetric", 2: "diagonal"}[int(h_raw[6])],
            "storagetype": "rowmajor" if h_raw[7] == 0 else "columnmajor",
        }
        if verbose:
            print(h)

        if h["datatype"] == "dense":
            if h["matrixtype"] != "general":
                raise NotImplementedError("Dense & !general not implemented.")
            elems = h["nrow"] * h["ncol"]
            dtype = _valuetype_to_dtype(h["valuetype"])
            Aelm = np.fromfile(fp, dtype=dtype, count=elems)
            if Aelm.size != elems:
                raise IOError(f"Failed to read {elems} elements (got {Aelm.size}).")
            order = "C" if h["storagetype"] == "rowmajor" else "F"
            A = np.reshape(Aelm, (h["nrow"], h["ncol"]), order="C")
            if order == "F":
                A = np.reshape(Aelm, (h["nrow"], h["ncol"]), order="F")
            return A.astype(float, copy=False)

        # sparse
        elems = h["elems"]
        matrixtype = h["matrixtype"]
        stor = h["storagetype"]
        dtype = _valuetype_to_dtype(h["valuetype"])

        if stor == "rowmajor":
            # rowmajor format: entries as triples (i,j,val) in sequence
            i = np.empty(elems, dtype=np.int32)
            j = np.empty(elems, dtype=np.int32)
            v = np.empty(elems, dtype=dtype)
            if matrixtype == "symmetric":
                for k in range(elems):
                    ij = np.fromfile(fp, dtype=np.int32, count=2)
                    i[k] = max(ij[0], ij[1])
                    j[k] = min(ij[0], ij[1])
                    v[k] = np.fromfile(fp, dtype=dtype, count=1)[0]
            elif matrixtype in ("general", "diagonal"):
                for k in range(elems):
                    ij = np.fromfile(fp, dtype=np.int32, count=2)
                    i[k] = ij[0]
                    j[k] = ij[1]
                    v[k] = np.fromfile(fp, dtype=dtype, count=1)[0]
            else:
                raise RuntimeError("Unexpected matrixtype.")
        else:
            # columnmajor: i[], j[], values[]
            i = np.fromfile(fp, dtype=np.int32, count=elems)
            j = np.fromfile(fp, dtype=np.int32, count=elems)
            v = np.fromfile(fp, dtype=dtype, count=elems)
            if i.size != elems or j.size != elems or v.size != elems:
                raise IOError("Unexpected end of file while reading sparse arrays.")

        # Expand for symmetric
        if matrixtype == "symmetric":
            # Should be all lower or all upper
            cond_lower = np.all(i >= j)
            cond_upper = np.all(i <= j)
            if not (cond_lower or cond_upper):
                raise ValueError("Both upper and lower triangle are specified; do not know what to do.")
            off = i != j
            ii = np.concatenate([i, j[off]])
            jj = np.concatenate([j, i[off]])
            vv = np.concatenate([v, v[off]])
            i, j, v = ii, jj, vv
        elif matrixtype == "diagonal":
            diag_mask = (i == j)
            i, j, v = i[diag_mask], j[diag_mask], v[diag_mask]

        # Build COO (indices are 0-based in the file)
        if sp is None:
            raise ImportError("SciPy is required to return a sparse matrix (scipy.sparse).")
        M = sp.coo_matrix((v.astype(float, copy=False), (i, j)), shape=(h["nrow"], h["ncol"]))
        return M


def write_fmesher_file(A: Any,
                       filename: Optional[str] = None,
                       verbose: bool = False,
                       debug: bool = False,
                       auto_convert: bool = False) -> str:
    """
    Write in the fmesher binary format (column-major storage for sparse).
    Types supported:
      - numpy.ndarray (dense)
      - scipy.sparse matrix (any) -> converted to COO
      - dict with keys {'i','j','values'} (1-based 'i','j' are converted to 0-based)
      - 1-D numpy array (diagonal)
    """
    if filename is None:
        fd, filename = tempfile.mkstemp()
        os.close(fd)
        os.unlink(filename)  # create later

    if debug:
        verbose = True

    version = 0

    # auto-convert near-integers to integer dtype if requested
    def affirm_int(arr):
        if not auto_convert:
            return arr
        arr = np.asarray(arr)
        if np.allclose(arr, np.round(arr)):
            return np.round(arr).astype(np.int32)
        return arr

    with open(filename, "wb") as fp:
        if isinstance(A, np.ndarray) and A.ndim == 2:
            nrow, ncol = A.shape
            elems = nrow * ncol
            datatype = 0  # dense
            valuetype = 0 if np.issubdtype(A.dtype, np.integer) else 1
            matrixtype = 0  # general
            storagetype = 1  # columnmajor
            h = np.array([version, elems, nrow, ncol, datatype, valuetype, matrixtype, storagetype], dtype=np.int32)
            if verbose:
                print(h.tolist())

            fp.write(struct.pack("=i", h.size))
            h.astype(np.int32).tofile(fp)

            if valuetype == 0:
                affirm_int(A.astype(np.int64).ravel(order="F")).astype(np.int32).tofile(fp)
            else:
                np.asarray(A, dtype=np.float64).ravel(order="F").tofile(fp)

        elif sp is not None and sp.issparse(A):
            M = A.tocoo()
            nrow, ncol = M.shape
            i = M.row.astype(np.int32)
            j = M.col.astype(np.int32)
            values = M.data

            # Sort in column-major order (by column first, then row within each column)
            # This matches R-INLA's fmesher format convention
            sort_idx = np.lexsort((i, j))
            i = i[sort_idx]
            j = j[sort_idx]
            values = values[sort_idx]

            elems = i.size
            datatype = 1  # sparse
            valuetype = 0 if np.issubdtype(values.dtype, np.integer) else 1
            matrixtype = 0  # general
            storagetype = 1  # columnmajor
            h = np.array([version, elems, nrow, ncol, datatype, valuetype, matrixtype, storagetype], dtype=np.int32)
            if verbose:
                print(h.tolist())
            fp.write(struct.pack("=i", h.size))
            h.tofile(fp)
            i.tofile(fp)
            j.tofile(fp)
            if valuetype == 0:
                affirm_int(values.astype(np.int64)).astype(np.int32).tofile(fp)
            else:
                values.astype(np.float64).tofile(fp)

        elif isinstance(A, dict) and {"i", "j", "values"} <= set(A.keys()):
            i = np.asarray(A["i"], dtype=np.int64)
            j = np.asarray(A["j"], dtype=np.int64)
            # Assume 1-based from R; convert to 0-based
            i = (i - 1).astype(np.int32)
            j = (j - 1).astype(np.int32)
            values = np.asarray(A["values"])
            if "dims" in A and A["dims"] is not None:
                nrow = int(A["dims"][0])
                ncol = int(A["dims"][1])
            else:
                nrow = int(i.max()) + 1
                ncol = int(j.max()) + 1

            elems = i.size
            datatype = 1
            valuetype = 0 if np.issubdtype(values.dtype, np.integer) else 1
            matrixtype = 0
            storagetype = 1
            h = np.array([version, elems, nrow, ncol, datatype, valuetype, matrixtype, storagetype], dtype=np.int32)
            if verbose:
                print(h.tolist())
            fp.write(struct.pack("=i", h.size))
            h.tofile(fp)
            i.tofile(fp)
            j.tofile(fp)
            if valuetype == 0:
                affirm_int(values.astype(np.int64)).astype(np.int32).tofile(fp)
            else:
                values.astype(np.float64).tofile(fp)

        elif isinstance(A, np.ndarray) and A.ndim == 1:
            # diagonal
            n = A.size
            i = np.arange(n, dtype=np.int32)
            j = np.arange(n, dtype=np.int32)
            values = np.asarray(A)
            elems = n
            datatype = 1
            valuetype = 0 if np.issubdtype(values.dtype, np.integer) else 1
            matrixtype = 2  # diagonal
            storagetype = 1
            h = np.array([version, elems, n, n, datatype, valuetype, matrixtype, storagetype], dtype=np.int32)
            if verbose:
                print(h.tolist())
            fp.write(struct.pack("=i", h.size))
            h.tofile(fp)
            i.tofile(fp)
            j.tofile(fp)
            if valuetype == 0:
                affirm_int(values.astype(np.int64)).astype(np.int32).tofile(fp)
            else:
                values.astype(np.float64).tofile(fp)

        else:
            raise TypeError("Unsupported type for A: expected 2-D ndarray, sparse matrix, dict{i,j,values}, or 1-D ndarray")

    return filename


def fmesher_make_dir(dir_path: str) -> str:
    dir_start = dir_path
    k = 1
    while os.path.exists(dir_path):
        dir_path = f"{dir_start}-{k}"
        k += 1
    return dir_path


def fmesher_write(m: Any, prefix: str, matrixname: str) -> str:
    filename = f"{prefix}{matrixname}"
    return write_fmesher_file(m, filename=filename)


def fmesher_read(prefix: str, matrixname: str):
    filename = f"{prefix}{matrixname}"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File '{filename}' does not exist.")
    return read_fmesher_file(filename)


__all__ = [
    "is_fmesher_file",
    "read_fmesher_file",
    "write_fmesher_file",
    "fmesher_make_dir",
    "fmesher_write",
    "fmesher_read",
]

# qinv.py
from __future__ import annotations

import math
from typing import Dict, Optional, Union

import numpy as np

try:
    import scipy.sparse as sp
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False
    sp = None


def _to_dense(Q: Union[np.ndarray, "sp.spmatrix"]) -> np.ndarray:
    if HAVE_SCIPY and sp.isspmatrix(Q):
        return Q.toarray()
    return np.asarray(Q, dtype=float)


def _nullspace(A: np.ndarray, rtol: float = 1e-12) -> np.ndarray:
    if A.size == 0 or A.shape[0] == 0:
        return np.eye(A.shape[1], dtype=float)
    U, s, Vh = np.linalg.svd(A, full_matrices=True)
    rank = (s > (rtol * s.max() if s.size else 0.0)).sum()
    return Vh[rank:].T


def qinv(Q: Union[np.ndarray, "sp.spmatrix"],
         constr: Optional[Dict[str, np.ndarray]] = None,
         jitter: Optional[float] = None,
         return_sparse: bool = False) -> Union[np.ndarray, "sp.csr_matrix"]:
    """
    Python stand-in for inla.qinv(Q, constr, ...).

    - If constr is None: returns full inverse of Q (dense).
    - If constr is given with A (k x n): returns the constrained generalized inverse U S^{-1} U^T,
      where U spans the nullspace of A and S = U^T (Q + δI) U.

    Parameters
    ----------
    Q : array or sparse
        SPD (or intrinsic with constraints).
    constr : dict or None
        {"A": (k x n) array, "e": (k,) array}. e is ignored (e=0 in scaling/inversion).
    jitter : float or None
        Optional diagonal jitter added *only* inside the constrained subspace.
        Default: 1e-12 * mean(diag(Q)).
    return_sparse : bool
        If True and SciPy is available, returns CSR; otherwise dense ndarray.

    Returns
    -------
    ndarray or csr_matrix
    """
    Qd = _to_dense(Q)
    n = Qd.shape[0]

    if constr is None or "A" not in constr or constr["A"] is None:
        # Plain SPD inverse
        Qinv = np.linalg.inv(Qd)
    else:
        A = np.asarray(constr["A"], dtype=float)
        U = _nullspace(A)  # (n x m)
        if U.shape[1] == 0:
            # No admissible subspace -> zero inverse
            Qinv = np.zeros((n, n), dtype=float)
        else:
            if jitter is None:
                jitter = 1e-12 * float(np.trace(Qd)) / n
            S = U.T @ (Qd + jitter * np.eye(n)) @ U
            Sinv = np.linalg.inv(S)
            Qinv = U @ Sinv @ U.T

    if return_sparse and HAVE_SCIPY:
        return sp.csr_matrix(Qinv)
    return Qinv

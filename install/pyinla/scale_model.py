# scale_model.py
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import scipy.sparse as sp
    import scipy.linalg as la
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False
    sp = None
    la = None


# --------------------------- utilities ---------------------------

def _to_dense(Q: Union[np.ndarray, "sp.spmatrix"]) -> np.ndarray:
    if HAVE_SCIPY and sp.isspmatrix(Q):
        return Q.toarray()
    return np.asarray(Q, dtype=float)


def _connected_components_from_Q(Q: Union[np.ndarray, "sp.spmatrix"]) -> List[np.ndarray]:
    """
    Build an undirected graph from the sparsity pattern of Q (off-diagonals).
    Return a list of node-index arrays, one per connected component.
    """
    A = _to_dense(Q)
    n = A.shape[0]
    B = (np.abs(A) > 0).astype(np.int8)
    np.fill_diagonal(B, 0)

    seen = np.zeros(n, dtype=bool)
    comps: List[np.ndarray] = []
    for s in range(n):
        if not seen[s]:
            stack = [s]
            seen[s] = True
            nodes = [s]
            while stack:
                u = stack.pop()
                nbrs = np.nonzero(B[u])[0]
                for v in nbrs:
                    if not seen[v]:
                        seen[v] = True
                        stack.append(v)
                        nodes.append(v)
            comps.append(np.array(sorted(nodes), dtype=np.int64))
    return comps


def _nullspace(A: np.ndarray, rtol: float = 1e-12) -> np.ndarray:
    """
    Orthonormal basis (columns) for the nullspace of A (rows are constraints).
    If A is empty, return I.
    """
    if A.size == 0 or A.shape[0] == 0:
        return np.eye(A.shape[1], dtype=float)
    U, s, Vh = np.linalg.svd(A, full_matrices=True)
    rank = (s > (rtol * s.max() if s.size else 0.0)).sum()
    N = Vh[rank:].T  # (n x (n-rank)), orthonormal columns
    return N


def _rowwise_diag_of_USUt(U: np.ndarray, Sinv: np.ndarray) -> np.ndarray:
    """diag(U S U^T) with S=Sinv given explicitly."""
    V = U @ Sinv
    return (U * V).sum(axis=1)


# --------------------------- public API ---------------------------

def scale_model_internal(Q: Union[np.ndarray, "sp.spmatrix"],
                         constr: Optional[Dict[str, np.ndarray]] = None,
                         eps: float = math.sqrt(np.finfo(float).eps)) -> Dict[str, object]:
    """
    Python version of inla.scale.model.internal(Q, constr, eps)
    Returns: dict(Q=<scaled dense Q>, var=<per-node marginal variances after scaling>)
    The scaling makes the geometric mean of the marg. variances = 1 **in each connected component**.
    """
    Qd = _to_dense(Q).copy()
    n_all = Qd.shape[0]
    comps = _connected_components_from_Q(Qd)
    marg_var = np.zeros(n_all, dtype=float)

    for comp in comps:
        QQ = Qd[np.ix_(comp, comp)]
        n = QQ.shape[0]

        if n == 1:
            QQ[0, 0] = 1.0
            marg_var[comp] = 1.0
            Qd[np.ix_(comp, comp)] = QQ
            continue

        # Restrict (optional) constraints to this component, drop zero rows
        if constr is not None and "A" in constr and constr["A"] is not None:
            A = np.asarray(constr["A"], dtype=float)[:, comp]
            keep = np.where(np.abs(A).sum(axis=1) > 0)[0]
            A = A[keep, :] if keep.size > 0 else A[:0, :]
        else:
            A = np.empty((0, n), dtype=float)

        # Nullspace basis of constraints on this component
        U = _nullspace(A)  # shape (n x m)
        # Stabilize and invert on the constrained subspace
        lam = float(np.max(np.diag(QQ)))
        R = U.T @ (QQ + lam * eps * np.eye(n)) @ U

        # Use Cholesky factorization for numerical stability (like R's GMRFLib)
        if HAVE_SCIPY and la is not None:
            try:
                # cho_factor returns (L, lower) tuple
                c, lower = la.cho_factor(R, lower=True)
                # Solve R @ Sinv = I column by column
                Sinv = la.cho_solve((c, lower), np.eye(R.shape[0]))
            except la.LinAlgError:
                # Fallback to direct inverse
                Sinv = np.linalg.inv(R)
        else:
            Sinv = np.linalg.inv(R)

        # Per-node marginal variances in constrained space
        mvar = _rowwise_diag_of_USUt(U, Sinv)
        # Geometric mean scaling
        fac = math.exp(np.log(mvar).mean())
        QQ_scaled = fac * QQ
        marg_var[comp] = mvar / fac
        Qd[np.ix_(comp, comp)] = QQ_scaled

    return {"Q": Qd, "var": marg_var}


def scale_model(Q: Union[np.ndarray, "sp.spmatrix"],
                constr: Optional[Dict[str, np.ndarray]] = None,
                eps: float = math.sqrt(np.finfo(float).eps)) -> Union[np.ndarray, "sp.spmatrix"]:
    """
    Python version of inla.scale.model(Q, constr, eps) → returns scaled Q (dense ndarray).
    """
    return scale_model_internal(Q=Q, constr=constr, eps=eps)["Q"]

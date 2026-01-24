# pc_bym.py
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from pyinla.scale_model import scale_model_internal

try:
    from scipy.interpolate import UnivariateSpline
    import scipy.linalg as la
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False
    UnivariateSpline = None
    la = None


# --------------------------- helpers ---------------------------

def _to_dense(Q: np.ndarray) -> np.ndarray:
    return np.asarray(Q, dtype=float)


def _connected_components_from_Q(Q: np.ndarray) -> List[np.ndarray]:
    A = _to_dense(Q)
    n = A.shape[0]
    B = (np.abs(A) > 0).astype(np.int8)
    np.fill_diagonal(B, 0)
    seen = np.zeros(n, dtype=bool)
    comps = []
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
    if A.size == 0 or A.shape[0] == 0:
        return np.eye(A.shape[1], dtype=float)
    U, s, Vh = np.linalg.svd(A, full_matrices=True)
    rank = (s > (rtol * s.max() if s.size else 0.0)).sum()
    return Vh[rank:].T


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p) - np.log1p(-p)


def _expit(t: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-t))


# --------------------------- BYM constraints ---------------------------

def bym_constr_internal(Q: np.ndarray, adjust_for_con_comp: bool = True) -> Dict[str, object]:
    """
    Python version of inla.bym.constr.internal
    """
    n = _to_dense(Q).shape[0]
    comps = _connected_components_from_Q(Q)
    sizes = [len(c) for c in comps]
    cc_n1 = sum(1 for s in sizes if s == 1)
    cc_n2 = sum(1 for s in sizes if s >= 2)

    if adjust_for_con_comp:
        k = cc_n2
        A = np.zeros((k, n), dtype=float)
        e = np.zeros(k, dtype=float)
        row = 0
        for c in comps:
            if len(c) >= 2:
                A[row, c] = 1.0
                row += 1
        assert row == k
        rankdef = cc_n2
    else:
        A = np.ones((1, n), dtype=float)
        e = np.ones(1, dtype=float)
        rankdef = cc_n1 + 1

    return dict(
        rankdef=rankdef,
        constr={"A": A, "e": e},
        cc_n=np.array(sizes, dtype=int),
        cc_n1=cc_n1,
        cc_n2=cc_n2
    )


# --------------------------- constrained logdet (proxy) ---------------------------

def sparse_det_bym(Q: np.ndarray,
                   rankdef: Optional[int] = None,
                   adjust_for_con_comp: bool = True,
                   log: bool = True,
                   constr: Optional[Dict[str, np.ndarray]] = None,
                   eps: float = 0.01 * math.sqrt(np.finfo(float).eps)) -> float:
    """
    Proxy for inla.sparse.det.bym: compute log|Q| on the constrained subspace via projection.

    log|Q|_* = log| U^T (Q + εI) U |, where rows(A) span the constraints and U spans Null(A).

    This mimics R's inla.sparse.det.bym which uses inla.qsample to compute:
        logdet = 2.0 * (logdens + (n - rankdef) / 2.0 * log(2*pi))
    where logdens is the log-density at x=0.
    """
    Qd = _to_dense(Q)
    n = Qd.shape[0]
    if constr is None:
        res = bym_constr_internal(Qd, adjust_for_con_comp=adjust_for_con_comp)
        constr = res["constr"]
        if rankdef is None:
            rankdef = res["rankdef"]

    A = np.asarray(constr["A"], dtype=float)
    U = _nullspace(A)
    dmean = float(np.diag(Qd).mean())
    R = U.T @ (Qd + dmean * eps * np.eye(n)) @ U

    # Use scipy.linalg.cho_factor for better numerical stability (matches GMRFLib behavior)
    if HAVE_SCIPY and la is not None:
        try:
            c, lower = la.cho_factor(R, lower=True)
            # logdet = 2 * sum(log(diag(L))) where L is Cholesky factor
            logdet = 2.0 * np.log(np.diag(c)).sum()
        except la.LinAlgError:
            # Fallback to numpy
            L = np.linalg.cholesky(R + 1e-12 * np.eye(R.shape[0]))
            logdet = 2.0 * np.log(np.diag(L)).sum()
    else:
        # Fallback to numpy Cholesky
        try:
            L = np.linalg.cholesky(R)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(R + 1e-12 * np.eye(R.shape[0]))
        logdet = 2.0 * np.log(np.diag(L)).sum()

    return logdet if log else math.exp(logdet)


# --------------------------- BYM scaling wrappers ---------------------------

def scale_model_bym_internal(Q: np.ndarray,
                             eps: float = math.sqrt(np.finfo(float).eps),
                             adjust_for_con_comp: bool = True) -> Dict[str, object]:
    """
    Python version of inla.scale.model.bym.internal.
    Uses general scale_model_internal with a global sum-to-zero constraint inside each block.
    """
    n = _to_dense(Q).shape[0]
    constr = {"A": np.ones((1, n), dtype=float), "e": np.zeros(1, dtype=float)}
    if adjust_for_con_comp:
        return scale_model_internal(Q, constr=constr, eps=eps)
    else:
        # Fallback branch in R when not adjusting per component
        Qd = _to_dense(Q).copy()
        mvar = np.full(n, np.nan, dtype=float)
        idx = np.where(np.diag(Qd) > 0)[0]
        if idx.size > 0:
            QQ = Qd[np.ix_(idx, idx)]
            nloc = QQ.shape[0]
            constr_loc = {"A": np.ones((1, nloc), dtype=float), "e": np.zeros(1, dtype=float)}
            res = scale_model_internal(QQ, constr=constr_loc, eps=eps)
            # res["Q"] already scaled; res["var"] are per-node marg.vars after scaling
            Qd[np.ix_(idx, idx)] = res["Q"]
            mvar[idx] = res["var"]
        return {"Q": Qd, "var": mvar}


def scale_model_bym(Q: np.ndarray,
                    eps: float = math.sqrt(np.finfo(float).eps),
                    adjust_for_con_comp: bool = True) -> np.ndarray:
    return scale_model_bym_internal(Q, eps=eps, adjust_for_con_comp=adjust_for_con_comp)["Q"]


# --------------------------- Laplacian from adjacency ---------------------------

def pc_bym_Q(graph: np.ndarray) -> np.ndarray:
    """
    Python version of inla.pc.bym.Q: Laplacian Q = D - A from an adjacency matrix.
    """
    A = _to_dense(graph)
    np.fill_diagonal(A, 0.0)
    deg = A.sum(axis=1)
    Q = np.diag(deg) - A
    return Q


# --------------------------- PC prior for φ (BYM2 and RW2d+iid) ---------------------------

def pc_bym_phi(*,
               graph: Optional[np.ndarray] = None,
               Q: Optional[np.ndarray] = None,
               eigenvalues: Optional[np.ndarray] = None,
               marginal_variances: Optional[Union[np.ndarray, float]] = None,
               rankdef: Optional[int] = None,
               alpha: float,
               u: float = 0.5,
               lambda_: Optional[float] = None,
               scale_model: bool = True,
               return_as_table: bool = False,
               adjust_for_con_comp: bool = True,
               eps: float = math.sqrt(np.finfo(float).eps),
               debug: bool = False):
    """
    Python version of inla.pc.bym.phi.

    Returns:
      - if return_as_table: {"type":"table", "theta": grid_in_logit, "logprior": logp(theta)}
      - else: callable prior_phi(phi) -> log density at phi in (0,1)
    """
    assert scale_model is True
    assert adjust_for_con_comp is True

    use_eigs = (eigenvalues is not None and marginal_variances is not None)

    if not use_eigs:
        if (graph is None) == (Q is None):
            raise ValueError("Give exactly one of `graph` or `Q` (unless using eigenvalues).")
        Q0 = pc_bym_Q(graph) if Q is None else _to_dense(Q)

        bc = bym_constr_internal(Q0, adjust_for_con_comp=True)
        if rankdef is None:
            rankdef = int(bc["rankdef"])

        n = Q0.shape[0]
        sm = scale_model_bym_internal(Q0, eps=eps, adjust_for_con_comp=True)
        Qs = sm["Q"]
        f = float(np.nanmean(sm["var"]) - 1.0)
        if debug:
            print(f"[pc_bym_phi] n={n} rankdef={rankdef} f={f:.6g}")
    else:
        ev = np.asarray(eigenvalues, dtype=float).ravel()
        n = ev.size
        ev = np.maximum(0.0, np.sort(ev)[::-1])
        if rankdef is None:
            raise ValueError("`rankdef` is required when providing eigenvalues.")
        var = np.asarray(marginal_variances, dtype=float)
        f = float(np.mean(var) - 1.0)
        gamma_invm1 = np.concatenate([1.0 / ev[: (n - rankdef)], np.zeros(rankdef)]) - 1.0

    if use_eigs:
        phi_s = 1.0 / (1.0 + np.exp(-np.linspace(-15, 12, 1000)))
        d = np.empty_like(phi_s)
        for k, phi in enumerate(phi_s):
            aa = n * phi * f
            bb = np.log1p(phi * gamma_invm1).sum()
            d[k] = math.sqrt(aa - bb) if aa >= bb else np.nan
    else:
        phi_s = 1.0 / (1.0 + np.exp(-np.concatenate([np.linspace(-15, 0, 40), np.arange(1, 13)])))
        d = np.empty_like(phi_s)
        log_q1_det = sparse_det_bym(Qs, adjust_for_con_comp=True, constr=bc["constr"], rankdef=rankdef)
        I = np.eye(n)
        for k, phi in enumerate(phi_s):
            aa = n * phi * f
            c = phi / (1.0 - phi)
            log_det_Qc = sparse_det_bym(Qs + c * I, adjust_for_con_comp=True, constr=bc["constr"], rankdef=rankdef)
            bb = (n * math.log((1.0 - phi) / phi) + log_det_Qc - (log_q1_det - n * math.log(phi)))
            d[k] = math.sqrt(aa - bb) if aa >= bb else np.nan
            if debug:
                print(f"phi={phi:.4f} aa={aa:.6g} bb={bb:.6g} d={d[k]:.6g}")

    # Clean and trim first 6 points (as in R)
    ok = np.isfinite(d)
    phi_s, d = phi_s[ok], d[ok]
    if d.size > 6:
        phi_s = phi_s[6:]
        d = d[6:]

    theta = _logit(phi_s)

    # Spline for log(d(theta))
    if HAVE_SCIPY:
        spl = UnivariateSpline(theta, np.log(d), s=0)
        def ff_d(tt: np.ndarray, deriv: int = 0) -> np.ndarray:
            if deriv == 0:
                return np.exp(spl(tt))
            elif deriv == 1:
                return np.exp(spl(tt)) * spl.derivative(1)(tt)
            else:
                raise ValueError("deriv must be 0 or 1")
    else:
        def _interp(tt):
            return np.interp(tt, theta, np.log(d))
        def ff_d(tt: np.ndarray, deriv: int = 0) -> np.ndarray:
            tt = np.asarray(tt)
            if deriv == 0:
                return np.exp(_interp(tt))
            h = 1e-4
            return np.exp(_interp(tt)) * (_interp(tt + h) - _interp(tt - h)) / (2 * h)

    theta_grid = np.linspace(theta.min(), theta.max(), 10000)
    d_grid = ff_d(theta_grid, deriv=0)

    if lambda_ is None:
        du = float(ff_d(_logit(np.array([u])))[0])
        if not (0 < alpha < 1) or not (0 < u < 1):
            raise ValueError("alpha in (0,1) and u in (0,1) are required.")
        lambda_ = -math.log(1.0 - alpha) / du

    log_jac = np.log(np.abs(ff_d(theta_grid, deriv=1)))
    log_prior_theta = np.log(lambda_) - lambda_ * d_grid + log_jac

    # Normalize over theta (trapezoid)
    wt = 0.5 * (np.concatenate([[0], np.diff(theta_grid)]) + np.concatenate([np.diff(theta_grid), [0]]))
    logZ = math.log(np.exp(log_prior_theta).dot(wt))
    log_prior_theta -= logZ

    if return_as_table:
        return {"type": "table", "theta": theta_grid, "logprior": log_prior_theta}

    # log p(phi) = log p(theta) + log |dtheta/dphi| = log p(theta) - log(phi(1-phi))
    if HAVE_SCIPY:
        spl_lp = UnivariateSpline(theta_grid, log_prior_theta, s=0)
        def prior_phi(phi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
            phi = np.asarray(phi, dtype=float)
            if np.any((phi <= 0) | (phi >= 1)):
                raise ValueError("phi must be in (0,1).")
            th = _logit(phi)
            cov = th + 2.0 * np.log1p(np.exp(-th))
            return np.asarray(spl_lp(th) + cov)
    else:
        def prior_phi(phi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
            phi = np.asarray(phi, dtype=float)
            if np.any((phi <= 0) | (phi >= 1)):
                raise ValueError("phi must be in (0,1).")
            th = _logit(phi)
            lp = np.interp(th, theta_grid, log_prior_theta)
            cov = th + 2.0 * np.log1p(np.exp(-th))
            return lp + cov

    return prior_phi


def pc_rw2diid_phi(size: Union[int, Tuple[int, int]],
                   alpha: float,
                   u: float = 0.5,
                   lambda_: Optional[float] = None,
                   return_as_table: bool = False,
                   debug: bool = False):
    """
    Python version of inla.pc.rw2diid.phi using FFT eigenvalues of the RW2d kernel.
    """
    def DFT2(x: np.ndarray) -> np.ndarray:
        return np.fft.fft2(x) / math.sqrt(x.size)
    def IDFT2(x: np.ndarray) -> np.ndarray:
        return np.fft.ifft2(x) / math.sqrt(x.size)
    def make_base_1(sz, delta=0.0):
        r, c = (sz, sz) if np.isscalar(sz) else sz
        base = np.zeros((r, c), dtype=float)
        base[0, 0] = 4.0 + delta
        base[0, 1] = base[1, 0] = base[r - 1, 0] = base[0, c - 1] = -1.0
        return base
    def make_base(sz, delta=0.0):
        base = make_base_1(sz, delta)
        return np.real(IDFT2(DFT2(base) ** 2)) * math.sqrt(base.size)

    # Map size to “FFT friendly” (as in R)
    if np.isscalar(size):
        size = (int(size), int(size))
    size = np.asarray(size, dtype=int)
    size = np.maximum(1, np.floor(1.55 * size)).astype(int)
    good = np.array([
        8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48, 50, 54, 60, 64, 72, 75, 80, 81,
        90, 96, 100, 108, 120, 125, 128, 135, 144, 150, 160, 162, 180, 192, 200, 216, 225, 240, 243, 250, 256, 270, 288,
        300, 320, 324, 360, 375, 384, 400, 405, 432, 450, 480, 486, 500, 512, 540, 576, 600, 625, 640, 648, 675, 720, 729,
        750, 768, 800, 810, 864, 900, 960, 972, 1000, 1024, 1080, 1125, 1152, 1200, 1215, 1250, 1280, 1296, 1350, 1440,
        1458, 1500, 1536, 1600, 1620, 1728, 1800, 1875, 1920, 1944, 2000, 2025, 2048, 2160, 2187, 2250, 2304, 2400, 2430,
        2500, 2560, 2592, 2700, 2880, 2916, 3000, 3072, 3125, 3200, 3240, 3375, 3456, 3600, 3645, 3750, 3840, 3888, 4000,
        4050, 4096
    ], dtype=int)
    sz = []
    for k in range(2):
        idx = int(np.searchsorted(good, size[k], side="left"))
        idx = min(idx, good.size - 1)
        sz.append(int(good[idx]))
    sz = tuple(sz)

    base = make_base(sz)
    # Normalize so that inverse_kernel(base)[0,0] = 1 (like R’s trick)
    eig = np.real(DFT2(base))
    inv = np.zeros_like(eig)
    # rankdef = 1 → drop the smallest one
    flat = np.sort(eig.ravel())
    thr = flat[1] if flat.size > 1 else flat[0]
    inv[eig > thr] = 1.0 / eig[eig > thr]
    inv[eig <= thr] = 0.0
    base = base * (np.real(np.fft.ifft2(inv)) / base.size)[0, 0]

    e_values = np.maximum(0.0, np.real(DFT2(base))).ravel()
    marg_var = 1.0

    return pc_bym_phi(
        eigenvalues=e_values,
        marginal_variances=marg_var,
        return_as_table=return_as_table,
        debug=debug,
        alpha=alpha,
        u=u,
        lambda_=lambda_,
        rankdef=1
    )

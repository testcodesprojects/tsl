# marginal.py
# Python conversion of R/INLA marginal utilities.
from __future__ import annotations

import numpy as np
from typing import Callable, Dict, List, Tuple, Union, Optional

# Try pandas for DataFrame support
try:
    import pandas as pd
    _PANDAS_OK = True
except ImportError:
    _PANDAS_OK = False

# Try SciPy for better interpolation/optimization
try:
    from scipy.interpolate import PchipInterpolator, CubicSpline
    from scipy.optimize import minimize_scalar
    _SCIPY_OK = True
except Exception:  # SciPy not available
    _SCIPY_OK = False

_EPS = np.finfo(float).eps * 1000.0


ArrayLike = Union[np.ndarray, List[float], Tuple[float, ...]]
Marginal = Union[np.ndarray, Dict[str, np.ndarray]]  # Nx2 matrix or {'x','y'}


def inla_is_marginal(marginal: Marginal) -> bool:
    """
    Return True if `marginal` is either:
      * a 2D ndarray with shape (n,2) and n>2
      * a dict with exactly keys {'x','y'}
      * a pandas DataFrame with columns 'x' and 'y'
    """
    if isinstance(marginal, np.ndarray) and marginal.ndim == 2:
        return marginal.shape[1] == 2 and marginal.shape[0] > 2
    if isinstance(marginal, dict) and set(marginal.keys()) == {"x", "y"}:
        return True
    # Check for pandas DataFrame
    if _PANDAS_OK and isinstance(marginal, pd.DataFrame):
        return "x" in marginal.columns and "y" in marginal.columns and len(marginal) > 2
    return False


def _as_xy(marginal: Marginal) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Return (x, y, is_matrix_input).
    """
    # Check for pandas DataFrame first (before dict, since DataFrames support dict-like access)
    if _PANDAS_OK and isinstance(marginal, pd.DataFrame):
        if "x" not in marginal.columns or "y" not in marginal.columns:
            raise ValueError("DataFrame marginal must have columns 'x' and 'y'.")
        x = np.asarray(marginal["x"], dtype=float)
        y = np.asarray(marginal["y"], dtype=float)
        return x, y, False
    elif isinstance(marginal, np.ndarray):
        if marginal.ndim != 2 or marginal.shape[1] != 2:
            raise ValueError("Matrix marginal must be an (n,2) array.")
        x = np.asarray(marginal[:, 0], dtype=float)
        y = np.asarray(marginal[:, 1], dtype=float)
        return x, y, True
    elif isinstance(marginal, dict):
        x = np.asarray(marginal["x"], dtype=float)
        y = np.asarray(marginal["y"], dtype=float)
        return x, y, False
    else:
        raise TypeError("Marginal must be an (n,2) array, dict with keys {'x','y'}, or pandas DataFrame with 'x' and 'y' columns.")


def _to_dict_or_mat(x: np.ndarray, y: np.ndarray, as_matrix: bool) -> Marginal:
    return np.column_stack([x, y]) if as_matrix else {"x": x, "y": y}


def inla_marginal_fix(marginal: Marginal) -> Dict[str, np.ndarray]:
    """
    Remove points with non-positive density or density too small relative to max.
    Matches R logic (no x filtering, only y-based).
    """
    x, y, _ = _as_xy(marginal)

    # Drop NaNs in y (and aligned x)
    mask = ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    if len(y) == 0:
        raise ValueError("Marginal has no finite density points after NaN removal.")

    # Ensure inputs are sorted for downstream interpolation
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    ymax = np.max(y)
    keep = (y > 0.0) & (np.abs(y / ymax) > _EPS)

    x = x[keep]
    y = y[keep]

    if x.size < 3:
        # keep at least 3 points for splines
        raise ValueError("Marginal too small after filtering. Need >=3 support points.")

    return {"x": x, "y": y}


def _pchip(x: np.ndarray, y: np.ndarray):
    if _SCIPY_OK:
        return PchipInterpolator(x, y, extrapolate=False)
    # fallback: linear
    class _Lin:
        def __call__(self, xx):
            return np.interp(xx, x, y, left=np.nan, right=np.nan)
    return _Lin()


def _cubic(x: np.ndarray, y: np.ndarray):
    if _SCIPY_OK:
        return CubicSpline(x, y, bc_type="not-a-knot", extrapolate=False)
    # fallback: linear
    class _Lin:
        def __call__(self, xx):
            return np.interp(xx, x, y, left=np.nan, right=np.nan)
    return _Lin()


def inla_spline(x_or_marginal: Marginal, **kwargs) -> Marginal:
    """
    R's `inla.spline` wrapper is not strictly needed in Python; provided for API parity.
    If given a matrix or dict marginal, return the same structure after cubic interpolation
    onto an equally spaced grid (size inferred from input). Primarily used internally.
    """
    x, y, is_mat = _as_xy(x_or_marginal)
    m = inla_marginal_fix({"x": x, "y": y})
    x, y = m["x"], m["y"]
    n = len(x)
    xx = np.linspace(x.min(), x.max(), num=n)
    # Interpolate log-density and exponentiate (like smarginal)
    S = _cubic(x, np.log(y))
    yy = np.exp(S(xx))
    return _to_dict_or_mat(xx, yy, is_mat)


def inla_smarginal(marginal: Marginal,
                   log: bool = False,
                   extrapolate: float = 0.0,
                   keep_type: bool = False,
                   factor: int = 15) -> Marginal:
    """
    Interpolate a marginal density in log(y) using the 'nonuniform t-map' logic
    found in the R code.
    """
    x_in, y_in, is_mat = _as_xy(marginal)
    m = inla_marginal_fix({"x": x_in, "y": y_in})

    x = m["x"]
    y = m["y"]
    r = np.ptp(x)
    xmin = x.min() - extrapolate * r
    xmax = x.max() + extrapolate * r

    if extrapolate:
        xx = np.concatenate(([xmin], x, [xmax]))
    else:
        xx = x.copy()

    n = max(factor * len(x), len(xx))
    # Build non-uniform t-grid over [xmin, xmax] as in R
    nx = len(xx)
    if nx < 2:
        raise ValueError("Need at least 2 points to build interpolation.")

    dxx = np.diff(xx)
    mean_dx = np.mean(dxx)
    dx_scaled = nx * dxx / mean_dx
    t = np.concatenate(([0.0], np.cumsum(np.sqrt(dx_scaled))))
    # Rescale t to [xmin, xmax]
    t = xmin + (xmax - xmin) * (t - t.min()) / (t.max() - t.min())

    # Build monotone maps t<->x
    t2x = _pchip(t, xx)
    x2t = _pchip(xx, t)

    # Interpolate log(y) on an equally spaced t-grid, then map back to x
    t_lin = np.linspace(t.min(), t.max(), num=n)
    ti = x2t(x)  # inverse map for original support points
    # R's spline() call applies the Hyman filter which preserves the monotone
    # behaviour of the transformed support.  Using the PCHIP variant gives the
    # same shape and keeps the density in sync with R-INLA's output.
    S = _pchip(ti, np.log(y))
    ylog_on_t = S(t_lin)
    x_on_t = t2x(t_lin)

    if log:
        ans = {"x": x_on_t, "y": ylog_on_t}
    else:
        y_on_x = np.exp(ylog_on_t)
        # reapply filter to ensure positive/large enough support
        ans = inla_marginal_fix({"x": x_on_t, "y": y_on_x})

    return _to_dict_or_mat(ans["x"], ans["y"], is_mat and keep_type)


def inla_sfmarginal(marginal: Marginal):
    """
    Return {'range': (xmin,xmax), 'fun': callable} where `fun(x)` is log-density
    over the range; outside the range, you should treat as -inf.
    """
    m = inla_marginal_fix(marginal)
    x = m["x"]
    y = m["y"]
    rng = (x.min(), x.max())
    S = _cubic(x, np.log(y))

    def _fun(xx: ArrayLike):
        xx = np.asarray(xx, dtype=float)
        out = S(xx)
        # outside range -> NaN; callers treat as -inf (density 0)
        return out

    return {"range": rng, "fun": _fun}


def _simpson_odd_grid(x: np.ndarray) -> np.ndarray:
    """
    Build the 'dx' weights used in the R code when applying Simpson-like integration
    on a (possibly) non-equispaced grid.
    """
    dx = np.diff(x)
    return 0.5 * (np.r_[dx, 0.0] + np.r_[0.0, dx])


def inla_emarginal(fun: Callable, marginal: Marginal, *args, **kwargs) -> np.ndarray:
    """
    Compute E[ fun(X) ] for X ~ `marginal`. Works if fun returns a vector per x.
    """
    xx = inla_smarginal(marginal)  # {'x','y'}
    x = xx["x"]
    y = xx["y"]

    n = len(x)
    if n % 2 == 0:
        # Make n odd by dropping the last point as in R
        n = n - 1
        x = x[:n]
        y = y[:n]

    dxw = _simpson_odd_grid(x)

    fvals = fun(x, *args, **kwargs)
    fvals = np.asarray(fvals)
    if fvals.ndim == 1:
        fvals = fvals[:, None]  # (n,1)
    elif fvals.shape[0] != n:
        raise ValueError("fun(x) must return an array with first dimension equal to len(x).")

    integrand = fvals * (y * dxw)[:, None]  # (n,k)

    # Simpson weights (1,4,2,...,4,1)
    i4 = np.arange(1, n - 1, 2)  # 2..n-1 step 2 (0-based indexing later)
    i2 = np.arange(2, n - 1, 2)  # 3..n-2 step 2

    def _simpson_sum(arr: np.ndarray) -> np.ndarray:
        return arr[0] + arr[-1] + 4.0 * arr[i4].sum(axis=0) + 2.0 * arr[i2].sum(axis=0)

    numer = _simpson_sum(integrand)
    denom = _simpson_sum((y * dxw)[:, None])
    out = (numer / denom).squeeze()
    return out


def inla_dmarginal(x: ArrayLike, marginal: Marginal, log: bool = False) -> np.ndarray:
    """
    Evaluate density (or log-density) at points x. Returns 0 (or -inf) outside range.
    """
    f = inla_sfmarginal(inla_smarginal(marginal))
    rng = f["range"]
    fun = f["fun"]

    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)

    inside = (x >= rng[0]) & (x <= rng[1])
    if np.any(inside):
        lv = fun(x[inside])
        if log:
            out[inside] = lv
        else:
            out[inside] = np.exp(lv)

    out[~inside] = (-np.inf if log else 0.0)
    return out


def inla_pmarginal(q: ArrayLike, marginal: Marginal, normalize: bool = True, length: int = 2048) -> np.ndarray:
    """
    Compute P(X <= q). Mirrors R code (including its clamping behavior).
    ``normalize`` is retained for API compatibility and ignored (the result is always normalized).
    """
    f = inla_sfmarginal(inla_smarginal(marginal))
    rng = f["range"]
    fun = f["fun"]

    xx = np.linspace(rng[0], rng[1], num=length)
    ef = np.exp(fun(xx))
    dx = xx[1] - xx[0]
    cdf = np.cumsum(0.5 * (ef[:-1] + ef[1:]) * dx)
    cdf = np.r_[0.0, cdf]
    cdf = cdf / cdf[-1]

    # map x->cdf using monotone spline
    fq = _pchip(xx, cdf)

    q = np.asarray(q, dtype=float)
    qq = np.clip(q, rng[0], rng[1])
    return fq(qq)


def inla_qmarginal(p: ArrayLike, marginal: Marginal, length: int = 2048) -> np.ndarray:
    """
    Compute quantiles by inverting the CDF (monotone interpolation).
    """
    f = inla_sfmarginal(inla_smarginal(marginal))
    rng = f["range"]
    fun = f["fun"]

    xx = np.linspace(rng[0], rng[1], num=length)
    ef = np.exp(fun(xx))
    dx = xx[1] - xx[0]
    cdf = np.cumsum(0.5 * (ef[:-1] + ef[1:]) * dx)
    cdf = np.r_[0.0, cdf]
    cdf = cdf / cdf[-1]

    # Remove duplicated 0's and 1's beyond the first (as in R)
    for val in (0.0, 1.0):
        idx = np.where(np.abs(cdf - val) <= _EPS)[0]
        if idx.size > 1:
            # keep the first
            keep = np.ones_like(cdf, dtype=bool)
            keep[idx[1:]] = False
            cdf = cdf[keep]
            xx = xx[keep]

    # Build inverse cdf using monotone mapping
    F_inv = _pchip(cdf, xx)

    p = np.asarray(p, dtype=float)
    pp = np.clip(p, 0.0, 1.0)
    return F_inv(pp)


def _minimize_on_interval(f: Callable[[float], float], a: float, b: float, tol: float = 1e-6) -> float:
    """
    Minimize f on [a,b]. Use SciPy if present; otherwise do a golden-section search.
    Returns the argmin.
    """
    if _SCIPY_OK:
        res = minimize_scalar(f, bounds=(a, b), method="bounded", options={"xatol": tol})
        return float(res.x)

    # Golden-section search
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = f(c)
    fd = f(d)
    while abs(b - a) > tol * (1.0 + abs(a) + abs(b)):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
    return 0.5 * (a + b)


def inla_hpdmarginal(p: ArrayLike, marginal: Marginal, length: int = 2048) -> np.ndarray:
    """
    Highest Posterior Density (HPD) intervals for unimodal marginals.
    Returns an array of shape (len(p), 2) with columns [low, high].
    """
    sm = inla_smarginal(marginal, keep_type=False)
    f = inla_sfmarginal(sm)
    rng = f["range"]
    fun = f["fun"]

    xx = np.linspace(rng[0], rng[1], num=length)
    dx = xx[1] - xx[0]
    ef = np.exp(fun(xx))

    # Pad zero-density outside range
    xx_ext = np.r_[xx.min() - dx, xx, xx.max() + dx]
    d = np.r_[0.0, ef, 0.0]
    d = np.cumsum(d)
    d = d / d[-1]

    # Remove duplicates at 0 and 1
    for val in (0.0, 1.0):
        idx = np.where(np.abs(d - val) <= _EPS)[0]
        if idx.size > 1:
            keep = np.ones_like(d, dtype=bool)
            keep[idx[1:]] = False
            d = d[keep]
            xx_ext = xx_ext[keep]

    # inverse CDF
    F_inv = _pchip(d, xx_ext)

    p = np.atleast_1d(np.asarray(p, dtype=float))
    pp = 1.0 - np.clip(p, 0.0, 1.0)

    def width(x0: float, conf: float) -> float:
        return float(F_inv(1.0 - conf + x0) - F_inv(x0))

    out = np.empty((pp.size, 2), dtype=float)
    tol = 1e-6
    for i, conf in enumerate(pp):
        x0 = _minimize_on_interval(lambda z: width(z, conf), 0.0, float(conf), tol=tol)
        low = float(F_inv(x0))
        high = float(F_inv(1.0 - conf + x0))
        # Clamp to original marginal range
        low = min(rng[1], max(low, rng[0]))
        high = min(rng[1], max(high, rng[0]))
        out[i, :] = [low, high]
    return out


def inla_rmarginal(n: int, marginal: Marginal, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Random draws from the marginal by inverse-CDF sampling.
    """
    if rng is None:
        rng = np.random.default_rng()
    u = rng.uniform(size=int(n))
    return inla_qmarginal(u, marginal)


def inla_deriv_func(fun: Callable[[ArrayLike], ArrayLike], step_size: float = _EPS ** 0.25) -> Callable[[ArrayLike], np.ndarray]:
    """
    Return a function computing derivative of `fun(x)` using a 5-point stencil.
    Assumes `fun` is vectorized.
    """
    h = float(step_size)

    def fd(x: ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return (-fun(x + 2.0 * h) + 8.0 * fun(x + h) - 8.0 * fun(x - h) + fun(x - 2.0 * h)) / (12.0 * h)

    return fd


def inla_tmarginal(fun: Callable[[ArrayLike], ArrayLike],
                   marginal: Marginal,
                   n: int = 2048,
                   h_diff: float = _EPS ** (1.0 / 3.0),
                   method: str = "quantile") -> Marginal:
    """
    Transform a marginal: Y = fun(X). Output is a new marginal approximation for Y.
    """
    ff = fun
    is_mat = isinstance(marginal, np.ndarray)
    m = inla_smarginal(marginal)
    xrng = (m["x"].min(), m["x"].max())

    method = method.lower()
    if method == "quantile":
        x = inla_qmarginal((np.arange(1, n + 1) / (n + 1.0)), marginal)
    elif method == "linear":
        x = np.linspace(xrng[0], xrng[1], num=n)
    else:
        raise ValueError("Unknown method; use 'quantile' or 'linear'.")

    xx = np.asarray(ff(x), dtype=float)
    fd = inla_deriv_func(ff, step_size=float(h_diff))
    denom = np.clip(np.abs(fd(x)), 1e-12, np.inf)
    dens = inla_dmarginal(x, marginal, log=False) / denom

    # Ensure increasing x for the output marginal
    if xx[0] > xx[-1]:
        xx = xx[::-1]
        dens = dens[::-1]

    return _to_dict_or_mat(xx, dens, is_mat)


def inla_mmarginal(marginal: Marginal) -> float:
    """
    Approximate mode of a marginal. Coarse grid via 1%..99% quantiles then refine.
    """
    p = np.arange(0.01, 1.00, 0.01)
    x = inla_qmarginal(p, marginal)
    dlog = inla_dmarginal(x, marginal, log=True)
    i = int(np.argmax(dlog))
    a = x[max(0, i - 1)]
    b = x[min(len(x) - 1, i + 1)]

    # Maximize log-density on [a,b] => minimize negative
    def neglog(u: float) -> float:
        return float(-inla_dmarginal([u], marginal, log=True)[0])

    xm = _minimize_on_interval(neglog, float(a), float(b), tol=1e-6)
    return xm


def inla_zmarginal(marginal: Marginal, silent: bool = False) -> Dict[str, float]:
    """
    Summary: mean, sd, mode, selected quantiles. Prints (unless silent) and returns dict.
    """
    if not inla_is_marginal(marginal):
        raise ValueError("Input is not a valid marginal (matrix Nx2 or dict {'x','y'}).")

    m1_m2 = inla_emarginal(lambda xx: np.column_stack([xx, xx ** 2]), marginal)
    mean = float(m1_m2[0])
    sd = float(np.sqrt(max(0.0, m1_m2[1] - mean ** 2)))
    qs = inla_qmarginal([0.025, 0.25, 0.5, 0.75, 0.975], marginal)
    mode = inla_mmarginal(marginal)

    if not silent:
        print(f"Mean            {mean:.6g}")
        print(f"Stdev           {sd:.6g}")
        print(f"Quantile  0.025 {qs[0]:.6g}")
        print(f"Quantile  0.25  {qs[1]:.6g}")
        print(f"Quantile  0.5   {qs[2]:.6g}")
        print(f"Quantile  0.75  {qs[3]:.6g}")
        print(f"Quantile  0.975 {qs[4]:.6g}")
        print(f"Mode            {mode:.6g}")

    return {
        "mean": mean,
        "sd": sd,
        "quant0.025": float(qs[0]),
        "quant0.25": float(qs[1]),
        "quant0.5": float(qs[2]),
        "quant0.75": float(qs[3]),
        "quant0.975": float(qs[4]),
        "mode": float(mode),
    }


# Aliases for completeness
def inla_marginal_transform(fun, marginal, n: int = 2048, h_diff: float = _EPS ** (1.0 / 3.0), method: str = "quantile"):
    return inla_tmarginal(fun, marginal, n=n, h_diff=h_diff, method=method)

# create_data_file.py
from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .sections import inla_write_fmesher_file

import numpy as np
import pandas as pd


# ---------------------------
# Small helpers / shims
# ---------------------------

def _one_of(x: str, group: Union[str, Sequence[str]]) -> bool:
    """Case-sensitive membership check."""
    if isinstance(group, str):
        return x == group
    return any(x == g for g in group)


def _as_dataframe(y_orig: Any) -> pd.DataFrame:
    """Convert various inputs to a pandas DataFrame."""
    if y_orig is None:
        raise ValueError("y_orig is None and no fallback provided via `mf`.")

    if isinstance(y_orig, pd.DataFrame):
        return y_orig.copy()

    if isinstance(y_orig, pd.Series):
        return y_orig.to_frame()

    if isinstance(y_orig, dict):
        # Interpreted as column-wise mapping
        # Filter out non-array keys (like '_class') and None values (like 'cure' when absent)
        filtered = {k: v for k, v in y_orig.items()
                    if v is not None and not isinstance(v, str)}
        return pd.DataFrame(filtered)

    arr = np.asarray(y_orig)
    if arr.ndim == 1:
        return pd.DataFrame({"Y": arr})
    elif arr.ndim == 2:
        # Generate generic column labels Y1..Yk
        cols = [f"Y{i+1}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=cols)
    else:
        raise TypeError("Unsupported y_orig type; cannot convert to DataFrame.")


def _ensure_len(name: str, val: Optional[Union[float, int, Sequence[float]]], n: int) -> Optional[np.ndarray]:
    """Broadcast scalar to length n; validate length if sequence."""
    if val is None:
        return None
    if callable(val):
        # The R function ignores `weights` if it's a function; mirror that behavior upstream
        return None
    arr = np.asarray(val).astype(float)
    if arr.size == 1:
        return np.repeat(arr.item(), n)
    if arr.size != n:
        raise ValueError(f"Length of `{name}` must equal number of observations: {arr.size} != {n}")
    return arr


def _touch(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb"):
        pass


def _mktemp(dirpath: Optional[str]) -> str:
    os.makedirs(dirpath or tempfile.gettempdir(), exist_ok=True)
    fd, path = tempfile.mkstemp(dir=dirpath or tempfile.gettempdir(), suffix=".dat")
    os.close(fd)
    return path


def _mask_datadir(path: str, data_dir: Optional[str]) -> str:
    if data_dir is None:
        return path
    # Replace the directory prefix with the INLA placeholder
    data_dir = os.path.abspath(data_dir)
    path_abs = os.path.abspath(path)
    if path_abs.startswith(data_dir):
        return path_abs.replace(data_dir, "$inladatadir", 1)
    return path


def _is_surv_like(df: pd.DataFrame) -> bool:
    cols = set(df.columns)
    return "time" in cols and ({"event", "lower", "upper", "truncation"} & cols)


def _ensure_int_array(name: str, val: Union[Sequence[int], np.ndarray], n: int) -> np.ndarray:
    arr = np.asarray(val)
    if arr.size == 1:
        arr = np.repeat(int(arr.item()), n)
    if arr.size != n:
        raise ValueError(f"Length of `{name}` must equal number of observations: {arr.size} != {n}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"`{name}` contains NaN/Inf.")
    if not np.all(np.floor(arr) == arr):
        raise ValueError(f"`{name}` must be integer-valued.")
    return arr.astype(int)


def _drop_rows_on_first_y_nan(mat: np.ndarray, y_col_index_0based: int) -> np.ndarray:
    mask = ~np.isnan(mat[:, y_col_index_0based])
    return mat[mask, :]


def _grep_columns(df: pd.DataFrame, pattern: str) -> List[str]:
    prog = re.compile(pattern)
    return [c for c in df.columns if prog.match(c)]


# --- optional stubs you can wire up to your model registry ---

@dataclass
class LikelihoodProps:
    survival: bool
    hyper_count: int


def inla_model_properties(model: str, section: str) -> LikelihoodProps:
    """
    Minimal stub for `inla.model.properties`. Replace with your own lookup.
    Only used by a few branches (`nmix`, survival families, etc.).
    """
    # Fallback assumptions that keep most branches functional.
    if section == "likelihood":
        if model in {"exponentialsurv", "weibullsurv", "loglogisticsurv", "qloglogisticsurv",
                     "lognormalsurv", "gammasurv", "mgammasurv", "gammajwsurv",
                     "fmrisurv", "gompertzsurv", "dgompertzsurv"}:
            return LikelihoodProps(survival=True, hyper_count=0)
        if model in {"nmix"}:
            return LikelihoodProps(survival=False, hyper_count=5)   # sensible default, adjust
        if model in {"nmixnb"}:
            return LikelihoodProps(survival=False, hyper_count=6)   # nb adds one overdisp
        if model in {"bgev"}:
            return LikelihoodProps(survival=False, hyper_count=4)   # placeholder
    # Generic default
    return LikelihoodProps(survival=False, hyper_count=0)


# ---------------------------
# Main function
# ---------------------------

def create_data_file(
    y_orig: Optional[Any] = None,
    MPredictor: Optional[Any] = None,   # unused here; kept for API parity
    mf: Optional[Any] = None,
    scale: Optional[Union[float, Sequence[float]]] = None,
    weights: Optional[Union[float, Sequence[float], Any]] = None,
    E: Optional[Union[float, Sequence[float]]] = None,
    Ntrials: Optional[Union[int, Sequence[int], np.ndarray]] = None,
    strata: Optional[Union[int, Sequence[int]]] = None,
    lp_scale: Optional[Union[float, Sequence[float]]] = None,
    event: Optional[Any] = None,        # (unused here; kept for parity)
    family: Optional[str] = None,
    data_dir: Optional[str] = None,
    file: Optional[str] = None,         # previous/temporary path to remove if error
    debug: bool = False,
    reuse_names: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Python port of R's `inla.create.data.file`.

    Returns:
        dict with 'file.data', 'file.weights', 'file.attr', 'file.lp.scale' paths.
        Paths are returned with `$inladatadir` prefix substitution if `data_dir` is given.
    """

    created_paths: List[str] = []

    def my_stop(msg: str):
        # Remove created files and any provided `file` path if exists.
        for f in [file, data_dir]:
            try:
                if f and os.path.exists(f) and os.path.isfile(f):
                    os.remove(f)
            except Exception:
                pass
        for p in created_paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        raise ValueError(msg)

    # --- y.attr handling (INLA multi-response metadata) ---
    # In R: attr(y.orig, "inla.ncols") else 0. Here we try pandas .attrs or a Python attribute.
    y_attr = None
    if hasattr(y_orig, "attrs") and isinstance(getattr(y_orig, "attrs"), dict):
        y_attr = y_orig.attrs.get("inla.ncols", None)
    if y_attr is None and hasattr(y_orig, "inla_ncols"):
        y_attr = getattr(y_orig, "inla_ncols")
    if y_attr is None:
        y_attr_vec = np.array([0], dtype=float)
    else:
        y_attr_arr = np.atleast_1d(np.asarray(y_attr))
        y_attr_vec = y_attr_arr.astype(float)

    # --- build y dataframe ---
    if y_orig is None:
        if mf is None:
            my_stop("`y_orig` is None and no `mf` provided.")
        mf_df = mf if isinstance(mf, pd.DataFrame) else pd.DataFrame(mf)
        y_df = mf_df.iloc[:, [0]].copy()
    else:
        y_df = _as_dataframe(y_orig)

        # Drop 'cure' column if present and empty (per R logic for surv without cure)
        if "cure" in y_df.columns and y_df["cure"].isna().all():
            y_df = y_df.drop(columns=["cure"])

    n_data = y_df.shape[0]
    ind = np.arange(n_data, dtype=int)  # 0..n-1, same as R

    if debug:
        print(f"[create.data.file] n.data = {n_data}")

    # --- weights (if provided as vector) ---
    if weights is not None and not callable(weights):
        weights_vec = _ensure_len("weights", weights, n_data)
    else:
        weights_vec = None

    # --- lp.scale ---
    if lp_scale is not None:
        lp_vec = _ensure_len("lp.scale", lp_scale, n_data).astype(float)
        lp_vec = np.where(np.isnan(lp_vec), 0.0, lp_vec)
        lp_vec = np.maximum(lp_vec, 0.0)
        lp_max = float(np.max(lp_vec)) if lp_vec.size > 0 else 0.0
        if debug:
            print(f"[create.data.file] lp.scale: max={lp_max:.6g}")
    else:
        lp_vec = None

    fam = str(family) if family is not None else None
    if fam is None:
        my_stop("`family` must be provided.")

    # --- Family groups (mirroring the R code) ---
    fam_gauss_like = {
        "gaussian", "stdgaussian", "normal", "stdnormal", "lognormal",
        "gengaussian", "exppower", "t", "sn", "gev", "logistic",
        "circularnormal", "wrappedcauchy", "iidgamma", "simplex",
        "gamma", "mgamma", "beta", "obeta", "tweedie", "fmri", "vm"
    }
    fam_sem = {"sem"}
    fam_bcgauss = {"bcgaussian"}
    fam_tstrata = {"tstrata"}
    fam_poisson_like = {
        "poisson", "npoisson", "nzpoisson", "cenpoisson", "gammacount",
        "gpoisson", "xpoisson",
        "zeroinflatedcenpoisson0", "zeroinflatedcenpoisson1",
        "zeroinflatednbinomial0", "zeroinflatednbinomial1", "zeroinflatednbinomial2",
        "zeroinflatedpoisson0", "zeroinflatedpoisson1", "zeroinflatedpoisson2",
        "poisson.special1", "bell"
    }
    fam_nbinomial = {"nbinomial"}
    fam_surv_basic = {"exponential", "weibull", "loglogistic", "gammajw", "gompertz"}
    fam_zinb_strata = {"zeroinflatednbinomial1strata2", "zeroinflatednbinomial1strata3"}
    fam_xbinom = {"xbinomial"}
    fam_binomial = {
        "binomial", "binomialtest", "betabinomial", "nbinomial2",
        "zeroinflatedbinomial0", "zeroinflatedbinomial1", "zeroinflatedbinomial2",
        "zeroninflatedbinomial2", "zeroninflatedbinomial3",
        "zeroinflatedbetabinomial0", "zeroinflatedbetabinomial1", "zeroinflatedbetabinomial2"
    }
    fam_betabinomialna = {"betabinomialna"}
    fam_cbinomial = {"cbinomial"}
    fam_survival = {
        "exponentialsurv", "weibullsurv", "loglogisticsurv", "qloglogisticsurv", "lognormalsurv",
        "gammasurv", "mgammasurv", "gammajwsurv", "fmrisurv",
        "gompertzsurv", "dgompertzsurv"
    }
    fam_y_only = {
        "stochvol", "stochvolln", "stochvolt", "stochvolnig", "stochvolsn",
        "loggammafrailty", "iidlogitbeta", "qkumar", "qloglogistic",
        "gp", "dgp", "pom", "egp", "logperiodogram"
    }
    fam_nmix = {"nmix", "nmixnb"}
    fam_agaussian = {"agaussian"}
    fam_fl = {"fl"}
    fam_rcpoisson = {"rcpoisson", "tpoisson"}
    fam_cenpoisson2 = {"cenpoisson2"}
    fam_cennbinomial2 = {"cennbinomial2"}
    fam_gaussianjw = {"gaussianjw"}
    fam_ggaussian = {"ggaussian"}
    fam_ggaussianS = {"ggaussianS"}
    fam_zero_prefix = {"0poisson", "0poissonS", "0binomial", "0binomialS"}
    fam_binomialmix = {"binomialmix"}
    fam_occupancy = {"occupancy"}
    fam_cloglike = {"cloglike"}
    fam_bgev = {"bgev"}

    # --- Build `response` matrix per family ---
    response: Optional[np.ndarray] = None

    # Common helpers to replicate R's `cbind(ind, ...)` behavior
    def _cbind(*cols: np.ndarray) -> np.ndarray:
        arrs = [np.asarray(c).reshape(-1, 1) if c.ndim == 1 else np.asarray(c) for c in cols]
        return np.hstack(arrs)

    # Family implementations:

    if _one_of(fam, fam_gauss_like):
        s = _ensure_len("scale", scale, n_data)
        if s is None:
            s = np.ones(n_data)
        y = y_df.to_numpy()
        response = _cbind(ind, s, y)
        # Drop rows where the *first* y column is NA
        response = _drop_rows_on_first_y_nan(response, 2)
        if np.isnan(response).any():
            my_stop(f"family:{fam}. NA's in argument 'scale' are not allowed")

    elif _one_of(fam, fam_sem):
        y = y_df.to_numpy()
        response = _cbind(ind, y)
        response = _drop_rows_on_first_y_nan(response, 1)

    elif _one_of(fam, fam_bcgauss):
        # y.orig is (y, mean, scale); reorder to (ind, mean, scale, y)
        if y_df.shape[1] != 3:
            my_stop("For family='bcgaussian', y_orig must have exactly 3 columns: (y, mean, scale).")
        y = y_df.to_numpy()
        response = _cbind(ind, y[:, 1], y[:, 2], y[:, 0])
        response = _drop_rows_on_first_y_nan(response, 3)
        if np.isnan(response).any():
            my_stop(f"family:{fam}. NA's in argument 'scale' or 'mean' are not allowed")

    elif _one_of(fam, fam_tstrata):
        s = _ensure_len("scale", scale, n_data)
        if s is None:
            s = np.ones(n_data)
        st = _ensure_int_array("strata", strata if strata is not None else 1, n_data)
        if np.any(st <= 0):
            my_stop("'strata' must be > 0 for all entries.")
        y = y_df.to_numpy()
        response = _cbind(ind, s, st - 1, y)
        response = _drop_rows_on_first_y_nan(response, 3)
        if np.isnan(response).any():
            my_stop(f"family:{fam}. NA's in argument 'scale' or 'strata' are not allowed")

    elif _one_of(fam, fam_poisson_like):
        e = _ensure_len("E", E, n_data)
        if e is None:
            e = np.ones(n_data)
        y = y_df.to_numpy()
        response = _cbind(ind, e, y)
        response = _drop_rows_on_first_y_nan(response, 2)
        if np.isnan(response).any():
            my_stop(f"family:{fam}. NA's in argument 'E' are not allowed")

    elif _one_of(fam, fam_nbinomial):
        e = _ensure_len("E", E, n_data)
        s = _ensure_len("scale", scale, n_data)
        if e is None:
            e = np.ones(n_data)
        if s is None:
            s = np.ones(n_data)
        y = y_df.to_numpy()
        response = _cbind(ind, e, s, y)
        response = _drop_rows_on_first_y_nan(response, 3)
        if np.isnan(response).any():
            my_stop(f"family:{fam}. NA's in argument 'E' or 'scale' are not allowed")

    elif _one_of(fam, fam_surv_basic):
        y = y_df.to_numpy()
        response = _cbind(ind, y)
        response = _drop_rows_on_first_y_nan(response, 1)

    elif _one_of(fam, fam_zinb_strata):
        e = _ensure_len("E", E, n_data)
        if e is None:
            e = np.ones(n_data)
        st = _ensure_int_array("strata", strata if strata is not None else 1, n_data)
        if not np.all(np.isin(st, np.arange(1, 11))):
            my_stop("'strata' must be in 1..10 for this family.")
        y = y_df.to_numpy()
        response = _cbind(ind, e, st - 1, y)
        response = _drop_rows_on_first_y_nan(response, 3)
        if np.isnan(response).any():
            my_stop(f"family:{fam}. NA's in argument 'scale' or 'strata' are not allowed")

    elif _one_of(fam, fam_xbinom):
        s = _ensure_len("scale", scale, n_data)
        if s is None:
            s = np.ones(n_data)
        nt = _ensure_len("Ntrials", Ntrials if Ntrials is not None else 1, n_data)
        y = y_df.to_numpy()
        response = _cbind(ind, nt, s, y)
        response = _drop_rows_on_first_y_nan(response, 3)
        if np.isnan(response).any():
            my_stop("NA's in argument 'Ntrials' or 'scale' are not allowed")

    elif _one_of(fam, fam_binomial):
        nt = _ensure_len("Ntrials", Ntrials if Ntrials is not None else 1, n_data)
        y = y_df.to_numpy()
        response = _cbind(ind, nt, y)
        response = _drop_rows_on_first_y_nan(response, 2)
        if np.isnan(response).any():
            my_stop("NA's in argument 'Ntrials' are not allowed")

    elif _one_of(fam, fam_betabinomialna):
        nt = _ensure_len("Ntrials", Ntrials if Ntrials is not None else 1, n_data)
        s = _ensure_len("scale", scale if scale is not None else 1.0, n_data)
        y = y_df.to_numpy()
        response = _cbind(ind, nt, s, y)
        response = _drop_rows_on_first_y_nan(response, 3)
        if np.isnan(response).any():
            my_stop(f"family:{fam}. NA's in argument 'Ntrials' or 'scale' are not allowed")

    elif _one_of(fam, fam_cbinomial):
        # Ntrials must be (n x 2) matrix
        Nt = np.asarray(Ntrials)
        if Nt.ndim != 2 or Nt.shape != (n_data, 2):
            my_stop(f"Argument 'Ntrials' for family='cbinomial' must be an {n_data}x2 matrix.")
        y = y_df.to_numpy()
        response = _cbind(ind, Nt, y)
        response = _drop_rows_on_first_y_nan(response, 3)
        if np.isnan(response).any():
            my_stop(f"family:{fam}. NA's in argument 'Ntrials' are not allowed")

    elif _one_of(fam, fam_survival):
        props = inla_model_properties(fam, "likelihood")
        if not props.survival:
            my_stop("Likely a configuration issue: expected a survival likelihood.")
        if not _is_surv_like(y_df):
            my_stop("Response must contain columns: time (and event/lower/upper/truncation as applicable).")

        df = y_df.copy()
        n = len(df["time"])
        for col, default in (("truncation", 0.0), ("lower", 0.0), ("upper", np.inf), ("event", 1.0)):
            if col not in df.columns:
                df[col] = default

        # Cure columns prefixed "cure" or "cure." + digits
        cure_cols = [c for c in df.columns if re.match(r"^cure\.?[0-9]*$", c)]
        for c in cure_cols:
            df[c] = df[c].fillna(0)

        idx = df["time"].notna().to_numpy()
        # Compose [IDX, event, truncation, lower, upper, cure..., time]
        pieces = [
            ind[idx],
            df.loc[idx, "event"].to_numpy(),
            df.loc[idx, "truncation"].to_numpy(),
            df.loc[idx, "lower"].to_numpy(),
            df.loc[idx, "upper"].to_numpy(),
        ]
        for c in cure_cols:
            pieces.append(df.loc[idx, c].to_numpy())
        pieces.append(df.loc[idx, "time"].to_numpy())
        response = _cbind(*[np.asarray(p) for p in pieces])

        if np.isnan(response).any():
            my_stop(f"family:{fam}. NA's in truncation/event/lower/upper/time are not allowed")

    elif _one_of(fam, fam_y_only):
        y = y_df.to_numpy()
        response = _cbind(ind, y)
        response = _drop_rows_on_first_y_nan(response, 1)

    elif _one_of(fam, fam_nmix):
        # Requires knowledge of hyper count (mmax) from model properties.
        props = inla_model_properties(fam, "likelihood")
        if fam == "nmix":
            mmax = props.hyper_count
        elif fam == "nmixnb":
            mmax = max(props.hyper_count - 1, 1)
        else:
            my_stop("Unexpected family in nmix-branch.")

        df = y_df.copy()
        df.insert(0, "IDX", ind)
        col_idx = ["IDX"]
        col_x = _grep_columns(df, r"^X[0-9]+$")
        col_y = _grep_columns(df, r"^Y[0-9]+$")

        if not (1 <= len(col_x) <= mmax):
            my_stop(f"Number of X-columns {len(col_x)} must be in [1, {mmax}] for {fam}.")

        # Remove rows with all Y's NA
        na_y = df[col_y].isna().all(axis=1)
        df = df.loc[~na_y, :].copy()

        X = df[col_x].to_numpy()
        Y = df[col_y].to_numpy()
        idxc = df[col_idx].to_numpy()

        # Replace NA in X with 0
        X = np.where(np.isnan(X), 0.0, X)
        # Augment X up to mmax columns with NaNs
        if X.shape[1] < mmax:
            X = np.hstack([X, np.full((X.shape[0], mmax - X.shape[1]), np.nan)])

        # Sort each row of Y so that NaNs are at the end (and non-NaNs sorted asc)
        Y_sorted = []
        for row in Y:
            vals = row[~np.isnan(row)]
            vals.sort()
            pad = np.full(row.size - vals.size, np.nan)
            Y_sorted.append(np.r_[vals, pad])
        Y = np.asarray(Y_sorted, dtype=float)

        yfake = -np.ones((Y.shape[0], 1), dtype=float)
        response = np.hstack([idxc, X, Y, yfake])

    elif _one_of(fam, fam_agaussian):
        df = y_df.copy()
        df.insert(0, "IDX", ind)
        col_y = _grep_columns(df, r"^Y[0-9]+$")
        if len(col_y) != 5:
            my_stop("For family='agaussian', response must have exactly 5 Y-columns (Y1..Y5).")
        na_y = df[col_y].isna().all(axis=1)
        df = df.loc[~na_y, :]
        response = df[["IDX"] + col_y].to_numpy()

    elif _one_of(fam, fam_fl):
        df = y_df.copy()
        df.insert(0, "IDX", ind)
        # Add fake response in the last column (0)
        df["fake"] = 0
        if df.shape[1] != 11:  # 2 + 9 as in R (IDX + 9 ci's + fake)
            my_stop("For family='fl', expected 9 c_i columns plus IDX and fake.")
        # Remove entries with NA in c_i's (columns 2..10 in 1-based -> here 1..9 after IDX)
        ci_cols = df.columns[1:10]
        df = df.loc[~df[ci_cols].isna().any(axis=1), :]
        # Build response: IDX + all Y (ci) + fake as last?
        y_cols = df.columns[1:]  # keep order
        response = df[["IDX"] + list(y_cols)].to_numpy()

    elif _one_of(fam, fam_rcpoisson):
        df = y_df.copy()
        df.insert(0, "IDX", ind)
        # Expected columns: IDX | Y | E | event | offset | X...
        na_y = df.iloc[:, [1]].isna().any(axis=1)
        df = df.loc[~na_y, :].copy()
        # Rebuild order: idx, E, event, offset, X..., Y
        if df.shape[1] < 6:
            my_stop(f"For family='{fam}', expected at least 5 columns after IDX (Y, E, event, offset, ...).")
        idx = df.iloc[:, [0]].to_numpy()
        Y = df.iloc[:, [1]].to_numpy()
        Ecol = df.iloc[:, [2]].to_numpy()
        eventcol = df.iloc[:, [3]].to_numpy()
        offsetcol = df.iloc[:, [4]].to_numpy()
        X = df.iloc[:, 5:].to_numpy() if df.shape[1] > 5 else np.zeros((df.shape[0], 0))
        response = np.hstack([idx, Ecol, eventcol, offsetcol, X, Y])

    elif _one_of(fam, fam_cenpoisson2):
        e = _ensure_len("E", E if E is not None else 1.0, n_data)
        y = y_df.to_numpy()
        response = _cbind(ind, e, y)
        if response.shape[1] != 5:
            my_stop("For family='cenpoisson2', y_orig must have 3 columns (Y1,Y2,Y3).")
        # drop NA in Y1
        response = response[~np.isnan(response[:, 2]), :]
        # Columns now: IDX, E, Y1, Y2, Y3 -> rename and recode inf
        # Code infinite/high bounds as -1
        Y2 = response[:, 3].copy()
        Y3 = response[:, 4].copy()
        Y3[np.isinf(Y3) | (Y3 < 0)] = -1
        Y2[np.isinf(Y2) | (Y2 < 0)] = -1
        response = np.c_[response[:, 0], response[:, 1], Y2, Y3, response[:, 2]]
        if np.isnan(response).any():
            my_stop(f"family:{fam}. NA's in E/LOW/HIGH not allowed")

    elif _one_of(fam, fam_cennbinomial2):
        s = _ensure_len("scale", scale if scale is not None else 1.0, n_data)
        e = _ensure_len("E", E if E is not None else 1.0, n_data)
        y = y_df.to_numpy()
        response = _cbind(ind, e, s, y)
        if response.shape[1] != 6:
            my_stop("For family='cennbinomial2', y_orig must have 3 columns (Y1,Y2,Y3).")
        response = response[~np.isnan(response[:, 3]), :]
        Y2 = response[:, 4].copy()
        Y3 = response[:, 5].copy()
        Y3[np.isinf(Y3) | (Y3 < 0)] = -1
        Y2[np.isinf(Y2) | (Y2 < 0)] = -1
        response = np.c_[response[:, 0], response[:, 1], response[:, 2], Y2, Y3, response[:, 3]]
        if np.isnan(response).any():
            my_stop(f"family:{fam}. NA's in E/S/LOW/HIGH not allowed")

    elif _one_of(fam, fam_gaussianjw):
        y = y_df.to_numpy()
        response = _cbind(ind, y)
        if response.shape[1] != 5:
            my_stop("For family='gaussianjw', y_orig must have exactly 4 columns (Y1..Y4).")
        response = response[~np.isnan(response[:, 1]), :]
        response = response[~np.isnan(response[:, 2]), :]
        # Rename and reorder: IDX, N=Y3, DF=Y4, VAR=Y2, Y=Y1
        response = np.c_[response[:, 0], response[:, 3], response[:, 4], response[:, 2], response[:, 1]]
        if np.isnan(response).any():
            my_stop(f"family:{fam}. NA's in N/DF/VAR not allowed")

    elif _one_of(fam, fam_ggaussian):
        df = y_df.copy()
        df.insert(0, "IDX", ind)
        df = df.loc[~df.iloc[:, [1]].isna().any(axis=1), :]
        ncovariates = df.shape[1] - 3
        if ncovariates < 0:
            my_stop(f"family={fam}. Number of covariates is {ncovariates}. Did you forget the 's' argument?")
        X = df.iloc[:, 3:] if ncovariates > 0 else pd.DataFrame(index=df.index)
        if not X.empty:
            X = X.fillna(0.0)
        SCALE = df.iloc[:, [2]].to_numpy()
        response = np.hstack([df.iloc[:, [0]].to_numpy(), SCALE, X.to_numpy() if not X.empty else np.zeros((df.shape[0], 0)), df.iloc[:, [1]].to_numpy()])
        if np.isnan(response).any():
            my_stop(f"family:{fam}. NA's in argument 's' are not allowed")

    elif _one_of(fam, fam_ggaussianS):
        df = y_df.copy()
        df.insert(0, "IDX", ind)
        df = df.loc[~df.iloc[:, [1]].isna().any(axis=1), :]
        ncovariates = df.shape[1] - 3
        if ncovariates < 0:
            my_stop(f"family={fam}. Number of covariates is {ncovariates}. Did you forget the 'offset' argument?")
        X = df.iloc[:, 3:] if ncovariates > 0 else pd.DataFrame(index=df.index)
        if not X.empty:
            X = X.fillna(0.0)
        OFFSET = df.iloc[:, [2]].to_numpy()
        response = np.hstack([df.iloc[:, [0]].to_numpy(), OFFSET, X.to_numpy() if not X.empty else np.zeros((df.shape[0], 0)), df.iloc[:, [1]].to_numpy()])
        if np.isnan(response).any():
            my_stop(f"family:{fam}. NA's in argument 'OFFSET' are not allowed")

    elif _one_of(fam, fam_zero_prefix):
        df = y_df.copy()
        df.insert(0, "IDX", ind)
        df = df.loc[~df.iloc[:, [1]].isna().any(axis=1), :]
        ncovariates = df.shape[1] - 3
        if ncovariates < 0:
            my_stop(f"family={fam}. Need at least 3 columns: IDX, Y, EorNtrials.")
        X = df.iloc[:, 3:] if ncovariates > 0 else pd.DataFrame(index=df.index)
        if not X.empty:
            X = X.fillna(0.0)
        EN = df.iloc[:, [2]].to_numpy()
        response = np.hstack([df.iloc[:, [0]].to_numpy(), EN, X.to_numpy() if not X.empty else np.zeros((df.shape[0], 0)), df.iloc[:, [1]].to_numpy()])
        if np.isnan(response).any():
            my_stop(f"family:{fam}. NA's in argument 'E/Ntrials' are not allowed")

    elif _one_of(fam, fam_binomialmix):
        ncy = y_df.shape[1]
        m = ncy - 7
        if not (m >= 0 and m % 2 == 0):
            my_stop("For family='binomialmix', expected #cols = 7 + 2*m (m >= 0 integer).")
        m = m // 2
        # R: 2*m+1 <= length(inla.models()$likelihood$binomialmix$hyper) -- we don't enforce here.
        idx_YY = [0, 1]  # assume first two columns are Y, Ntrials
        idx_WW = [ncy - 2, ncy - 1]
        YY = y_df.iloc[:, idx_YY].copy()
        WW = y_df.iloc[:, idx_WW].copy()
        ZZ = y_df.drop(y_df.columns[idx_YY + idx_WW], axis=1).copy()

        # Filter NA in Y
        mask = ~YY.iloc[:, [0]].isna().any(axis=1)
        YY, WW, ZZ = YY.loc[mask, :], WW.loc[mask, :], ZZ.loc[mask, :]
        ZZ = ZZ.fillna(0.0)

        if (WW.values < 0).any():
            my_stop("binomialmix: W entries must be >= 0.")
        if np.any(np.row_stack(WW.apply(np.sum, axis=1).values) > 1.0):
            my_stop("binomialmix: rowSums(W) must be <= 1.")

        response = np.hstack([
            ind[mask].reshape(-1, 1),
            ZZ.to_numpy(),
            WW.to_numpy(),
            YY.iloc[:, [1]].to_numpy(),  # Ntrials
            YY.iloc[:, [0]].to_numpy(),  # Y
        ])

    elif _one_of(fam, fam_occupancy):
        # This needs y_attr from `inla.mdata`. We expect a metadata vector:
        # y_attr[0] = 2, y_attr[1] = ny, y_attr[2] = m, where m % ny == 0
        if y_attr_vec[0] != 2:
            my_stop("occupancy: missing/invalid `inla.ncols` attribute (expect first element == 2).")
        if y_attr_vec.size < 3:
            my_stop("occupancy: `inla.ncols` must include ny and m.")
        ny = int(y_attr_vec[1])
        m = int(y_attr_vec[2])
        if ny <= 0 or m <= 0 or (m % ny) != 0:
            my_stop("occupancy: invalid (ny, m) in `inla.ncols`.")
        df = y_df.copy()
        df.insert(0, "IDX", ind)
        # drop rows with all NA in first ny response columns
        mask = ~df.iloc[:, 1: 1 + ny].isna().all(axis=1)
        df = df.loc[mask, :]
        # Build response: IDX | Y1..Yny | X1..Xm | fake y
        Y = df.iloc[:, 1: 1 + ny]
        X = df.iloc[:, 1 + ny: 1 + ny + m]
        if X.shape[1] != m:
            my_stop("occupancy: mismatch in X block size vs `m`.")
        yfake = np.zeros((df.shape[0], 1))
        response = np.hstack([df.iloc[:, [0]].to_numpy(), Y.to_numpy(), X.to_numpy(), yfake])

    elif _one_of(fam, fam_cloglike):
        if y_attr_vec[0] != 1:
            my_stop("cloglike: missing/invalid `inla.ncols` attribute (expect first element == 1).")
        if y_attr_vec.size < 2:
            my_stop("cloglike: `inla.ncols` must include ny.")
        ny = int(y_attr_vec[1])
        if ny <= 0:
            my_stop("cloglike: invalid ny in `inla.ncols`.")
        df = y_df.copy()
        df.insert(0, "IDX", ind)
        mask = ~df.iloc[:, 1: 1 + ny].isna().all(axis=1)
        df = df.loc[mask, :]
        response = np.hstack([df.iloc[:, [0]].to_numpy(), df.iloc[:, 1: 1 + ny].to_numpy()])

    elif _one_of(fam, fam_bgev):
        s = _ensure_len("scale", scale if scale is not None else 1.0, n_data)
        if s is None:
            s = np.ones(n_data)
        df = y_df.copy()
        df.insert(0, "IDX", ind)
        Xcols = _grep_columns(df, r"^X[0-9]+$")
        Ycols = _grep_columns(df, r"^Y[0-9]+$")
        if len(Ycols) != 1:
            my_stop("bgev: expected exactly one Y-column (Y1).")
        # Remove rows with NA in Y
        mask = ~df[Ycols].isna().any(axis=1)
        df = df.loc[mask, :]
        s = s[mask]
        X = df[Xcols].fillna(0.0).to_numpy() if Xcols else np.zeros((df.shape[0], 0))
        response = np.hstack([df[["IDX"]].to_numpy(), s.reshape(-1, 1), X, df[Ycols].to_numpy()])

        # y_attr reorder logic; if missing, make a conservative default so downstream code doesn't break
        if y_attr_vec.size == 1 and y_attr_vec[0] == 0:
            # fabricate a minimal 4-length vector as in the R code's final shape
            y_attr_vec = np.array([3, 0, 0, 1], dtype=float)
        else:
            # Place-holder transformation equivalent to the R code's reshuffle
            # R does: y.attr <- c(y.attr[1], y.attr[-c(1, 2)], y.attr[2])
            if y_attr_vec.size >= 2:
                y_attr_vec = np.r_[y_attr_vec[0], y_attr_vec[2:], y_attr_vec[1]]

        if np.isnan(response).any():
            my_stop(f"family:{fam}. NA's in argument 'scale' are not allowed")

    else:
        my_stop(f"Family '{fam}' not recognised in create.data.file.")

    # --- Write files ---
    reuse_names = reuse_names or {}

    def _reuse_or_temp(key: str) -> str:
        name = reuse_names.get(key)
        if name:
            return os.path.join(data_dir, name) if data_dir else name
        return _mktemp(data_dir)

    try:
        file_data = _reuse_or_temp("file.data")
        if debug:
            print(f"[create.data.file] Writing data matrix {response.shape} -> {file_data}")
        inla_write_fmesher_file(response, file_data, debug=debug)
        created_paths.append(file_data)

        file_weights = _reuse_or_temp("file.weights")
        if weights_vec is not None:
            inla_write_fmesher_file(weights_vec, file_weights, debug=debug)
        else:
            _touch(file_weights)
        created_paths.append(file_weights)

        file_lp = _reuse_or_temp("file.lp.scale")
        if lp_vec is not None:
            inla_write_fmesher_file(lp_vec, file_lp, debug=debug)
        else:
            _touch(file_lp)
        created_paths.append(file_lp)

        file_attr = _reuse_or_temp("file.attr")
        # As in R: write y.attr as a column
        inla_write_fmesher_file(np.asarray(y_attr_vec, dtype=float).reshape(-1, 1), file_attr, debug=debug)
        created_paths.append(file_attr)

    except Exception as e:
        my_stop(f"Failed to write files: {e}")

    # Mask data_dir in returned paths
    ret = {
        "file.data": _mask_datadir(file_data, data_dir),
        "file.weights": _mask_datadir(file_weights, data_dir),
        "file.attr": _mask_datadir(file_attr, data_dir),
        "file.lp.scale": _mask_datadir(file_lp, data_dir),
    }
    return ret

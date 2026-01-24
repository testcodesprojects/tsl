# -*- coding: utf-8 -*-
"""
collect.py — Python equivalent of R-INLA's `inla.collect.results` and helpers.
"""

from __future__ import annotations
import os
import re
import struct
import logging
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

# Optional deps
try:
    from scipy.sparse import coo_matrix as _scipy_coo
    _HAS_SCIPY = True
except Exception:
    _scipy_coo = None
    _HAS_SCIPY = False

try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    Image = None
    _HAS_PIL = False

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

# === Use the canonical binary helpers (R-faithful) ===
_HAS_BINARY_HELPERS = True

try:
    from .binary import (  # type: ignore[attr-defined]
        inla_read_binary_file,
        inla_interpret_vector,
        inla_interpret_vector_list,
    )
except ImportError:
    import importlib
    import sys

    _MODULE_DIR = os.path.dirname(__file__)
    _CANDIDATE_PATHS = [_MODULE_DIR]
    for _path in _CANDIDATE_PATHS:
        if _path and os.path.isdir(_path) and _path not in sys.path:
            sys.path.insert(0, _path)

    binary_mod = None
    try:
        binary_mod = importlib.import_module("binary")
    except ModuleNotFoundError:
        binary_mod = None

    if binary_mod is None:
        _HAS_BINARY_HELPERS = False

        def _missing(*_args, **_kwargs):
            raise RuntimeError(
                "pyinla binary helpers are unavailable; reinstall 'pyinla' or provide"
                " a compatible 'binary' module to enable result collection."
            )

        inla_read_binary_file = _missing  # type: ignore[assignment]
        inla_interpret_vector = _missing  # type: ignore[assignment]
        inla_interpret_vector_list = _missing  # type: ignore[assignment]
    else:
        inla_read_binary_file = binary_mod.inla_read_binary_file
        inla_interpret_vector = binary_mod.inla_interpret_vector
        inla_interpret_vector_list = binary_mod.inla_interpret_vector_list


# ======================================================================
#                            Low-level helpers
# ======================================================================

def _read_lines(path: str) -> Optional[List[str]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return [ln.rstrip("\n") for ln in f]


def _read_bytes(f, nbytes: int) -> bytes:
    """Read exactly nbytes or raise EOFError."""
    b = f.read(nbytes)
    if len(b) != nbytes:
        raise EOFError("Unexpected EOF")
    return b


def _read_int32s(f, count: int) -> np.ndarray:
    if count <= 0:
        return np.empty((0,), dtype=np.int32)
    b = _read_bytes(f, 4 * count)
    return np.frombuffer(b, dtype="<i4")


def _read_float64s(f, count: int) -> np.ndarray:
    if count <= 0:
        return np.empty((0,), dtype=np.float64)
    b = _read_bytes(f, 8 * count)
    return np.frombuffer(b, dtype="<f8")


def _read_float64_auto(path: str) -> Optional[np.ndarray]:
    """
    Best-effort reader for INLA/GMRFLib .dat files:
    - If the file looks like [int32 n][n float64] EXACTLY, return the n doubles.
    - Otherwise, treat the whole file as a plain sequence of float64.
    """
    if not os.path.exists(path):
        return None
    size = os.path.getsize(path)
    if size == 0:
        return np.array([], dtype=np.float64)
    with open(path, "rb") as f:
        # Try length-prefixed format: 4 + n*8 bytes
        if size >= 12:  # at least one int + one double
            pos = f.tell()
            try:
                hdr = _read_bytes(f, 4)
                n = struct.unpack("<i", hdr)[0]
            except Exception:
                n = -1
            rest = size - 4
            if 0 <= n <= (rest // 8) and rest == n * 8:
                try:
                    data = _read_float64s(f, n)
                    if data.size == n:
                        return data
                except Exception:
                    pass
            f.seek(pos)
        # Fallback: whole file as doubles
        try:
            b = f.read()
            return np.frombuffer(b, dtype="<f8")
        except Exception:
            return None

def _to_xy_df(a: np.ndarray) -> pd.DataFrame:
    """
    Ensure a 2-column numeric matrix is a DataFrame with columns ['x','y'].
    Accepts 1D even-length vectors by reshaping to pairs.
    Falls back to a plain DataFrame if shape is unexpected.
    """
    arr = np.asarray(a)
    if arr.ndim == 1 and arr.size % 2 == 0:
        arr = arr.reshape(-1, 2)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return pd.DataFrame(arr, columns=["x", "y"])
    return pd.DataFrame(arr)

def _read_numeric_text_or_binary(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    # Try UTF-8 text first (R's scan() behavior)
    try:
        with open(path, "r", encoding="utf-8", errors="strict") as f:
            txt = f.read().strip()
        if txt:
            return np.array([float(tok) for tok in txt.split()], dtype=float)
    except Exception:
        pass
    # Fallback to the standard INLA/GMRFLib binary reader
    return _read_float64_auto(path)


# ----------------------------------------------------------------------
# Quantile/Mode helpers (Gaussian fallback) + robust quantile reader
# ----------------------------------------------------------------------

def _fmt_prob(p: float) -> str:
    return f"{float(p):.6g}"  # e.g. '0.025', '0.5', '0.975'

# 2-sided normal quantiles to match R's default print (approx qnorm)
_Z_FALLBACK = {0.025: -1.959963984540054, 0.5: 0.0, 0.975: 1.959963984540054}

def _ensure_quant_mode_columns(
    mat: np.ndarray,
    colnames: List[str],
    *_,
    **__
) -> Tuple[np.ndarray, List[str]]:
    """
    R parity: DO NOT create missing quantiles/mode.
    Only order columns actually present to: [ID?] mean, sd, <quant by p>, mode, <cdf by p>, kld, <extras>.
    """
    def _prob_from(col: str, suffix: str) -> float:
        try:
            return float(col[:-len(suffix)])
        except Exception:
            return float("nan")

    quant_cols = sorted([c for c in colnames if c.endswith("quant")],
                        key=lambda c: _prob_from(c, "quant"))
    cdf_cols   = sorted([c for c in colnames if c.endswith("cdf")],
                        key=lambda c: _prob_from(c, "cdf"))

    ordered = []
    if "ID" in colnames:
        ordered.append("ID")
    for base in ("mean", "sd"):
        if base in colnames:
            ordered.append(base)
    ordered += quant_cols
    if "mode" in colnames:
        ordered.append("mode")
    ordered += cdf_cols
    if "kld" in colnames:
        ordered.append("kld")

    extras = [c for c in colnames if c not in ordered]
    final_cols = ordered + extras
    idx = [colnames.index(c) for c in final_cols]
    return (mat[:, idx] if mat.size else mat), final_cols

def _read_quantcdf_block(path: str, n_rows: int) -> Optional[Tuple[np.ndarray, List[str]]]:
    """
    Robust reader for quantile/cdf blocks that INLA writes as a flattened vector
    decoded by inla.interpret.vector. Returns (values_matrix, prob_labels).
      - values_matrix: shape (n_rows, n_probs), the Y-part per row
      - prob_labels:   ['0.025', '0.5', '0.975', ...]
    """
    raw = inla_read_binary_file(path)
    if raw is None or raw.size == 0:
        return None
    mat = inla_interpret_vector(raw)
    if mat is None or mat.size == 0:
        return None
    # mat shape: (n_probs, 1 + 2*n_rows)  -> first col is probs,
    # remaining columns are [x1,y1,x2,y2,...,xn,yn]
    total_cols = mat.shape[1]
    if total_cols == 1 + 2 * n_rows:
        probs = mat[:, 0]
        vals = mat[:, 1 + np.arange(0, 2 * n_rows, 2)]
    elif total_cols == 2 * n_rows:
        probs = mat[:, 0]
        vals = mat[:, 1::2]
    else:
        vec = np.asarray(raw, dtype=float)
        expect = 1 + 2 * n_rows
        if vec.size % expect != 0:
            return None
        n_prob = vec.size // expect
        alt = vec.reshape(n_prob, expect)
        probs = alt[:, 0]
        vals = alt[:, 1 + np.arange(0, 2 * n_rows, 2)]
    return vals.T, [f"{float(p):.6g}" for p in probs]


def _read_mode_block(path: str, n_rows: int) -> Optional[np.ndarray]:
    """
    Read mode block (same inla.interpret.vector layout), return 1D array of length n_rows
    with the 'y' from the first column-pair.
    """
    raw = inla_read_binary_file(path)
    if raw is None or raw.size == 0:
        return None
    mat = inla_interpret_vector(raw)
    if mat is None or mat.size == 0:
        return None
    # columns: [prob, x1,y1,x2,y2,...]
    total_cols = mat.shape[1]
    if total_cols == 1 + 2 * n_rows:
        vals = mat[:, 1 + np.arange(0, 2 * n_rows, 2)]
    elif total_cols == 2 * n_rows:
        vals = mat[:, 1::2]
    else:
        vec = np.asarray(raw, dtype=float)
        expect = 1 + 2 * n_rows
        if vec.size % expect != 0:
            return None
        n_rows_in_file = vec.size // expect
        alt = vec.reshape(n_rows_in_file, expect)
        vals = alt[:, 1 + np.arange(0, 2 * n_rows, 2)]
    return vals[0, :]


def _interpret_vector_pairs(vec: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Turn flat [a1,b1,a2,b2,...] into (N×2) matrix.
    """
    if vec is None or vec.size == 0:
        return None
    if vec.size % 2 != 0:
        vec = vec[:-1]
    return vec.reshape(-1, 2)


def _interpret_vector_list(vec: Optional[np.ndarray]) -> Optional[List[np.ndarray]]:
    """
    INLA "vector-of-vectors" for marginals:
    [n1, x1_1, y1_1, ..., x1_n1, y1_n1,  n2, x2_1, y2_1, ...].
    Returns a list of (ni×2) arrays.
    """
    if vec is None or vec.size == 0:
        return None
    out: List[np.ndarray] = []
    i = 0
    N = vec.size
    while i < N:
        n = int(vec[i]); i += 1
        if n <= 0:
            out.append(np.zeros((0, 2)))
            continue
        need = 2 * n
        blk = vec[i:i + need]
        if blk.size < need:
            break
        out.append(blk.reshape(n, 2))
        i += need
    return out if out else None


def _reshape_matrix_rowwise(vec: np.ndarray, ncol: int) -> np.ndarray:
    """
    R: matrix(vec, ncol = ncol, byrow = TRUE)
    -> numpy reshape(-1, ncol) with row-major order ('C').
    """
    if vec.size % ncol != 0:
        return np.empty((0, ncol))
    return vec.reshape(-1, ncol)


def _read_pbm_pixels(path: str) -> Optional[np.ndarray]:
    if not _HAS_PIL or not os.path.exists(path):
        return None
    try:
        im = Image.open(path)
        return np.array(im)
    except Exception:
        return None


def _scipy_or_dict(shape: Tuple[int, int], i: np.ndarray, j: np.ndarray, x: np.ndarray):
    if _HAS_SCIPY:
        return _scipy_coo((x, (i, j)), shape=shape)
    return {"rows": i.astype(np.int32), "cols": j.astype(np.int32), "data": x.astype(float), "shape": shape}


def _inla_num(x: int | np.ndarray) -> str | List[str]:
    """Pad with leading zeros (width 3) like R's inla.num."""
    def _fmt(v: int) -> str:
        return f"{int(v):03d}"
    if isinstance(x, np.ndarray):
        return [_fmt(v) for v in x]
    return _fmt(x)


# ======================================================================
#                Discovery of the actual results directory
# ======================================================================

def _resolve_latest_model_dir(entry: str, prefix: str = "inla.model") -> Optional[str]:
    """
    If `entry` is a parent dir, pick the numerically latest `inla.model-*`.
    If `entry` itself is a run dir, return it unchanged.
    """
    if not os.path.isdir(entry):
        return None
    subs = [d for d in os.listdir(entry) if os.path.isdir(os.path.join(entry, d))]
    cand = []
    pat = re.compile(rf"^{re.escape(prefix)}(?:-(\d+))?$")
    for d in subs:
        m = pat.match(d)
        if m:
            num = int(m.group(1) or 0)
            cand.append((num, os.path.join(entry, d)))
    if not cand:  # assume entry is already a run dir
        return entry
    cand.sort(key=lambda x: x[0])
    return cand[-1][1]


def _resolve_results_dir_like_R(base: str) -> Optional[str]:
    """
    Exact R logic, plus scanning results.files-0000000000, -0000000001, ... until `.ok` is found.
    """
    if not os.path.isdir(base):
        return None
    if os.path.exists(os.path.join(base, ".ok")):
        return base

    orig = os.path.join(base, "results.files")
    if os.path.isdir(orig) and os.path.exists(os.path.join(orig, ".ok")):
        return orig

    for count in range(0, 1_000_001):
        cand = f"{orig}-{count:010d}"
        if os.path.isdir(cand) and os.path.exists(os.path.join(cand, ".ok")):
            return cand
    return None


# ======================================================================
#                             Block collectors
# ======================================================================

def _collect_size(dirpath: str) -> Dict[str, int]:
    vv = _read_float64_auto(os.path.join(dirpath, "size.dat"))
    if vv is None or vv.size != 5:
        return {"n": 0, "N": 0, "Ntotal": 0, "ngroup": 0, "nrep": 0}
    n, N, Ntotal, ngroup, nrep = vv
    if np.isnan(n) or n < 0:
        raise RuntimeError("Invalid size.dat")
    if np.isnan(N) or N <= 0:
        N = n
    if np.isnan(Ntotal) or Ntotal <= 0:
        Ntotal = N
    if np.isnan(ngroup) or ngroup <= 0:
        ngroup = 1
    if np.isnan(nrep) or nrep <= 0:
        nrep = 1
    return {"n": int(n), "N": int(N), "Ntotal": int(Ntotal), "ngroup": int(ngroup), "nrep": int(nrep)}

def _collect_fixed(results_dir: str) -> Dict[str, Any]:
    alldir = sorted(os.listdir(results_dir))
    fix_dirs = [d for d in alldir if d.startswith("fixed.effect")]
    if "intercept" in alldir:
        fix_dirs.append("intercept")

    if not fix_dirs:
        return {"names.fixed": None, "summary.fixed": None, "marginals.fixed": None}

    names_fixed: List[str] = []
    row_dicts: Dict[str, Dict[str, float]] = {}
    margs: Dict[str, np.ndarray] = {}

    for d in fix_dirs:
        base = os.path.join(results_dir, d)
        nm = (_read_lines(os.path.join(base, "TAG")) or ["NameMissing"])[0]
        names_fixed.append(nm)

        row: Dict[str, float] = {}

        # summary.dat -> drop header: [*, mean, sd, ...]
        s_path = os.path.join(base, "summary.dat")
        if os.path.exists(s_path):
            summ = _read_float64_auto(s_path)
            if summ is not None and summ.size >= 3:
                summ = summ[1:]
                row["mean"] = float(summ[0])
                row["sd"] = float(summ[1])
            else:
                row["mean"] = np.nan
                row["sd"] = np.nan
        else:
            row["mean"] = np.nan
            row["sd"] = np.nan

        # quantiles
        q_path = os.path.join(base, "quantiles.dat")
        q_mat = inla_interpret_vector(inla_read_binary_file(q_path))
        if q_mat is not None and q_mat.size:
            for prob, val in q_mat:
                row[f"{_fmt_prob(prob)}quant"] = float(val)
        else:
            # Gaussian fallback
            mu, sd = row.get("mean", np.nan), row.get("sd", np.nan)
            if np.isfinite(mu) and np.isfinite(sd):
                row["0.025quant"] = float(mu + _Z_FALLBACK[0.025] * sd)
                row["0.5quant"] = float(mu)
                row["0.975quant"] = float(mu + _Z_FALLBACK[0.975] * sd)

        # mode
        m_path = os.path.join(base, "mode.dat")
        m_mat = inla_interpret_vector(inla_read_binary_file(m_path))
        if m_mat is not None and m_mat.size:
            row["mode"] = float(m_mat[0, 1])
        else:
            mu = row.get("mean", np.nan)
            if np.isfinite(mu):
                row["mode"] = float(mu)

        # cdf (optional)
        c_path = os.path.join(base, "cdf.dat")
        c_mat = inla_interpret_vector(inla_read_binary_file(c_path))
        if c_mat is not None and c_mat.size:
            for prob, val in c_mat:
                row[f"{_fmt_prob(prob)}cdf"] = float(val)

        # KLD
        kld = _read_float64_auto(os.path.join(base, "symmetric-kld.dat"))
        if kld is not None and kld.size >= 2:
            row["kld"] = float(kld.reshape(-1, 2)[:, 1][0])
        else:
            row["kld"] = np.nan

        row_dicts[nm] = row

        xx = inla_read_binary_file(os.path.join(base, "marginal-densities.dat"))
        mat = inla_interpret_vector(xx)  # (N×2)
        if mat is None or mat.size == 0:
            mat = np.array([[np.nan, np.nan],
                            [np.nan, np.nan],
                            [np.nan, np.nan]], dtype=float)
        margs[nm] = _to_xy_df(mat)


    # DataFrame
    df = pd.DataFrame.from_dict(row_dicts, orient="index")
    # Order like R
    cols = list(df.columns)
    quant_cols = sorted([c for c in cols if c.endswith("quant")], key=lambda c: float(c[:-5]))
    cdf_cols = sorted([c for c in cols if c.endswith("cdf")], key=lambda c: float(c[:-3]))
    ordered = []
    for base in ["mean", "sd"]:
        if base in df.columns:
            ordered.append(base)
    ordered += quant_cols
    if "mode" in df.columns:
        ordered.append("mode")
    ordered += cdf_cols
    if "kld" in df.columns:
        ordered.append("kld")
    extras = [c for c in cols if c not in ordered]
    df = df.reindex(columns=ordered + extras)
    df = df.reindex(index=names_fixed)

    return {
        "names.fixed": names_fixed,
        "summary.fixed": df,
        "marginals.fixed": margs if margs else None,
    }


def _read_kld_column(path: str, n_rows: int) -> Optional[np.ndarray]:
    vec = _read_float64_auto(path)
    if vec is None or vec.size == 0:
        return None
    if vec.size % 2 != 0:
        return None
    mat = vec.reshape(-1, 2)
    col = mat[:, 1]
    if col.size == n_rows:
        return col
    if col.size == 1:
        return np.full(n_rows, col[0], dtype=float)
    return None

def _collect_lincomb(results_dir: str, derived: bool) -> Dict[str, Any]:
    """
    Collect linear-combination summaries/marginals.

    - summary: pandas.DataFrame with columns like
        [ID, mean, sd, 0.025quant, 0.5quant, 0.975quant, mode, <cdf...>, kld, ...]
      (If none found, returns an empty DataFrame.)
    - marginals: dict[name] -> DataFrame with ['x','y'] (or None if absent)
    - size: dict with n/N/Ntotal/ngroup/nrep

    Requires helpers already in this module:
      _reshape_matrix_rowwise, _read_quantcdf_block, _read_mode_block,
      _read_kld_column, _collect_size, _to_xy_df, _ensure_quant_mode_columns.
    """
    alldir = sorted(os.listdir(results_dir))
    if derived:
        lin = [d for d in alldir if d.startswith("lincomb") and "derived.all" in d]
    else:
        l_derived = [d for d in alldir if d.startswith("lincomb") and "derived.all" in d]
        l_all     = [d for d in alldir if d.startswith("lincomb")]
        lin = [d for d in l_all if d not in l_derived]

    keyS = "summary.lincomb.derived" if derived else "summary.lincomb"
    keyM = "marginals.lincomb.derived" if derived else "marginals.lincomb"
    keyZ = "size.lincomb.derived" if derived else "size.lincomb"

    # No lincomb directory: return an empty summary table with canonical headers
    # so downstream CSV export has a header row (avoids empty-file CSV issues)
    #if not lin:
    #    empty_cols = [
    #        "ID", "mean", "sd", "0.025quant", "0.5quant", "0.975quant", "mode", "kld"
    #    ]
    #    return {keyS: pd.DataFrame(columns=empty_cols), keyM: None, keyZ: None}

    if not lin:
        return {keyS: None, keyM: None, keyZ: None}

    base = os.path.join(results_dir, lin[0])  # match R behavior: take the first
    # ---- Summary ----
    dd = inla_read_binary_file(os.path.join(base, "summary.dat"))
    if dd is None or dd.size == 0:
        # R fallback: use text file "N" (not N.dat) to size an NA table with mean/sd/kld only
        n_file = os.path.join(base, "N")
        try:
            N = int(float((_read_lines(n_file) or ["0"])[0]))
        except Exception:
            N = 0
        Summ = pd.DataFrame({
            "mean": [np.nan] * N,
            "sd":   [np.nan] * N,
            "kld":  [np.nan] * N,
        })
        rn = None
        n_rows = N
    else:

        # ID, mean, sd (row-major by R convention)
        Mat = _reshape_matrix_rowwise(dd, 3)
        Svals = Mat
        colnames = ["ID", "mean", "sd"]
        n_rows = Svals.shape[0]

        # quantiles
        qc = _read_quantcdf_block(os.path.join(base, "quantiles.dat"), n_rows)
        if qc is not None:
            qmat, qnames = qc
            Svals = np.column_stack([Svals, qmat])
            colnames.extend([f"{q}quant" for q in qnames])

        # mode
        mv = _read_mode_block(os.path.join(base, "mode.dat"), n_rows)
        if mv is not None:
            Svals = np.column_stack([Svals, mv])
            colnames.append("mode")

        # cdf
        cc = _read_quantcdf_block(os.path.join(base, "cdf.dat"), n_rows)
        if cc is not None:
            cmat, cnames = cc
            Svals = np.column_stack([Svals, cmat])
            colnames.extend([f"{c}cdf" for c in cnames])

        # kld
        kv = _read_kld_column(os.path.join(base, "symmetric-kld.dat"), n_rows)
        if kv is not None:
            Svals = np.column_stack([Svals, kv])
            colnames.append("kld")

        # Names from NAMES file (strip "lincomb." prefix)
        rn = _read_lines(os.path.join(base, "NAMES"))
        if rn:
            rn = [re.sub(r"^lincomb[.]", "", r) for r in rn]
            rn = rn[:n_rows]

        # Ensure standard columns and order (adds missing quant/mode if needed)
        Svals, colnames = _ensure_quant_mode_columns(Svals, colnames)

        # Always build a DataFrame (no dict fallback)
        Summ = pd.DataFrame(Svals, columns=colnames)
        if rn is not None and len(rn) == Summ.shape[0]:
            Summ.index = rn

    # ---- Marginals ----
    xx = inla_read_binary_file(os.path.join(base, "marginal-densities.dat"))
    rr_list = inla_interpret_vector_list(xx)
    if rr_list is not None:
        if rn and len(rn) >= len(rr_list):
            names_rr = rn[:len(rr_list)]
        else:
            names_rr = [f"index.{i}" for i in range(1, len(rr_list) + 1)]
        rr_named = {names_rr[k]: _to_xy_df(rr_list[k]) for k in range(len(rr_list))}
    else:
        rr_named = None

    size = _collect_size(base)

    return {keyS: Summ, keyM: rr_named, keyZ: size}

def _collect_random_like(results_dir: str, prefix: str) -> Dict[str, Any]:
    alldir = sorted(os.listdir(results_dir))
    rdirs = [d for d in alldir if d.startswith(prefix)]
    n_random = len(rdirs)
    key_model = f"model.{prefix}"
    key_summary = f"summary.{prefix}"
    key_marginals = f"marginals.{prefix}"
    key_size = f"size.{prefix}"

    if n_random == 0:
        return {key_model: None, key_summary: None, key_marginals: None, key_size: None}

    names_random: List[str] = []
    model_random: List[str] = []
    summary_random: Dict[str, Any] = {}
    marginals_random: Dict[str, Any] = {}
    size_random: Dict[str, Any] = {}

    for d in rdirs:
        base = os.path.join(results_dir, d)
        nm = (_read_lines(os.path.join(base, "TAG")) or ["missing NAME"])[0]
        mdl = (_read_lines(os.path.join(base, "MODEL")) or ["NoModelName"])[0]
        names_random.append(nm)
        model_random.append(mdl)

        rownames_for_marginals: Optional[List[str]] = None  # we will reuse this for naming marginals

        if os.path.exists(os.path.join(base, "summary.dat")):
            dd = inla_read_binary_file(os.path.join(base, "summary.dat"))
            if dd is None or dd.size == 0:
                S = None
                n_rows = 0
            else:
                Mat = _reshape_matrix_rowwise(dd, 3)  # ID, mean, sd
                Svals = Mat
                colnames = ["ID", "mean", "sd"]
                n_rows = Svals.shape[0]

                # quantiles
                qc = _read_quantcdf_block(os.path.join(base, "quantiles.dat"), n_rows)
                if qc is not None:
                    qmat, qnames = qc
                    Svals = np.column_stack([Svals, qmat])
                    colnames.extend([f"{q}quant" for q in qnames])

                # mode
                mv = _read_mode_block(os.path.join(base, "mode.dat"), n_rows)
                if mv is not None:
                    Svals = np.column_stack([Svals, mv])
                    colnames.append("mode")

                # cdf
                cc = _read_quantcdf_block(os.path.join(base, "cdf.dat"), n_rows)
                if cc is not None:
                    cmat, cnames = cc
                    Svals = np.column_stack([Svals, cmat])
                    colnames.extend([f"{c}cdf" for c in cnames])

                # kld
                kv = _read_kld_column(os.path.join(base, "symmetric-kld.dat"), n_rows)
                if kv is not None:
                    Svals = np.column_stack([Svals, kv])
                    colnames.append("kld")

                # Ensure quantiles/mode exist; order like R
                Svals, colnames = _ensure_quant_mode_columns(Svals, colnames)

                # R keeps default 1..N rownames; ID column may be overridden by id-names
                idnames = _read_lines(os.path.join(base, "id-names.dat"))
                try:
                    S = pd.DataFrame(Svals, columns=colnames)
                except Exception:
                    S = {"values": Svals, "_rownames": None, "_colnames": colnames}
                if isinstance(S, pd.DataFrame) and idnames and len(idnames) >= n_rows:
                    S.loc[:n_rows - 1, "ID"] = idnames[:n_rows]
                rownames_for_marginals = (idnames[:n_rows] if idnames and len(idnames) >= n_rows else None)

            # marginals (list-of-blocks) → named dict
            xx = inla_read_binary_file(os.path.join(base, "marginal-densities.dat"))
            rr_list = inla_interpret_vector_list(xx)
            if rr_list is not None:
                if rownames_for_marginals is not None and len(rownames_for_marginals) >= len(rr_list):
                    names_rr = rownames_for_marginals[:len(rr_list)]
                else:
                    names_rr = [f"index.{i}" for i in range(1, len(rr_list) + 1)]
                rr_named = {names_rr[k]: _to_xy_df(rr_list[k]) for k in range(len(rr_list))}
            else:
                rr_named = None
            marginals_random[nm] = rr_named

        else:
            # empty -> NA table with KLD
            Nfile = os.path.join(base, "N")
            try:
                N = int(float((_read_lines(Nfile) or ["0"])[0]))
            except Exception:
                N = 0
            Svals = np.column_stack([np.full(N, np.nan), np.full(N, np.nan), np.full(N, np.nan)])  # mean, sd, kld
            try:
                S = pd.DataFrame(Svals, index=[f"index.{i}" for i in range(1, N + 1)],
                                 columns=["mean", "sd", "kld"])
            except Exception:
                S = {"values": Svals, "_rownames": [f"index.{i}" for i in range(1, N + 1)],
                     "_colnames": ["mean", "sd", "kld"]}
            marginals_random[nm] = None

        summary_random[nm] = S
        size_random[nm] = _collect_size(base)

    # Could be a list of lists of None/NA
    if not any(v is not None for v in marginals_random.values()):
        marginals_random = None

    return {
        key_model:     model_random,
        key_summary:   summary_random,
        key_marginals: marginals_random,
        key_size:      size_random,
    }


def _collect_random(results_dir: str) -> Dict[str, Any]:
    res = _collect_random_like(results_dir, "random.effect")
    return {
        "model.random":      res.get("model.random.effect"),
        "summary.random":    res.get("summary.random.effect"),
        "marginals.random":  res.get("marginals.random.effect"),
        "size.random":       res.get("size.random.effect"),
    }


def _collect_spde2_blc(results_dir: str) -> Dict[str, Any]:
    res = _collect_random_like(results_dir, "spde2.blc")
    return {
        "model.spde2.blc": res.get("model.spde2.blc"),
        "summary.spde2.blc": res.get("summary.spde2.blc"),
        "marginals.spde2.blc": res.get("marginals.spde2.blc"),
        "size.spde2.blc": res.get("size.spde2.blc"),
    }


def _collect_spde3_blc(results_dir: str) -> Dict[str, Any]:
    res = _collect_random_like(results_dir, "spde3.blc")
    return {
        "model.spde3.blc": res.get("model.spde3.blc"),
        "summary.spde3.blc": res.get("summary.spde3.blc"),
        "marginals.spde3.blc": res.get("marginals.spde3.blc"),
        "size.spde3.blc": res.get("size.spde3.blc"),
    }


def _collect_dic(results_dir: str) -> Optional[Dict[str, Any]]:
    d = os.path.join(results_dir, "dic")
    if not os.path.isdir(d):
        return None
    vals = _read_float64_auto(os.path.join(d, "dic.dat"))
    if vals is None or vals.size < 4:
        return None

    def v1(x):
        arr = _read_fmesher(os.path.join(d, x))
        return None if arr is None else np.asarray(arr).ravel()

    dev_e = v1("deviance_e.dat")
    dev_e_sat = v1("deviance_e_sat.dat")
    e_dev = v1("e_deviance.dat")
    e_dev_sat = v1("e_deviance_sat.dat")
    sig = v1("sign.dat")

    out: Dict[str, Any] = {
        "dic": float(vals[3]),
        "p.eff": float(vals[2]),
        "mean.deviance": float(vals[0]),
        "deviance.mean": float(vals[1]),
    }

     # R parity: include '.sat' metrics if available
    if vals.size >= 8:
        out["mean.deviance.sat"] = float(vals[4])
        out["deviance.mean.sat"] = float(vals[5])
        out["dic.sat"] = float(vals[7])

    fidx = _read_fmesher(os.path.join(d, "family_idx.dat"))
    if isinstance(fidx, np.ndarray):
        fidx = (fidx.astype(float) + 1.0).ravel()  # R indexing as vector
        out["family"] = fidx

    if e_dev is not None and dev_e is not None:
        out["local.dic"] = 2.0 * e_dev - dev_e
        out["local.p.eff"] = e_dev - dev_e
    if e_dev_sat is not None and dev_e_sat is not None:
        out["local.dic.sat"] = 2.0 * e_dev_sat - dev_e_sat

    if sig is not None and e_dev_sat is not None:
        ee = e_dev_sat.copy()
        ee[np.isnan(ee)] = 0
        ee = np.maximum(0, ee)
        sg = sig.copy()
        sg[np.isnan(sg)] = 0
        devres = np.sqrt(ee) * sg
        devres[np.isnan(e_dev_sat)] = np.nan
        out["residuals.deviance"] = devres

    if "family" in out and out["family"] is not None and not np.all(np.isnan(out["family"])):
        k = int(np.nanmax(out["family"]))
        fam_dic = np.zeros(k)
        fam_dic_sat = np.zeros(k)
        fam_peff = np.zeros(k)
        for j in range(1, k + 1):
            idx = np.where(out["family"] == j)[0]
            if "local.dic" in out:
                fam_dic[j - 1] = np.nansum(out["local.dic"][idx])
            if "local.dic.sat" in out:
                fam_dic_sat[j - 1] = np.nansum(out["local.dic.sat"][idx])
            if "local.p.eff" in out:
                fam_peff[j - 1] = np.nansum(out["local.p.eff"][idx])
        out["family.dic"] = fam_dic
        out["family.dic.sat"] = fam_dic_sat
        out["family.p.eff"] = fam_peff
    return out


def _collect_cpo(results_dir: str) -> Dict[str, Any]:
    d = os.path.join(results_dir, "cpo")
    if not os.path.isdir(d):
        return {"cpo": None, "pit": None, "failure": None}

    def _scatter(p):
        v = _read_float64_auto(p)
        if v is None or v.size == 0:
            return None
        n = int(v[0])
        data = v[1:]
        arr = np.full(n, np.nan)
        idx = data[0::2].astype(int)
        val = data[1::2]
        arr[idx] = val
        return arr

    return {"cpo": _scatter(os.path.join(d, "cpo.dat")),
            "pit": _scatter(os.path.join(d, "pit.dat")),
            "failure": _scatter(os.path.join(d, "failure.dat"))}


def _collect_gcpo(results_dir: str) -> Dict[str, Any]:
    d = os.path.join(results_dir, "gcpo")
    if not os.path.isdir(d):
        return {"gcpo": None, "kld": None, "mean": None, "sd": None, "groups": None}
    v = _read_float64_auto(os.path.join(d, "gcpo.dat"))
    if v is None or v.size == 0:
        return {"gcpo": None, "kld": None, "mean": None, "sd": None, "groups": None}
    n = int(v[0])
    rest = v[1:]
    values = rest[:n]
    kld = rest[n:2 * n]
    mean = rest[2 * n:3 * n]
    sd = rest[3 * n:4 * n]
    off = 4 * n
    groups = []
    while off < rest.size:
        nn = int(rest[off]); off += 1
        if nn > 0:
            idx = rest[off:off + nn]; off += nn
            cor = rest[off:off + nn]; off += nn
            groups.append({"idx": idx, "corr": cor})
        else:
            groups.append({"idx": None, "corr": None})
    return {"gcpo": values, "kld": kld, "mean": mean, "sd": sd, "groups": groups}


def _collect_po(results_dir: str) -> Dict[str, Any]:
    d = os.path.join(results_dir, "po")
    if not os.path.isdir(d):
        return {"po": None}
    v = _read_float64_auto(os.path.join(d, "po.dat"))
    if v is None or v.size == 0:
        return {"po": None}
    n = int(v[0])
    data = v[1:]
    idx = data[0::3].astype(int)
    po = data[1::3]
    out_vec = np.full(n, np.nan)
    out_vec[idx] = po
    # Parity with R: 'po' is a list with one element named 'po'
    return {"po": {"po": out_vec}}



def _collect_waic_from_po(results_dir: str) -> Optional[Dict[str, Any]]:
    d = os.path.join(results_dir, "po")
    if not os.path.isdir(d):
        return None
    v = _read_float64_auto(os.path.join(d, "po.dat"))
    if v is None or v.size == 0:
        return None
    n = int(v[0])
    data = v[1:]
    idx = data[0::3].astype(int)
    po = data[1::3]
    po2 = data[2::3]
    po_res = np.full(n, np.nan); po_res[idx] = po
    po2_res = np.full(n, np.nan); po2_res[idx] = po2
    with np.errstate(divide="ignore", invalid="ignore"):
        logpo = np.where(po_res > 0, np.log(po_res), np.nan)
    return {
        "waic": float(-2.0 * (np.nansum(logpo) - np.nansum(po2_res))),
        "p.eff": float(np.nansum(po2_res)),
        "local.waic": -2.0 * (logpo - po2_res),
        "local.p.eff": po2_res
    }



def _collect_mlik(results_dir: str) -> Dict[str, Any]:
    d = os.path.join(results_dir, "marginal-likelihood")
    if not os.path.isdir(d):
        return {"mlik": None}
    v = _read_float64_auto(os.path.join(d, "marginal-likelihood.dat"))
    if v is None or v.size < 2:
        return {"mlik": None}
    s = pd.Series(
        [float(v[0]), float(v[1])],
        index=[
            "log marginal-likelihood (integration)",
            "log marginal-likelihood (Gaussian)",
        ],
        dtype=float,
    )
    return {"mlik": s}





def _collect_q(results_dir: str) -> Optional[Dict[str, Any]]:
    qd = os.path.join(results_dir, "Q")
    if not os.path.isdir(qd):
        return None
    Q = _read_pbm_pixels(os.path.join(qd, "precision-matrix.pbm"))
    Qr = _read_pbm_pixels(os.path.join(qd, "precision-matrix-reordered.pbm"))
    L = _read_pbm_pixels(os.path.join(qd, "precision-matrix_L.pbm"))
    if Q is None and Qr is None and L is None:
        return None
    return {"Q": Q, "Q.reorder": Qr, "L": L}


def _collect_graph(results_dir: str) -> Dict[str, Any]:
    path = os.path.join(results_dir, "graph.dat")
    if not os.path.exists(path):
        return {"graph": None}
    try:
        with open(path, "rb") as f:
            sig = f.read(16)
        if all(32 <= b <= 126 or b in (9, 10, 13) for b in sig if b != 0):
            return {"graph": _read_lines(path)}  # text
        else:
            with open(path, "rb") as f:
                return {"graph": f.read()}  # binary
    except Exception:
        return {"graph": None}
def _collect_predictor(results_dir: str) -> Dict[str, Any]:
    alldir = sorted(os.listdir(results_dir))
    out: Dict[str, Any] = {
        "summary.linear.predictor": None,
        "marginals.linear.predictor": None,
        "summary.fitted.values": None,
        "marginals.fitted.values": None,
        "size.linear.predictor": None,
    }

    # ---------- linear predictor (latent scale) ----------
    sub = os.path.join(results_dir, "predictor")
    if os.path.isdir(sub) and os.path.exists(os.path.join(sub, "summary.dat")):
        dd = inla_read_binary_file(os.path.join(sub, "summary.dat"))
        if dd is not None and dd.size:
            Mat = _reshape_matrix_rowwise(dd, 3)[:, 1:]  # mean, sd
            colnames = ["mean", "sd"]
        else:
            Mat = np.empty((0, 2)); colnames = ["mean", "sd"]

        size_info = _collect_size(sub)
        out["size.linear.predictor"] = size_info
        n_rows = Mat.shape[0]

        # quantiles
        qc = _read_quantcdf_block(os.path.join(sub, "quantiles.dat"), n_rows)
        if qc is not None:
            qmat, qnames = qc
            Mat = np.column_stack([Mat, qmat]); colnames.extend([f"{q}quant" for q in qnames])

        # mode
        mv = _read_mode_block(os.path.join(sub, "mode.dat"), n_rows)
        if mv is not None:
            Mat = np.column_stack([Mat, mv]); colnames.append("mode")

        # cdf
        cc = _read_quantcdf_block(os.path.join(sub, "cdf.dat"), n_rows)
        if cc is not None:
            cmat, cnames = cc
            Mat = np.column_stack([Mat, cmat]); colnames.extend([f"{c}cdf" for c in cnames])

        # kld
        kv = _read_kld_column(os.path.join(sub, "symmetric-kld.dat"), n_rows)
        if kv is not None:
            Mat = np.column_stack([Mat, kv]); colnames.append("kld")

        # Ensure standard quantiles/mode and order like R
        Mat, colnames = _ensure_quant_mode_columns(Mat, colnames)

        # rownames like R
        rn: Optional[List[str]]
        if size_info:
            A = (size_info.get("nrep", 1) == 2)
            n = size_info.get("n", 0)
            nA = size_info.get("Ntotal", 0) - n
            if A:
                rn = [f"APredictor.{_inla_num(i)}" for i in range(1, nA + 1)] + \
                     [f"Predictor.{_inla_num(i)}" for i in range(1, n + 1)]
            else:
                rn = [f"Predictor.{_inla_num(i)}" for i in range(1, size_info.get("Ntotal", 0) + 1)]
        else:
            rn = None

        try:
            out["summary.linear.predictor"] = pd.DataFrame(Mat, columns=colnames, index=rn)
        except Exception:
            out["summary.linear.predictor"] = {"values": Mat, "_rownames": rn, "_colnames": colnames}

        # marginals: list-of-blocks → named dict (match R naming)
        rr_list = inla_interpret_vector_list(inla_read_binary_file(os.path.join(sub, "marginal-densities.dat")))
        if rr_list is not None:
            if size_info and (size_info.get("nrep", 1) == 2):
                n = size_info.get("n", 0)
                nA = size_info.get("Ntotal", 0) - n
                names_rr = [f"APredictor.{_inla_num(i)}" for i in range(1, nA + 1)] + \
                           [f"Predictor.{_inla_num(i)}" for i in range(1, n + 1)]
            else:
                # Follow R: use n (not Ntotal) when A==False
                n = size_info.get("n", len(rr_list)) if size_info else len(rr_list)
                names_rr = [f"Predictor.{_inla_num(i)}" for i in range(1, n + 1)]
                if len(names_rr) != len(rr_list):
                    names_rr = [f"Predictor.{_inla_num(i)}" for i in range(1, len(rr_list) + 1)]
            out["marginals.linear.predictor"] = {names_rr[k]: _to_xy_df(rr_list[k]) for k in range(len(rr_list))}
        else:
            out["marginals.linear.predictor"] = None

    # ---------- fitted values (user scale) ----------
    subu = os.path.join(results_dir, "predictor-user-scale")
    if os.path.isdir(subu) and os.path.exists(os.path.join(subu, "summary.dat")):
        dd = inla_read_binary_file(os.path.join(subu, "summary.dat"))
        if dd is not None and dd.size:
            Mat = _reshape_matrix_rowwise(dd, 3)[:, 1:]  # mean, sd
            colnames = ["mean", "sd"]
        else:
            Mat = np.empty((0, 2)); colnames = ["mean", "sd"]

        n_rows = Mat.shape[0]

        qc = _read_quantcdf_block(os.path.join(subu, "quantiles.dat"), n_rows)
        if qc is not None:
            qmat, qnames = qc
            Mat = np.column_stack([Mat, qmat]); colnames.extend([f"{q}quant" for q in qnames])

        mv = _read_mode_block(os.path.join(subu, "mode.dat"), n_rows)
        if mv is not None:
            Mat = np.column_stack([Mat, mv]); colnames.append("mode")

        cc = _read_quantcdf_block(os.path.join(subu, "cdf.dat"), n_rows)
        if cc is not None:
            cmat, cnames = cc
            Mat = np.column_stack([Mat, cmat]); colnames.extend([f"{c}cdf" for c in cnames])

        # Ensure standard quantiles/mode and order like R
        Mat, colnames = _ensure_quant_mode_columns(Mat, colnames)

        size_info = out.get("size.linear.predictor") or {}
        A = (size_info.get("nrep", 1) == 2)
        n = size_info.get("n", 0)
        nA = size_info.get("Ntotal", 0) - n
        if A:
            rn = [f"fitted.APredictor.{_inla_num(i)}" for i in range(1, nA + 1)] + \
                 [f"fitted.Predictor.{_inla_num(i)}" for i in range(1, n + 1)]
        else:
            rn = [f"fitted.Predictor.{_inla_num(i)}" for i in range(1, size_info.get("Ntotal", 0) + 1)]

        try:
            out["summary.fitted.values"] = pd.DataFrame(Mat, columns=colnames, index=rn)
        except Exception:
            out["summary.fitted.values"] = {"values": Mat, "_rownames": rn, "_colnames": colnames}

        rr_list = inla_interpret_vector_list(inla_read_binary_file(os.path.join(subu, "marginal-densities.dat")))
        if rr_list is not None:
            if A:
                names_rr = [f"fitted.APredictor.{_inla_num(i)}" for i in range(1, nA + 1)] + \
                           [f"fitted.Predictor.{_inla_num(i)}" for i in range(1, n + 1)]
            else:
                # Follow R: use n (not Ntotal)
                names_rr = [f"fitted.Predictor.{_inla_num(i)}" for i in range(1, n + 1)]
                if len(names_rr) != len(rr_list):
                    names_rr = [f"fitted.Predictor.{_inla_num(i)}" for i in range(1, len(rr_list) + 1)]
            out["marginals.fitted.values"] = {names_rr[k]: _to_xy_df(rr_list[k]) for k in range(len(rr_list))}
        else:
            out["marginals.fitted.values"] = None

    return out

def _read_fmesher(path: str):
    """
    Best-effort GMRFLib/INLA 'fmesher' dense/sparse reader (binary).
    This supports many matrix dumps used by INLA.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            # Header length (int32)
            hb = _read_bytes(f, 4)
            H = struct.unpack("<i", hb)[0]
            # Read H int32 in header
            hdr = _read_bytes(f, 4 * H)
            ints = np.frombuffer(hdr, dtype="<i4")
            if ints.size < 6:
                return None
            elems, nrow, ncol, dtype_flag, valtype_flag = ints[1], ints[2], ints[3], ints[4], ints[5]
            val_dtype = "<i4" if valtype_flag == 0 else "<f8"

            if dtype_flag == 0:  # dense
                b = _read_bytes(f, elems * (4 if valtype_flag == 0 else 8))
                data = np.frombuffer(b, dtype=val_dtype)
                if data.size != elems:
                    return None
                return data.reshape((nrow, ncol), order="F")
            else:  # COO
                ib = _read_bytes(f, elems * 4)
                jb = _read_bytes(f, elems * 4)
                vb = _read_bytes(f, elems * (4 if valtype_flag == 0 else 8))
                i = np.frombuffer(ib, dtype="<i4")
                j = np.frombuffer(jb, dtype="<i4")
                v = np.frombuffer(vb, dtype=val_dtype)
                if i.size != elems or j.size != elems or v.size != elems:
                    return None
                if _HAS_SCIPY:
                    return _scipy_coo((v, (i, j)), shape=(nrow, ncol))
                return {"rows": i, "cols": j, "data": v, "shape": (nrow, ncol)}
    except Exception:
        return None


def _collect_misc_configs(d: str) -> Optional[Dict[str, Any]]:
    """
    Read misc/config/configs.dat (post-optim configs).
    """
    fnm = os.path.join(d, "config", "configs.dat")
    if not os.path.exists(fnm):
        return None
    res: Dict[str, Any] = {".preopt": False}
    try:
        with open(fnm, "rb") as fp:
            iarr = _read_int32s(fp, 3)
            n, nz, ntheta = map(int, iarr.tolist())
            res.update({"n": n, "nz": nz, "ntheta": ntheta})

            configs_i = _read_int32s(fp, nz)   # 0-based
            configs_j = _read_int32s(fp, nz)   # 0-based
            res["i"] = configs_i
            res["j"] = configs_j

            nconfig = int(_read_int32s(fp, 1)[0])
            res["nconfig"] = nconfig

            nc = int(_read_int32s(fp, 1)[0])
            if nc > 0:
                A = _read_float64s(fp, n * nc).reshape(nc, n, order="C")
                e = _read_float64s(fp, nc)
                res["constr"] = {"nc": nc, "A": A, "e": e}
            else:
                res["constr"] = None

            # external content files
            theta_tag = _read_lines(os.path.join(d, "config", "theta-tag.dat")) or []
            res["contents"] = {
                "tag": _read_lines(os.path.join(d, "config", "tag.dat")) or [],
                "start": (np.asarray(_read_lines(os.path.join(d, "config", "start.dat")) or [], dtype=float).astype(int) + 1).tolist(),
                "length": np.asarray(_read_lines(os.path.join(d, "config", "n.dat")) or [], dtype=float).astype(int).tolist()
            }

            configs = []
            # Pre-allocate transpose helper
            dif = np.where(configs_i != configs_j)[0]
            iadd = configs_j[dif] if dif.size > 0 else np.array([], dtype=np.int32)
            jadd = configs_i[dif] if dif.size > 0 else np.array([], dtype=np.int32)

            for _ in range(nconfig):
                log_post = float(_read_float64s(fp, 1)[0])
                log_post_orig = float(_read_float64s(fp, 1)[0])
                if ntheta > 0:
                    theta = _read_float64s(fp, ntheta)
                    # attach names
                    if theta_tag and len(theta_tag) == len(theta):
                        theta = pd.Series(theta, index=theta_tag)
                else:
                    theta = None

                mean = _read_float64s(fp, n)
                improved_mean = _read_float64s(fp, n)
                skewness = _read_float64s(fp, n)
                offsets = _read_float64s(fp, n)
                mean = mean + offsets
                improved_mean = improved_mean + offsets

                Q = _read_float64s(fp, nz)
                Qinv = _read_float64s(fp, nz)
                Qprior = _read_float64s(fp, n)

                # Symmetrize
                if dif.size > 0:
                    Q_all = np.concatenate([Q, Q[dif]])
                    Qinv_all = np.concatenate([Qinv, Qinv[dif]])
                    i_all = np.concatenate([configs_i, iadd])
                    j_all = np.concatenate([configs_j, jadd])
                else:
                    Q_all, Qinv_all = Q, Qinv
                    i_all, j_all = configs_i, configs_j

                Q_sp = _scipy_or_dict((n, n), i_all, j_all, Q_all)
                Qinv_sp = _scipy_or_dict((n, n), i_all, j_all, Qinv_all)

                configs.append({
                    ".preopt": False,
                    "theta": theta,
                    "log.posterior": log_post,
                    "log.posterior.orig": log_post_orig,
                    "mean": mean,
                    "improved.mean": improved_mean,
                    "skewness": skewness,
                    "Q": Q_sp,
                    "Qinv": Qinv_sp,
                    "Qprior.diag": Qprior
                })

            # rescale the log.posteriors
            if configs:
                max_lp_orig = max(c["log.posterior.orig"] for c in configs)
                max_lp = max(c["log.posterior"] for c in configs)
                for c in configs:
                    c["log.posterior"] = c["log.posterior"] - max_lp
                    c["log.posterior.orig"] = c["log.posterior.orig"] - max_lp_orig
                res["max.log.posterior"] = max_lp_orig
            else:
                res["max.log.posterior"] = None

            res["config"] = configs
        return res
    except Exception:
        return None


def _read_r_string(fp) -> str:
    """
    Attempt to read an R `writeBin(character(), 1)` string: R writes a 32-bit length
    followed by that many bytes of (usually) UTF-8.
    """
    try:
        n = int(_read_int32s(fp, 1)[0])
    except Exception:
        return ""
    if n <= 0:
        return ""
    b = _read_bytes(fp, n)
    try:
        return b.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _collect_misc_configs_preopt(d: str) -> Optional[Dict[str, Any]]:
    """
    Read misc/config_preopt/configs.dat (pre-optimization configs with prediction payloads).
    """
    fnm = os.path.join(d, "config_preopt", "configs.dat")
    if not os.path.exists(fnm):
        return None
    res: Dict[str, Any] = {".preopt": True}
    try:
        with open(fnm, "rb") as fp:
            iarr = _read_int32s(fp, 9)
            (lite, mpred, npred, mnpred, Npred, n, nz, prior_nz, ntheta) = map(int, iarr.tolist())
            res.update({"lite": bool(lite), "mpred": mpred, "npred": npred, "mnpred": mnpred, "Npred": Npred,
                        "n": n, "nz": nz, "prior_nz": prior_nz, "ntheta": ntheta})

            configs_i = _read_int32s(fp, nz)
            configs_j = _read_int32s(fp, nz)
            iprior = _read_int32s(fp, prior_nz)
            jprior = _read_int32s(fp, prior_nz)
            res["i"] = configs_i
            res["j"] = configs_j
            res["iprior"] = iprior
            res["jprior"] = jprior

            nconfig = int(_read_int32s(fp, 1)[0])
            res["nconfig"] = nconfig

            nc = int(_read_int32s(fp, 1)[0])
            if nc > 0:
                A = _read_float64s(fp, n * nc).reshape(nc, n, order="C")
                e = _read_float64s(fp, nc)
                res["constr"] = {"nc": nc, "A": A, "e": e}
            else:
                res["constr"] = None

            offsets = _read_float64s(fp, mnpred)
            res["offsets"] = offsets

            # external content files
            theta_tag = _read_lines(os.path.join(d, "config_preopt", "theta-tag.dat")) or []
            res["contents"] = {
                "tag": _read_lines(os.path.join(d, "config_preopt", "tag.dat")) or [],
                "start": (np.asarray(_read_lines(os.path.join(d, "config_preopt", "start.dat")) or [],
                                     dtype=float).astype(int) + 1).tolist(),
                "length": np.asarray(_read_lines(os.path.join(d, "config_preopt", "n.dat")) or [],
                                     dtype=float).astype(int).tolist()
            }

            # Matrices A and pA (FMesher dump)
            res["A"] = _read_fmesher(os.path.join(d, "config_preopt", "A.dat"))
            pA_path = os.path.join(d, "config_preopt", "pA.dat")
            if os.path.exists(pA_path):
                res["pA"] = _read_fmesher(pA_path)
            else:
                # empty (0 × nrow(A))
                nrow_A = res["A"].shape[0] if hasattr(res["A"], "shape") else 0
                res["pA"] = np.empty((0, nrow_A))

            configs = []

            # Symmetrization helpers
            dif = np.where(configs_i != configs_j)[0]
            iadd = configs_j[dif] if dif.size > 0 else np.array([], dtype=np.int32)
            jadd = configs_i[dif] if dif.size > 0 else np.array([], dtype=np.int32)
            difp = np.where(iprior != jprior)[0]
            iprior_add = jprior[difp] if difp.size > 0 else np.array([], dtype=np.int32)
            jprior_add = iprior[difp] if difp.size > 0 else np.array([], dtype=np.int32)

            for _ in range(nconfig):
                log_post = float(_read_float64s(fp, 1)[0])
                log_post_orig = float(_read_float64s(fp, 1)[0])
                if ntheta > 0:
                    theta = _read_float64s(fp, ntheta)
                    if theta_tag and len(theta_tag) == len(theta):
                        theta = pd.Series(theta, index=theta_tag)
                else:
                    theta = None

                mean = _read_float64s(fp, n)
                improved_mean = _read_float64s(fp, n)

                Q = _read_float64s(fp, nz)
                Qinv = _read_float64s(fp, nz)
                Qprior = _read_float64s(fp, prior_nz)

                # Optional prediction payloads
                output = _read_float64s(fp, 2)
                if output.size == 2 and output[0] > 0:
                    cpod_mv = _read_float64s(fp, Npred * 3).reshape(-1, 3)
                else:
                    cpod_mv = np.empty((0, 3))
                if output.size == 2 and output[1] > 0:
                    gcpod_mv = _read_float64s(fp, Npred * 3).reshape(-1, 3)
                else:
                    gcpod_mv = np.empty((0, 3))

                # Optional strings (R writeBin(character()))
                arg_str = None
                try:
                    have_arg = int(float(_read_float64s(fp, 1)[0]))
                except Exception:
                    have_arg = 0
                if have_arg > 0:
                    arg_str = []
                    for _i in range(Npred):
                        s = _read_r_string(fp)
                        arg_str.append(s)

                # Optional log-likelihood differential info
                ll_info = None
                try:
                    have_ll = int(float(_read_float64s(fp, 1)[0]))
                except Exception:
                    have_ll = 0
                if have_ll > 0:
                    ll_vec = _read_float64s(fp, 3 * Npred)
                    ll_vec[np.isnan(ll_vec)] = np.nan
                    ll_info = ll_vec.reshape(Npred, 3)
                    # columns: gradient, hessian, deriv3

                # Optional predictor moments (with offsets)
                AP_mv = np.empty((0, 2))
                P_mv = np.empty((0, 2))
                have_lpred = 0
                try:
                    have_lpred = int(float(_read_float64s(fp, 1)[0]))
                except Exception:
                    pass
                if have_lpred > 0:
                    lpred_mean = _read_float64s(fp, mnpred) + offsets[:mnpred]
                    lpred_var = _read_float64s(fp, mnpred)
                    lpred_mean[np.isnan(lpred_mean)] = np.nan
                    lpred_var[np.isnan(lpred_var)] = np.nan
                    off = 0
                    if mpred > 0:
                        off = mpred
                        idx = np.arange(0, mpred)
                        AP_mv = np.column_stack([lpred_mean[idx], lpred_var[idx]])
                    idx = np.arange(off, off + npred)
                    P_mv = np.column_stack([lpred_mean[idx], lpred_var[idx]])

                # Symmetrize sparse pieces
                if dif.size > 0:
                    Q_all = np.concatenate([Q, Q[dif]])
                    Qinv_all = np.concatenate([Qinv, Qinv[dif]])
                    i_all = np.concatenate([configs_i, iadd])
                    j_all = np.concatenate([configs_j, jadd])
                else:
                    Q_all, Qinv_all = Q, Qinv
                    i_all, j_all = configs_i, configs_j

                if difp.size > 0:
                    Qprior_all = np.concatenate([Qprior, Qprior[difp]])
                    ip_all = np.concatenate([iprior, iprior_add])
                    jp_all = np.concatenate([jprior, jprior_add])
                else:
                    Qprior_all = Qprior
                    ip_all, jp_all = iprior, jprior

                Q_sp = _scipy_or_dict((n, n), i_all, j_all, Q_all)
                Qinv_sp = _scipy_or_dict((n, n), i_all, j_all, Qinv_all)
                Qprior_sp = _scipy_or_dict((n, n), ip_all, jp_all, Qprior_all)

                configs.append({
                    "theta": theta,
                    "log.posterior": log_post,
                    "log.posterior.orig": log_post_orig,
                    "mean": mean,
                    "improved.mean": improved_mean,
                    "Q": Q_sp,
                    "Qinv": Qinv_sp,
                    "Qprior": Qprior_sp,
                    "cpodens.moments": cpod_mv,
                    "gcpodens.moments": gcpod_mv,
                    "arg.str": arg_str,
                    "ll.info": ll_info,
                    "APredictor": AP_mv,
                    "Predictor": P_mv
                })

            # Rescale log posteriors
            if configs:
                max_lp_orig = max(c["log.posterior.orig"] for c in configs)
                max_lp = max(c["log.posterior"] for c in configs)
                res["max.log.posterior"] = max_lp_orig
                for c in configs:
                    c["log.posterior"] = c["log.posterior"] - max_lp
                    c["log.posterior.orig"] = c["log.posterior.orig"] - max_lp_orig
            else:
                res["max.log.posterior"] = None

            res["config"] = configs
        return res
    except Exception:
        return None


def _collect_misc(results_dir: str, debug: bool = False) -> Optional[Dict[str, Any]]:
    d = os.path.join(results_dir, "misc")
    d_info = os.path.isdir(d)
    if debug:
        print(f"collect misc from {d}")
    if not d_info:
        return None

    # theta tags & (raw) transforms
    tags = _read_lines(os.path.join(d, "theta-tags"))
    theta_from = _read_lines(os.path.join(d, "theta-from"))
    theta_to = _read_lines(os.path.join(d, "theta-to"))

    # cov internals: format is [n,  n^2 values]
    cov_intern = None; cor_intern = None
    vv = _read_float64_auto(os.path.join(d, "covmat-hyper-internal.dat"))
    if vv is not None and vv.size >= 1:
        n = int(vv[0]); rest = vv[1:]
        if rest.size == n * n:
            cov_intern = rest.reshape(n, n)
            dd = np.diag(cov_intern).astype(float)
            s = np.zeros((n, n))
            with np.errstate(divide="ignore", invalid="ignore"):
                s[np.diag_indices(n)] = 1.0 / np.sqrt(dd)
            cor_intern = s @ cov_intern @ s
            np.fill_diagonal(cor_intern, 1.0)

    ev = _read_float64_auto(os.path.join(d, "covmat-eigenvectors.dat"))
    cov_intern_evec = None
    if ev is not None and ev.size >= 1:
        n = int(ev[0]); rest = ev[1:]
        if rest.size == n * n:
            cov_intern_evec = rest.reshape(n, n)

    evals = _read_float64_auto(os.path.join(d, "covmat-eigenvalues.dat"))
    cov_intern_eval = None
    if evals is not None and evals.size >= 1:
        n = int(evals[0]); rest = evals[1:]
        if rest.size == n:
            cov_intern_eval = rest

    reord = _read_float64_auto(os.path.join(d, "reordering.dat"))
    if reord is not None:
        reord = reord.astype(np.int32)

    stpos = _read_fmesher(os.path.join(d, "stdev_corr_pos.dat"))
    stneg = _read_fmesher(os.path.join(d, "stdev_corr_neg.dat"))
    ldcorr = _read_fmesher(os.path.join(d, "lincomb_derived_correlation_matrix.dat"))
    ldcov = _read_fmesher(os.path.join(d, "lincomb_derived_covariance_matrix.dat"))
    optdir = _read_fmesher(os.path.join(d, "opt_directions.dat"))

    #mode_status = _read_float64_auto(os.path.join(d, "mode-status.dat"))
    #nfunc = _read_float64_auto(os.path.join(d, "nfunc.dat"))
    #lpm = _read_float64_auto(os.path.join(d, "log-posterior-mode.dat"))
    #log_post_mode = (None if lpm is None or lpm.size == 0 else float(lpm[0]))

    mode_status = _read_numeric_text_or_binary(os.path.join(d, "mode-status.dat"))
    nfunc       = _read_numeric_text_or_binary(os.path.join(d, "nfunc.dat"))
    lpm         = _read_numeric_text_or_binary(os.path.join(d, "log-posterior-mode.dat"))
    log_post_mode = (None if lpm is None or lpm.size == 0 else float(lpm[0]))

    # Config readers
    configs_post = _collect_misc_configs(d)
    configs_pre = _collect_misc_configs_preopt(d)

    warn = _read_lines(os.path.join(d, "warnings.txt"))

    # Optimizer trace (optional)
    opt_trace = None
    fnm = os.path.join(d, "opt-trace.dat")
    if os.path.exists(fnm):
        try:
            with open(fnm, "rb") as fp:
                nt = int(_read_int32s(fp, 1)[0])
                niter = int(_read_int32s(fp, 1)[0])
                nfuncs = _read_int32s(fp, niter)
                fs = _read_float64s(fp, niter)
                theta = _read_float64s(fp, niter * nt).reshape(niter, nt, order="C")
            opt_trace = {"f": fs, "nfunc": nfuncs, "theta": theta}
        except Exception:
            opt_trace = None

    if debug:
        print(f"collect misc from {d} ...done")

    # Linkfunctions (kept here as well; R adds to misc)
    lfn = _read_lines(os.path.join(results_dir, "linkfunctions.names"))
    link_idx = None
    link_path = os.path.join(results_dir, "linkfunctions.link")
    if os.path.exists(link_path):
        with open(link_path, "rb") as f:
            try:
                n = struct.unpack("<i", _read_bytes(f, 4))[0]
                idx = np.frombuffer(_read_bytes(f, 8 * n), dtype="<f8")
                ok = ~np.isnan(idx)
                idx2 = np.full_like(idx, np.nan, dtype=float)
                idx2[ok] = idx[ok] + 1.0  # 1-based family index like R
                link_idx = idx2.astype(float)
            except Exception:
                link_idx = None

    misc: Dict[str, Any] = {
        "cov.intern": cov_intern, "cor.intern": cor_intern,
        "cov.intern.eigenvalues": cov_intern_eval, "cov.intern.eigenvectors": cov_intern_evec,
        "reordering": reord, "theta.tags": tags, "log.posterior.mode": log_post_mode,
        "stdev.corr.negative": np.asarray(stneg) if stneg is not None else None,
        "stdev.corr.positive": np.asarray(stpos) if stpos is not None else None,
        "lincomb.derived.correlation.matrix": np.asarray(ldcorr) if ldcorr is not None else None,
        "lincomb.derived.covariance.matrix": np.asarray(ldcov) if ldcov is not None else None,
        "opt.directions": np.asarray(optdir) if optdir is not None else None,
        "mode.status": mode_status, "nfunc": nfunc,
        "warnings": warn, "opt.trace": opt_trace,
        "configs": configs_post if configs_post is not None else configs_pre,
        # raw transform strings (kept for parity; not evaluated)
        "to.theta": theta_to, "from.theta": theta_from,
    }

    if lfn is not None and link_idx is not None:
        misc["linkfunctions"] = {"names": lfn, "link": link_idx}
        misc["family"] = link_idx

    return misc

def _collect_hyperpar(results_dir: str) -> Dict[str, Any]:
    alldir = sorted(os.listdir(results_dir))
    hyp_all = [d for d in alldir if d.startswith("hyperparameter")]
    hyp_usr = [d for d in hyp_all if d.endswith("user-scale")]
    hyp_int = [d for d in hyp_all if not d.endswith("user-scale")]

    def _decode_mean_sd(path: str) -> Optional[Tuple[float, float]]:
        dd = _read_float64_auto(path)
        if dd is None or dd.size == 0:
            return None
        # R does: inla.read.binary.file(file)[-1L]
        # Typical layout is [sentinel, mean, sd]; keep robust fallbacks.
        if dd.size >= 3:
            dd2 = dd[1:3]
        elif dd.size >= 2:
            # If only two numbers are present, assume they are [mean, sd]
            dd2 = dd[:2]
        else:
            return None
        return float(dd2[0]), float(dd2[1])

    def _one(hdirs: List[str]) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, np.ndarray]]]:
        n = len(hdirs)
        if n == 0:
            return None, None
        names: List[str] = []
        rows: List[np.ndarray] = []
        colnames: Optional[List[str]] = None
        margdict: Dict[str, np.ndarray] = {}

        for dname in hdirs:
            base = os.path.join(results_dir, dname)
            nm = (_read_lines(os.path.join(base, "TAG")) or ["missing NAME"])[0]
            names.append(nm)

            # --- mean, sd (drop sentinel)
            ms = _decode_mean_sd(os.path.join(base, "summary.dat"))
            if ms is not None:
                row_vals = [ms[0], ms[1]]
            else:
                row_vals = [np.nan, np.nan]
            cn = ["mean", "sd"]

            # --- quantiles: some hyperpar files store [0, N, p1, q1, p2, q2, ...]
            qv = _read_float64_auto(os.path.join(base, "quantiles.dat"))
            if qv is not None and qv.size:
                if qv.size >= 2 and abs(qv[0]) < 1e-12 and (qv.size - 2) % 2 == 0:
                    # strip header 0, N
                    qmat = qv[2:].reshape(-1, 2)
                else:
                    qmat = _interpret_vector_pairs(qv)
                if qmat is not None and qmat.size:
                    row_vals.extend([float(v) for v in qmat[:, 1]])
                    cn.extend([f"{_fmt_prob(p)}quant" for p in qmat[:, 0]])

            # --- mode: may also carry [0, 1, x, y]
            mv = _read_float64_auto(os.path.join(base, "mode.dat"))
            if mv is not None and mv.size:
                if mv.size >= 4 and abs(mv[0]) < 1e-12 and int(round(mv[1])) == 1:
                    row_vals.append(float(mv[3]))
                    cn.append("mode")
                else:
                    mm2 = _interpret_vector_pairs(mv)
                    if mm2 is not None and mm2.size:
                        row_vals.append(float(mm2[0, 1]))
                        cn.append("mode")

            # --- cdf (prob, value) pairs with optional header 0, N
            cv = _read_float64_auto(os.path.join(base, "cdf.dat"))
            if cv is not None and cv.size:
                if cv.size >= 2 and abs(cv[0]) < 1e-12 and (cv.size - 2) % 2 == 0:
                    cmat = cv[2:].reshape(-1, 2)
                else:
                    cmat = _interpret_vector_pairs(cv)
                if cmat is not None and cmat.size:
                    row_vals.extend([float(v) for v in cmat[:, 1]])
                    cn.extend([f"{_fmt_prob(p)}cdf" for p in cmat[:, 0]])

            if colnames is None:
                colnames = cn
            # pad to uniform width if later rows miss some columns
            if len(row_vals) < len(colnames):
                row_vals += [np.nan] * (len(colnames) - len(row_vals))

            rows.append(np.array(row_vals, dtype=float))

            # --- marginals: single (N×2) matrix (NOT a list-of-lists)
            xx = inla_read_binary_file(os.path.join(base, "marginal-densities.dat"))
            rr = inla_interpret_vector(xx)
            if rr is not None:
                margdict[nm] = _to_xy_df(rr)


        Mat = np.vstack(rows) if rows else None
        # Normalize columns to standard set/order (mean, sd, 0.025/0.5/0.975 quant, mode, then cdf)
        if Mat is not None and colnames is not None and Mat.size:
            Mat, colnames = _ensure_quant_mode_columns(Mat, colnames)
        table = pd.DataFrame(Mat, index=names, columns=colnames)

        marg = margdict if margdict else None
        return table, marg

    sum_usr, marg_usr = _one(hyp_usr)
    sum_int, marg_int = _one(hyp_int)

    return {
        "summary.hyperpar": sum_usr,
        "marginals.hyperpar": marg_usr,
        "internal.summary.hyperpar": sum_int,
        "internal.marginals.hyperpar": marg_int,
    }

def _collect_model_matrix(results_dir: str) -> Dict[str, Any]:
    """
    Best-effort reader for a saved model matrix under results_dir/model.matrix.
    Tries FMesher-style dumps first, then (i,j,v)+dim triplets.
    Returns {"model.matrix": <sparse or dense object>} or {"model.matrix": None}.
    """
    d = os.path.join(results_dir, "model.matrix")
    if not os.path.isdir(d):
        return {"model.matrix": None}

    # 1) Try FMesher dumps
    for cand in ("matrix.dat", "model-matrix.dat", "model.matrix.dat", "X.dat", "A.dat"):
        p = os.path.join(d, cand)
        M = _read_fmesher(p)
        if M is not None:
            return {"model.matrix": M}

    # 2) Try COO triplets (i,j,v) with optional dim
    def _read_ints_auto(path):
        if not os.path.exists(path):
            return None
        # Allow either binary int32 or float64 files; coerce to int via astype
        v = _read_float64_auto(path)
        if v is None:
            return None
        return v.astype(np.int64)

    i = _read_ints_auto(os.path.join(d, "i.dat"))
    j = _read_ints_auto(os.path.join(d, "j.dat"))
    v = _read_float64_auto(os.path.join(d, "v.dat"))
    dim = _read_float64_auto(os.path.join(d, "dim.dat"))

    if i is not None and j is not None and v is not None and i.size == j.size == v.size:
        if dim is not None and dim.size >= 2:
            shape = (int(dim[0]), int(dim[1]))
        else:
            # infer a shape; assume 0-based indices
            shape = (int(i.max()) + 1, int(j.max()) + 1)
        return {"model.matrix": _scipy_or_dict(shape, i, j, v)}

    return {"model.matrix": None}


# ======================================================================
#                               Main entry
# ======================================================================

def collect_inla_results(entry_dir: str,
                         debug: bool = False,
                         only_hyperparam: bool = False,
                         file_log: Optional[str] = None,
                         file_log2: Optional[str] = None,
                         allow_parent: bool = True) -> Optional[Dict[str, Any]]:
    """
    Collect results roughly equivalent to R's inla.collect.results.

    Parameters
    ----------
    entry_dir : str
        Either a parent folder containing runs (inla.model-*) or a run/results folder.
    debug : bool
        Print progress.
    only_hyperparam : bool
        If True, collect only hyperparameters.
    file_log, file_log2 : Optional[str]
        Optional log files to include (exactly like R).
    allow_parent : bool
        If True, and `entry_dir` looks like a parent, choose latest inla.model-*.

    Returns
    -------
    dict or None
    """
    if not _HAS_BINARY_HELPERS:
        raise RuntimeError(
            "collect_inla_results requires the binary helper module. Reinstall 'pyinla'"
            " or provide a compatible pyinla.binary module to enable result collection."
        )

    def dbg(msg):
        if debug:
            print(f"[collect] {msg}")

    model_dir = _resolve_latest_model_dir(entry_dir) if allow_parent else entry_dir
    if model_dir is None or not os.path.isdir(model_dir):
        dbg(f"Not a directory: {entry_dir}")
        return None

    res_dir = _resolve_results_dir_like_R(model_dir)
    if res_dir is None:
        dbg(f"No valid results.files(-NNNN...) with .ok under: {model_dir}")
        return None

    # If DRYRUN, return early
    dry = os.path.join(res_dir, "dryrun")
    #if os.path.exists(dry):
    #    return {"dryrun": _read_lines(dry), "ok": True}
    if os.path.exists(dry):
        return {"dryrun": _read_lines(dry)}  # remove any extra flags like ok=True

    res_ok = True  # .ok exists by construction

    # ---- Misc first (tags etc)
    dbg("misc"); misc = _collect_misc(res_dir, debug=debug)

    # ---- Main blocks
    if not only_hyperparam:
        dbg("fixed"); fixed = _collect_fixed(res_dir)
        dbg("lincomb"); lin = _collect_lincomb(res_dir, derived=False)
        dbg("lincomb*"); lind = _collect_lincomb(res_dir, derived=True)
        # Name lincomb.derived correlation/covariance matrices like R (ID -> tag ordering)
        try:
            sld = (lind or {}).get("summary.lincomb.derived")
            if isinstance(sld, pd.DataFrame) and "ID" in sld.columns:
                idx = sld["ID"].astype(int).to_numpy()
                tags = sld.index.to_list()
                names = [tags[i - 1] for i in idx if 1 <= i <= len(tags)]
                for key in ("lincomb.derived.correlation.matrix", "lincomb.derived.covariance.matrix"):
                    M = (misc or {}).get(key)
                    if M is None:
                        continue
                    if hasattr(M, "shape") and M.shape == (len(names), len(names)):
                        misc[key] = pd.DataFrame(np.asarray(M), index=names, columns=names)
        except Exception:
            pass

        dbg("dic"); dic = _collect_dic(res_dir)
        dbg("cpo"); cpo = _collect_cpo(res_dir)
        dbg("gcpo"); gcpo = _collect_gcpo(res_dir)
        dbg("po/waic"); po = _collect_po(res_dir); waic = _collect_waic_from_po(res_dir)
        dbg("random"); rnd = _collect_random(res_dir)
        dbg("predictor"); pred = _collect_predictor(res_dir)
        dbg("spde2.blc"); spde2 = _collect_spde2_blc(res_dir)
        dbg("spde3.blc"); spde3 = _collect_spde3_blc(res_dir)
    else:
        fixed = {"names.fixed": None, "summary.fixed": None, "marginals.fixed": None}
        lin = {"summary.lincomb": None, "marginals.lincomb": None, "size.lincomb": None}
        lind = {"summary.lincomb.derived": None, "marginals.lincomb.derived": None, "size.lincomb.derived": None}
        dic = None; cpo = {"cpo": None, "pit": None, "failure": None}
        gcpo = {"gcpo": None, "kld": None, "mean": None, "sd": None, "groups": None}
        po = {"po": None}; waic = None
        rnd = {"model.random": None, "summary.random": None, "marginals.random": None, "size.random": None}
        pred = {"summary.linear.predictor": None, "marginals.linear.predictor": None,
                "summary.fitted.values": None, "marginals.fitted.values": None, "size.linear.predictor": None}
        spde2 = {"model.spde2.blc": None, "summary.spde2.blc": None, "marginals.spde2.blc": None, "size.spde2.blc": None}
        spde3 = {"model.spde3.blc": None, "summary.spde3.blc": None, "marginals.spde3.blc": None, "size.spde3.blc": None}

    dbg("mlik"); mlik = _collect_mlik(res_dir)
    dbg("Q"); Q = _collect_q(res_dir)
    dbg("graph"); graph = _collect_graph(res_dir)

    # offset (top level)
    off_path = os.path.join(res_dir, "totaloffset", "totaloffset.dat")
    if not os.path.exists(off_path):
        raise RuntimeError(f"Missing required file: {off_path}")
    off = _read_float64_auto(off_path)
    offset = {"offset.linear.predictor": off}


    # modes and version
    tm = _read_float64_auto(os.path.join(res_dir, ".theta_mode"))
    if tm is None:
        tm = np.array([])
    xm = _read_float64_auto(os.path.join(res_dir, ".x_mode"))
    if xm is None:
        xm = np.array([])
    theta_mode = tm[1:] if tm.size > 0 else np.array([])
    x_mode = xm[1:] if xm.size > 0 else np.array([])
    gitid = _read_lines(os.path.join(res_dir, ".gitid"))

    # hyperparameters + joint
    if theta_mode.size > 0:
        dbg("hyperpar")
        hyper = _collect_hyperpar(res_dir)
        joint = None
        jp = os.path.join(res_dir, "joint.dat")
        if os.path.exists(jp) and os.path.getsize(jp) > 0:
            try:
                arr = np.loadtxt(jp)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                tags = (misc or {}).get("theta.tags") or []
                cols = list(tags[:max(0, arr.shape[1] - 2)]) + [
                    "Log posterior density", "Total integration weight (log.dens included)"
                ]
                # Export as DataFrame with column names like R
                joint = pd.DataFrame(arr, columns=cols)
            except Exception:
                joint = None
        nhyper = int(theta_mode.size)
    else:
        hyper = {"summary.hyperpar": None, "marginals.hyperpar": None,
                 "internal.summary.hyperpar": None, "internal.marginals.hyperpar": None}
        joint = None
        nhyper = 0

    # logfile(s)
    def _collect_logfile(path: Optional[str]) -> List[str]:
        if not path or not os.path.exists(path):
            return []
        lines = _read_lines(path) or []
        lines = [ln.replace("\t", "        ") for ln in lines]
        return [ln for ln in lines if len(ln) > 0]

    # Fallback to standard INLA log files in model_dir if not provided
    #if not file_log:
    #    cand = os.path.join(model_dir, "Logfile.txt")
    #    if os.path.exists(cand):
    #        file_log = cand
    #if not file_log2:
    #    cand2 = os.path.join(model_dir, "Logfile2.txt")
    #    if os.path.exists(cand2):
    #        file_log2 = cand2

    logfile_vec: List[str] = []
    if file_log:
        logfile_vec += _collect_logfile(file_log)
        logfile_vec += ["", "*" * 72, ""]
    if file_log2:
        logfile_vec += _collect_logfile(file_log2)
    logfile = {"logfile": logfile_vec if logfile_vec else []}

    # residuals: migrate from DIC if both dic and po exist (as in R)
    if dic is not None and po is not None and "po" in po:
        devres = {"deviance.residuals": dic.get("residuals.deviance") if dic else None}
        if dic and "residuals.deviance" in dic:
            dic.pop("residuals.deviance", None)
    else:
        devres = {"deviance.residuals": None}

    # cpu.intern
    cpu_intern = _read_lines(os.path.join(res_dir, "cpu-intern"))

    dbg("model.matrix"); mm = _collect_model_matrix(res_dir)

    # assemble
    res: Dict[str, Any] = {}
    res.update(fixed)
    res.update(lin)
    res.update(lind)
    res.update(mlik)
    res.update({"cpo": cpo})
    res.update({"gcpo": gcpo})
    res.update(po)
    res.update({"waic": waic})
    res.update({"residuals": devres})
    res.update(rnd)
    res.update(pred)
    res.update(hyper)
    res.update(offset)
    res.update(spde2)
    res.update(spde3)
    res.update(logfile)
    res.update({
        "misc": misc,
        "dic": dic,
        "mode": {
            "theta": theta_mode, "x": x_mode,
            "theta.tags": (misc or {}).get("theta.tags") if misc else None,
            "mode.status": (misc or {}).get("mode.status") if misc else None,
            "log.posterior.mode": (misc or {}).get("log.posterior.mode") if misc else None,
        },
        "joint.hyper": joint,
        "nhyper": nhyper,
        "version": {"inla.call": (gitid[0] if gitid else None), "R.INLA": None},
        "Q": Q,
    })
    res.update(graph)
    res.update(mm)
    res["ok"] = res_ok
    res["cpu.intern"] = cpu_intern
    return res

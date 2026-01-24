# utils.py
"""
Python equivalents for many utilities in R-INLA's utils.R.

Highlights
----------
- Numeric/string helpers: inla_numlen, inla_num, inla_trim, inla_namefix, ...
- Matrix helpers: inla_unique_rows, inla_factor2matrix, inla_sparse_matrix_pattern, ...
- Files/paths: inla_tempfile, inla_tempdir, inla_writeLines, inla_readLines, ...
- Linear algebra: inla_ginv, inla_gdet, inla_rw (RW1/RW2 precision), inla_ensure_spd, inla_matern_cf
- Misc: inla_one_of, inla_divisible/even/odd, inla_toeplitz, etc.

Notes
-----
- Some R-specific features that operate on R package namespaces or R packages
  are provided as stubs or best-effort analogues and documented below.
- Sparse matrices rely on SciPy (scipy.sparse). Functions that require SciPy
  will raise ImportError if SciPy is unavailable.
"""

from __future__ import annotations

import os
import re
import sys
import math
import struct
import tempfile
import inspect
import warnings
import importlib
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import scipy.sparse as sp
except Exception:
    sp = None  # Only a subset needs SciPy; we raise where required.

try:
    from scipy.special import kv as bessel_k
except Exception:
    bessel_k = None  # Fallback provided below where used.


# ---------------------------------------------------------------------
# Basic numeric/string helpers
# ---------------------------------------------------------------------

def inla_numlen(x: Union[int, Sequence[int]]) -> int:
    """Number of digits required to represent |x| (scalar or sequence)."""
    arr = np.atleast_1d(x)
    if arr.size == 0:
        return 1
    m = int(np.max(np.abs(arr)))
    if m == 0:
        return 1
    return int(math.floor(math.log10(m)) + 1)


def inla_num(x: Union[float, Sequence[float]],
             width: Optional[int] = None,
             digits: Optional[int] = None) -> Union[str, List[str]]:
    """
    Format numbers with leading zeros (like R's formatC(..., format='g', flag='0')).

    width: default = inla_numlen(x) if len(x)>1 else 10
    digits: default = max(4, width)
    """
    arr = np.atleast_1d(x).astype(float)
    if width is None:
        width = inla_numlen(arr) if arr.size > 1 else 10
    if digits is None:
        digits = max(4, width)

    def fmt(v: float) -> str:
        s = f"{v:.{digits}g}"   # general format with 'digits' significant digits
        if len(s) < width:
            s = s.rjust(width, "0")
        return s

    out = [fmt(v) for v in arr]
    return out if arr.size > 1 else out[0]


def inla_trim(string: Union[str, Sequence[str]]) -> Union[str, List[str]]:
    """Trim leading/trailing whitespace (handles list of strings as well)."""
    if isinstance(string, str):
        return re.sub(r"[ \t]+$", "", re.sub(r"^[ \t]+", "", string))
    return [inla_trim(s) for s in string]


def inla_namefix(string: str) -> str:
    """Replace special characters as in R (repeatedly apply '$' -> '|S|')."""
    old = inla_trim(string)
    while True:
        new = re.sub(r"[$]", r"|S|", old)
        if new == old:
            break
        old = new
    return old


def inla_nameunfix(string: str) -> str:
    # R version is effectively a no-op here
    return string


def inla_strcmp(s: str, ss: str) -> bool:
    return s == ss


def inla_strncmp(s: Union[str, Sequence[str]], ss: Union[str, Sequence[str]]):
    """
    Compare prefix: substr(s, 1, nchar(ss)) == ss
    - If s scalar and ss is a list: vectorized over ss
    - If both lengths > 1: error (as in R)
    """
    if isinstance(s, (list, tuple)) and isinstance(ss, (list, tuple)):
        raise ValueError("length(s) > 1 && length(ss) > 1: not allowed.")
    if isinstance(s, (list, tuple)):
        s = s[0]
    if isinstance(ss, (list, tuple)):
        return [s[:len(t)] == t for t in ss]
    return s[:len(ss)] == ss


def inla_strcasecmp(s: str, ss: str) -> bool:
    return s.lower() == ss.lower()


def inla_strncasecmp(s: Union[str, Sequence[str]], ss: Union[str, Sequence[str]]):
    if isinstance(s, (list, tuple)) and isinstance(ss, (list, tuple)):
        raise ValueError("length(s) > 1 && length(ss) > 1: not allowed.")
    if isinstance(s, (list, tuple)):
        s = s[0]
    if isinstance(ss, (list, tuple)):
        return [s[:len(t)].lower() == t.lower() for t in ss]
    return s[:len(ss)].lower() == ss.lower()


def inla_pause(msg: Optional[str] = None) -> None:
    """Print a message and wait for Enter."""
    if msg:
        print(msg)
    try:
        input()  # pragma: no cover
    except EOFError:
        pass


def inla_paste(strings: Sequence[str], sep: str = " ") -> str:
    return sep.join(map(str, strings))


# ---------------------------------------------------------------------
# R-specific updater (stub)
# ---------------------------------------------------------------------

def inla_my_update(*args, **kwargs):
    """R-specific (namespace hot-reload) helper; not applicable in Python."""
    raise NotImplementedError("inla.my.update is R-specific and not applicable in Python.")


# ---------------------------------------------------------------------
# List/data-frame like helpers
# ---------------------------------------------------------------------

def inla_remove(name: Sequence[str], from_obj: Any) -> Any:
    """
    Remove NAME from a dict (keys) or pandas.DataFrame (columns).
    """
    try:
        import pandas as pd  # type: ignore
    except Exception:
        pd = None

    if isinstance(from_obj, dict):
        for nm in list(name):
            if nm in from_obj:
                del from_obj[nm]
        return from_obj

    if pd is not None and isinstance(from_obj, pd.DataFrame):
        cols = [c for c in from_obj.columns if c in set(name)]
        return from_obj.drop(columns=cols)

    # Fallback: try attribute removal
    for nm in list(name):
        if hasattr(from_obj, nm):
            delattr(from_obj, nm)
    return from_obj


def inla_ifelse(test: bool, yes: Any, no: Any) -> Any:
    return yes if test else no


# ---------------------------------------------------------------------
# Sparse matrix pattern (downsampled/binary)
# ---------------------------------------------------------------------

def _require_scipy():
    if sp is None:
        raise ImportError("SciPy is required for this function (scipy.sparse).")


def _as_coo(A: Any):
    """Local COO converter with duplicate coalescing (inla_as_dgTMatrix-like)."""
    _require_scipy()
    if sp.issparse(A):
        M = A.tocoo(copy=True)
    else:
        M = sp.coo_matrix(A)
    M.sum_duplicates()
    return M


def inla_sparse_matrix_pattern(A: Any,
                               factor: float = 1.0,
                               size: Optional[Tuple[int, int]] = None,
                               reordering: Optional[Union[Sequence[int], Dict[str, Any]]] = None,
                               binary_pattern: bool = True):
    """
    Calculate a (possibly downsampled) binary pattern of a sparse matrix.

    - Uses 1-based scaling logic (as in R), then converts to 0-based for SciPy.
    - 'reordering' can be an index vector (1-based) or a dict with key 'reordering'.
    """
    M = _as_coo(A)
    n = np.array(M.shape, dtype=int)

    # 1-based indices for intermediate (to match R logic)
    ii = M.row.astype(np.int64) + 1
    jj = M.col.astype(np.int64) + 1

    if reordering is not None:
        if isinstance(reordering, dict) and "reordering" in reordering and reordering["reordering"] is not None:
            rr = np.asarray(reordering["reordering"], dtype=int)
            ii = rr[ii - 1]
            jj = rr[jj - 1]
        else:
            rr = np.asarray(reordering, dtype=int)
            ii = rr[ii - 1]
            jj = rr[jj - 1]

    if size is None:
        size = tuple(np.ceil(n * factor).astype(int))
    elif isinstance(size, (int, np.integer)):
        size = (int(size), int(round(size / n[0] * n[1])))
    size = tuple(size)
    fac = np.array(size) / n

    # Scale + clamp (1-based -> 0-based for SciPy)
    si = np.minimum(size[0], np.ceil(ii * fac[0]).astype(int)) - 1
    sj = np.minimum(size[1], np.ceil(jj * fac[1]).astype(int)) - 1
    data = np.ones_like(si, dtype=float)
    if binary_pattern:
        data[:] = 1.0

    out = sp.coo_matrix((data, (si, sj)), shape=size, dtype=float)
    out.sum_duplicates()
    if binary_pattern:
        out.data[:] = 1.0
    return out


# ---------------------------------------------------------------------
# Miscellaneous helpers
# ---------------------------------------------------------------------

def inla_scale(x: np.ndarray) -> np.ndarray:
    """Center and scale columns like R's scale()."""
    arr = np.asarray(x, dtype=float)
    m = arr.mean(axis=0)
    s = arr.std(axis=0, ddof=1)
    s[s == 0] = 1.0
    return (arr - m) / s


def inla_trim_family(family: str) -> str:
    family = re.sub(r":.*$", ":", family)
    family = re.sub(r"[_ \t.]+", "", family).lower()
    return family


def inla_is_list_of_lists(a_list: Any) -> bool:
    return isinstance(a_list, list) and len(a_list) > 0 and all(isinstance(x, list) for x in a_list) and (getattr(a_list, "__dict__", None) is None)


def inla_as_list_of_lists(a: Any) -> Any:
    try:
        import pandas as pd  # type: ignore
    except Exception:
        pd = None

    if isinstance(a, np.ndarray) and a.ndim == 2:
        # list of columns (like as.list(as.data.frame(as.matrix(a)))) in R
        return [list(a[:, j]) for j in range(a.shape[1])]
    if pd is not None and isinstance(a, pd.DataFrame):
        return [list(a[col].values) for col in a.columns]
    if inla_is_list_of_lists(a):
        return a
    return a  # best-effort: return as-is


def inla_replicate_list(a_list: Any, nrep: int) -> List[Any]:
    if nrep <= 0:
        raise ValueError("nrep must be > 0")
    return [a_list for _ in range(nrep)]


def inla_tictac(num: int, elms: str = "|/-\\|/-\\") -> str:
    """
    Return the character from `elms` cycling by position `num`.
    Default sequence matches R helper: '|', '/', '-', '\\', '|', '/', '-', '\\'
    """
    if not elms:
        raise ValueError("elms must be a non-empty string")
    ln = len(elms)
    i = num % ln
    return elms[i]


def inla_2list(x: Optional[Union[str, Sequence[Any]]] = None) -> Optional[str]:
    """
    Return an R-like 'c(...)' string. (Useful for debugging parity.)
    """
    if x is None:
        return None
    if isinstance(x, str):
        s = x.strip()
        s = re.sub(r"^[cC]\s*\((.*)\)$", r"\1", s)
        items = re.split(r"[,\s]+", s.strip()) if s else []
        return "c(" + ",".join(items) + ")"
    if isinstance(x, (list, tuple, np.ndarray)):
        return "c(" + ",".join(str(v) for v in x) + ")"
    return f"c({str(x)})"


def inla_even(n: Union[int, np.ndarray]) -> Union[bool, np.ndarray]:
    return inla_divisible(n, by=2)


def inla_odd(n: Union[int, np.ndarray]) -> Union[bool, np.ndarray]:
    return inla_divisible(n, by=-2)


def inla_divisible(n: Union[int, np.ndarray], by: int = 2) -> Union[bool, np.ndarray]:
    if by == 0:
        return np.ones_like(np.atleast_1d(n), dtype=bool) if not np.isscalar(n) else True
    n_arr = np.atleast_1d(n)
    out = (n_arr % abs(by)) == 0 if by > 0 else (n_arr % abs(by)) != 0
    return out if not np.isscalar(n) else bool(out.item())


def inla_one_of(family: str, candidates: Optional[Sequence[str]]) -> bool:
    if not candidates:
        return False
    f = inla_trim_family(family)
    C = [inla_trim_family(c) for c in candidates]
    return any(f == c for c in C)


def inla_get_HOME() -> str:
    path = os.environ.get("USERPROFILE") if os.name == "nt" else os.environ.get("HOME", "")
    return str(path).replace("\\", "/")


def inla_get_USER() -> str:
    for env in ("USER", "USERNAME", "LOGNAME"):
        u = os.environ.get(env, "")
        if u:
            return str(u)
    return "UnknownUserName"


def inla_eval(command: str, env: Optional[Dict[str, Any]] = None) -> Any:
    """
    Evaluate an expression (or execute statements) in a given dict environment.
    """
    ns = {} if env is None else env
    try:
        return eval(command, ns, ns)
    except SyntaxError:
        exec(command, ns, ns)  # pragma: no cover
        return None


def inla_tempfile(pattern: str = "file", tmpdir: Optional[str] = None) -> str:
    tmpdir = tmpdir or tempfile.gettempdir()
    fd, path = tempfile.mkstemp(prefix=pattern, dir=tmpdir)
    os.close(fd)
    return path.replace("\\", "/")


def inla_tempdir() -> str:
    path = tempfile.mkdtemp()
    return path.replace("\\", "/")


def inla_formula2character(formula: Any) -> str:
    # Python does not have R formulas; return string and remove trailing '()' if present.
    s = str(formula)
    return re.sub(r"\(\)$", "", s)


def inla_unique_rows(A: np.ndarray) -> Dict[str, Any]:
    """
    Return unique rows and a 1-based index mapping, preserving FIRST occurrence
    ordering (matches the R implementation based on duplicated()).
    """
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError("A must be a 2-D matrix")
    n, p = A.shape
    keys = [inla_paste([str(x) for x in row], sep="<|>") for row in A]
    first_index: Dict[str, int] = {}
    rows_list: List[np.ndarray] = []
    idx = np.empty(n, dtype=int)
    k = 0
    for i, key in enumerate(keys):
        if key in first_index:
            idx[i] = first_index[key]
        else:
            k += 1
            first_index[key] = k
            rows_list.append(A[i, :].copy())
            idx[i] = k
    rows = np.vstack(rows_list) if rows_list else np.empty((0, p))
    return {"rows": rows, "idx": idx}  # idx is 1-based already


def inla_is_dir(dir_path: str) -> bool:
    return os.path.isdir(dir_path)


def inla_dirname(path: str) -> str:
    if path.endswith(os.sep):
        return path[:-1]
    return os.path.dirname(path)


def inla_affirm_integer(A: Any, tol: float = np.finfo(float).eps * 2.0) -> Any:
    """
    If A is numerically integer (within tol), cast to integer dtype.
    """
    arr = np.asarray(A)
    if np.all(np.abs(arr - np.round(arr)) <= tol):
        return np.round(arr).astype(np.int64)
    return A


def inla_affirm_double(A: Any) -> Any:
    arr = np.asarray(A)
    return arr.astype(float, copy=False)


def inla_matrix2list(A: np.ndarray, byrow: bool = False) -> Dict[str, List[float]]:
    """
    Convert matrix to dict of row/col lists.
    byrow=False -> first index is columns (as in R helper)
    """
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError("A must be a 2-D matrix")
    if byrow:
        A = A.T
    out: Dict[str, List[float]] = {}
    for i in range(A.shape[0]):
        out[f"row{i+1}"] = list(A[i, :].tolist())
    return out


def inla_dir_create(dir_path: str,
                    showWarnings: bool = True,
                    recursive: bool = True,
                    mode: str = "0777",
                    StopOnError: bool = True) -> Optional[str]:
    if inla_is_dir(dir_path):
        return dir_path
    try:
        os.makedirs(dir_path, exist_ok=recursive)
        return dir_path
    except Exception:
        if StopOnError:
            raise RuntimeError(f"Failed to create directory [{dir_path}]. Stop.")
        return None


def inla_is_element(name: str, alist: Dict[str, Any]) -> bool:
    if name not in alist:
        return False
    v = alist[name]
    if v is None:
        return False
    # length==0
    if hasattr(v, "__len__") and not isinstance(v, (str, bytes)) and len(v) == 0:
        return False
    # NA
    try:
        if isinstance(v, float) and math.isnan(v):
            return False
        if isinstance(v, np.ndarray) and v.size == 1 and np.isnan(v).all():
            return False
    except Exception:
        pass
    return True


def inla_get_element(name: str, alist: Dict[str, Any]) -> Any:
    return alist[name] if inla_is_element(name, alist) else None


def inla_require_inherits(x: Any, what: Union[type, Tuple[type, ...]], name: str = "Object") -> None:
    if not isinstance(x, what):
        if isinstance(what, tuple):
            opts = ", ".join(w.__name__ for w in what[:-1]) + f", or {what[-1].__name__}"
        else:
            opts = what.__name__
        raise TypeError(f"{name} must inherit from class {opts}.")
    return None


def inla_function2source(the_function: Any, newline: str = "<<NEWLINE>>") -> str:
    src = inspect.getsource(the_function)
    return newline.join(src.splitlines())


def inla_source2function(the_source: str, newline: str = "<<NEWLINE>>") -> Any:
    if the_source == "(null)":
        warnings.warn("'the.source' is NULL, use identity mapping")
        code = "def _tmp(x):\n    return x"
    else:
        code = "\n".join(the_source.split(newline))
    ns: Dict[str, Any] = {}
    exec(code, ns, ns)
    funcs = [v for v in ns.values() if callable(v)]
    if not funcs:
        raise ValueError("No function defined in source.")
    return funcs[-1]


def inla_writeLines(filename: str, lines: Sequence[str]) -> str:
    """
    Write lines in a platform-agnostic binary format:
    int32: number of lines
    for each line: int32 nchars, then raw bytes (utf-8)
    """
    with open(filename, "wb") as fp:
        fp.write(struct.pack("=i", len(lines)))
        for s in lines:
            b = s.encode("utf-8")
            fp.write(struct.pack("=i", len(b)))
            fp.write(b)
    return filename


def inla_readLines(filename: str) -> Optional[List[str]]:
    if not os.path.exists(filename):
        return None
    out: List[str] = []
    with open(filename, "rb") as fp:
        data = fp.read(4)
        if len(data) < 4:
            return []
        (nlines,) = struct.unpack("=i", data)
        for _ in range(nlines):
            (ln,) = struct.unpack("=i", fp.read(4))
            out.append(fp.read(ln).decode("utf-8"))
    return out


def inla_is_matrix(A: Any) -> bool:
    if sp is not None and sp.issparse(A):
        return True
    if isinstance(A, np.ndarray):
        return A.ndim == 2
    shp = getattr(A, "shape", None)
    return isinstance(shp, tuple) and len(shp) == 2


def pd_isna(x: Any) -> np.ndarray:
    try:
        import pandas as pd  # type: ignore
        return pd.isna(x)
    except Exception:
        x = np.asarray(x, dtype=object)
        return np.equal(x, None)  # crude fallback


def inla_factor2matrix(f: Sequence[Any], sparse: bool = False):
    """
    Build indicator matrix for a categorical vector.
    If sparse=True, return scipy.sparse.coo_matrix; else dense numpy array.
    """
    if sparse:
        _require_scipy()

    f = np.asarray(list(f), dtype=object)
    ok = ~pd_isna(f)
    levels = np.unique(f[ok])
    n = f.size
    k = levels.size
    if sparse:
        # map value -> level index
        mapping = {levels[i]: i for i in range(k)}
        codes = np.zeros(n, dtype=int)
        for idx in np.where(ok)[0]:
            codes[idx] = mapping[f[idx]]
        i = np.where(ok)[0]
        j = codes[ok]
        data = np.ones_like(i, dtype=float)
        return sp.coo_matrix((data, (i, j)), shape=(n, k))
    else:
        M = np.zeros((n, k), dtype=float)
        for c, lvl in enumerate(levels):
            M[ok & (f == lvl), c] = 1.0
        return M


def inla_require(pkg: str, stop_on_error: bool = False, quietly: bool = True, **kwargs) -> bool:
    try:
        importlib.import_module(pkg)
        return True
    except Exception:
        if stop_on_error:
            raise RuntimeError(f"Package '{pkg}' is required to proceed, but is not installed. Please install.")
        return False


def inla_inlaprogram_has_crashed() -> None:
    raise RuntimeError("The inla-program exited with an error. Rerun with verbose output; if it persists, contact <help@r-inla.org>.")


def inla_inlaprogram_timeout(timeused: float, timeout: float) -> None:
    if timeout > 0 and timeused > timeout:
        raise TimeoutError(f"*** Interrupted after {timeused:.1f} seconds due to timeout = {timeout:.1f} seconds")


def inla_eval_dots(stop_if_no_name: bool = True, allowed_names: Optional[Sequence[str]] = None, **kwargs) -> Dict[str, Any]:
    """
    Python analogue: return kwargs, optionally validating allowed names.
    """
    if allowed_names is not None:
        for k in kwargs.keys():
            if k not in set(allowed_names):
                raise ValueError(f"This argument is not allowed: {k}")
    return kwargs


def match_arg_vector(arg: Optional[List[str]], choices: Sequence[str], length: Optional[int] = None) -> List[str]:
    if length is None:
        length = 1 if arg is None else len(arg)
    if arg is None:
        arg = [choices[0]]
    else:
        for i, v in enumerate(arg):
            if v not in choices:
                raise ValueError(f"Invalid option '{v}'. Must be one of {choices}")
            arg[i] = v
    if len(arg) < length:
        arg = arg + arg * (length - len(arg))
    elif len(arg) > length:
        raise ValueError("Option list too long.")
    return arg


def inla_get_var(var: str, data: Optional[Dict[str, Any]] = None) -> Any:
    if data is None:
        return globals().get(var, None)
    return data.get(var, None)


# ---------------------------------------------------------------------
# Linear algebra utilities
# ---------------------------------------------------------------------

def inla_ginv(x: np.ndarray, tol: float = math.sqrt(sys.float_info.epsilon), rankdef: Optional[int] = None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    U, s, Vt = np.linalg.svd(x, full_matrices=False)
    if rankdef is not None and rankdef > 0:
        n = s.size
        if not (1 <= rankdef <= n):
            raise ValueError("rankdef out of range")
        Positive = np.array([True] * (n - rankdef) + [False] * rankdef, dtype=bool)
    else:
        Positive = s > max(tol * s[0], 0.0)
    if not np.any(Positive):
        return np.zeros((x.shape[1], x.shape[0]))
    s_inv = np.zeros_like(s)
    s_inv[Positive] = 1.0 / s[Positive]
    return (Vt.T @ np.diag(s_inv) @ U.T)


def inla_gdet(x: np.ndarray, tol: float = math.sqrt(sys.float_info.epsilon), rankdef: Optional[int] = None, log: bool = True) -> float:
    x = np.asarray(x, dtype=float)
    vals = np.linalg.eigvals(x)
    vals = np.real(vals)
    if rankdef is None or rankdef == 0:
        non_zero = vals > max(tol * float(np.max(vals)), 0.0)
        vals = vals[non_zero]
    else:
        vals = np.sort(vals)
        vals = vals[rankdef:]  # drop the smallest 'rankdef'
    return float(np.sum(np.log(vals))) if log else float(np.prod(vals))


def inla_rw1(n: int, **kwargs) -> Any:
    return inla_rw(n, order=1, **kwargs)


def inla_rw2(n: int, **kwargs) -> Any:
    return inla_rw(n, order=2, **kwargs)


def inla_rw(n: int, order: int = 1, sparse: bool = True, scale_model: bool = False, cyclic: bool = False) -> Any:
    """
    Random-walk precision matrix of given order.
    - If sparse=True: return scipy.sparse.coo_matrix
    - scale_model=True rescales so that generalized variance is 1 (as in R)
    """
    if n < 1 + 2 * order:
        raise ValueError("n must be >= 1 + 2*order")
    _require_scipy()

    if not cyclic:
        U = np.diff(np.eye(n), n=order, axis=0)
        Q = U.T @ U
    else:
        m = 2 * order + 1
        k = 1 + order  # R 1-based
        U = np.diff(np.eye(m), n=order, axis=0)
        U = U.T @ U
        # Toeplitz-like wrap (convert to 0-based carefully)
        first_col = np.concatenate([
            U[k-1, k-1:m],                           # U[k, k:m] in R
            np.zeros(n - m),
            U[k-1, m-1:k-1:-1]                       # U[k, m:(k+1)] in R
        ])
        Q = inla_toeplitz(first_col)

    if scale_model:
        rd = order if not cyclic else (1 if order == 1 else order - 1)
        fac = math.exp(np.mean(np.log(np.diag(inla_ginv(np.asarray(Q), rankdef=rd)))))
        Q = fac * Q

    if sparse:
        return sp.coo_matrix(Q)
    return Q


def inla_mclapply(func, iterable, mc_cores: Optional[int] = None, parallel: bool = True):
    """
    Minimal parallel map. On Windows we fall back to serial map by default.
    """
    if not parallel or os.name == "nt":
        return list(map(func, iterable))
    import multiprocessing as mp  # type: ignore
    if mc_cores is None:
        mc_cores = max(1, (os.cpu_count() or 1))
    with mp.Pool(processes=mc_cores) as pool:
        return pool.map(func, iterable)


def inla_cmpfun(fun, options: Dict[str, Any] = None):
    """No JIT by default; return the function unchanged."""
    return fun


def inla_sn_reparam(moments: Optional[Sequence[float]] = None,
                    param: Optional[Sequence[float]] = None) -> Dict[str, float]:
    """
    Reparameterize skew-normal between (xi, omega, alpha) and (mean, variance, skewness).
    """
    if (moments is None) == (param is None):
        raise ValueError("Provide exactly one of 'moments' or 'param'.")
    if param is not None:
        xi, omega, alpha = float(param[0]), float(param[1]), float(param[2])
        delta = alpha / math.sqrt(1 + alpha * alpha)
        mean = xi + omega * delta * math.sqrt(2.0 / math.pi)
        variance = omega * omega * (1.0 - 2.0 * delta * delta / math.pi)
        skewness = ((4.0 - math.pi) / 2.0) * (delta * math.sqrt(2.0 / math.pi)) ** 3 / (1.0 - 2.0 * delta * delta / math.pi) ** 1.5
        return {"mean": mean, "variance": variance, "skewness": skewness}
    else:
        mean, variance, skewness = float(moments[0]), float(moments[1]), float(moments[2])
        delta = math.sqrt(math.pi / 2.0 * abs(skewness) ** (2.0 / 3.0) /
                          (abs(skewness) ** (2.0 / 3.0) + ((4.0 - math.pi) / 2.0) ** (2.0 / 3.0)))
        delta = math.copysign(delta, skewness)
        alpha = delta / math.sqrt(1.0 - delta * delta)
        omega = math.sqrt(variance / (1.0 - 2.0 * delta * delta / math.pi))
        xi = mean - omega * delta * math.sqrt(2.0 / math.pi)
        return {"xi": xi, "omega": omega, "alpha": alpha}


def inla_runjags2dataframe(*args, **kwargs):
    raise NotImplementedError("runjags is R-specific; no direct Python equivalent here.")


def inla_check_location(*args, **kwargs):
    # No-op in this Python port (depends on INLA model registry in R)
    return None


def inla_dynload_workaround():
    raise RuntimeError("This function is replaced by: inla.binary.install()")


def inla_matern_cf(dist: Union[float, np.ndarray], range: float = 1.0, nu: float = 0.5) -> np.ndarray:
    """
    Matérn correlation with 'range' parameterization used by SPDE models.
    """
    d = np.asarray(dist, dtype=float)
    kappa = math.sqrt(8.0 * nu) / range
    z = kappa * d
    res = np.zeros_like(z)
    is_zero = (d == 0.0)
    res[is_zero] = 1.0
    nonzero = ~is_zero
    zz = z[nonzero]
    if bessel_k is None:
        # crude fallback: asymptotic K_nu(z) ~ sqrt(pi/(2z)) e^{-z}
        K = np.sqrt(np.pi/(2*zz)) * np.exp(-zz)
    else:
        K = bessel_k(nu, zz)
    res[nonzero] = (1.0 / (2.0 ** (nu - 1.0) * math.gamma(nu))) * (zz ** nu) * K
    return res


def inla_sn_par(mean: np.ndarray, variance: np.ndarray, skew: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Return (xi, omega, alpha) parameters given moments, with skew clamped to 0.99.
    """
    skew = np.asarray(skew, dtype=float)
    skew_max = 0.99
    skew = np.nan_to_num(skew, nan=0.0)
    if np.any(np.abs(skew) > skew_max):
        warnings.warn(f"One or more abs(skewness) are too high. Coerced to be {skew_max}")
        skew = np.clip(skew, -skew_max, skew_max)

    mu = np.asarray(mean, dtype=float)
    var = np.asarray(variance, dtype=float)

    delta = np.sign(skew) * np.sqrt((math.pi / 2.0) * (np.abs(skew) ** (2.0 / 3.0)) /
                                    (((4.0 - math.pi) / 2.0) ** (2.0 / 3.0) + np.abs(skew) ** (2.0 / 3.0)))
    alpha = delta / np.sqrt(1.0 - delta ** 2)
    xi = mu - delta * np.sqrt((2.0 * var) / (math.pi - 2.0 * delta ** 2))
    omega = np.sqrt((math.pi * var) / (math.pi - 2.0 * delta ** 2))
    return {"xi": xi, "omega": omega, "alpha": alpha}


def inla_ensure_spd(A: np.ndarray, tol: float = math.sqrt(sys.float_info.epsilon)) -> np.ndarray:
    """Ensure SPD by flooring eigenvalues to tol * max_eig."""
    A = np.asarray(A, dtype=float)
    vals, vecs = np.linalg.eigh(A)
    floor = tol * float(np.max(vals))
    vals = np.maximum(vals, floor)
    return (vecs @ np.diag(vals) @ vecs.T)


def inla_read_state(filename: str) -> Dict[str, Any]:
    """
    Read a binary state file with layout:
      double fval; int32 nfun; int32 ntheta; double[ntheta] theta; int32 nx; double[nx] x
    """
    filename = os.path.abspath(filename)
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    with open(filename, "rb") as fp:
        fval = struct.unpack("=d", fp.read(8))[0]
        nfun = struct.unpack("=i", fp.read(4))[0]
        ntheta = struct.unpack("=i", fp.read(4))[0]
        theta = np.fromfile(fp, dtype=np.float64, count=ntheta) if ntheta > 0 else None
        nx = struct.unpack("=i", fp.read(4))[0]
        x = np.fromfile(fp, dtype=np.float64, count=nx) if nx > 0 else None
    return {"fval": fval, "nfun": nfun, "mode": {"theta": theta, "x": x}}


def inla_toeplitz(x: Sequence[float]) -> np.ndarray:
    """Circular Toeplitz matrix (matches the R helper behavior)."""
    x = np.asarray(x)
    n = x.size
    A = np.empty((n, n), dtype=x.dtype)
    for r in range(n):
        for c in range(n):
            A[r, c] = x[((c - r) + n) % n]
    return A


def inla_anyMultibyteUTF8Characters(string: str) -> bool:
    """True if any character in the string is multi-byte in UTF-8."""
    return any(len(ch.encode("utf-8")) > 1 for ch in string)


def inla_sparse_write_mtx(Q: Any, filename: str = "sparse.matrix.mtx") -> str:
    """
    Write a (symmetric) sparse matrix in MatrixMarket coordinate format (real symmetric).

    We mimic the R code's behavior:
    - ensure we have only one triangle + diagonal
    - write 1-based indices
    - header line + size line
    """
    _require_scipy()
    M = Q.tocoo(copy=True) if sp.issparse(Q) else sp.coo_matrix(Q)
    n = M.shape[0]
    diag_mask = M.row == M.col
    lower_mask = M.row > M.col
    if not np.any(lower_mask):
        M = M.T.tocoo()
        diag_mask = M.row == M.col
        lower_mask = M.row > M.col
    # 1-based indices
    i_eq = (M.row[diag_mask] + 1).astype(int)
    j_eq = (M.col[diag_mask] + 1).astype(int)
    x_eq = M.data[diag_mask]
    i_lo = (M.row[lower_mask] + 1).astype(int)
    j_lo = (M.col[lower_mask] + 1).astype(int)
    x_lo = M.data[lower_mask]
    nnz_half = i_eq.size + i_lo.size
    with open(filename, "w", encoding="utf-8") as fp:
        fp.write("%%MatrixMarket matrix coordinate real symmetric\n")
        fp.write(f"{n} {n} {nnz_half}\n")
        for ii, jj, vv in zip(i_eq, j_eq, x_eq):
            fp.write(f"{ii} {jj} {vv:.16g}\n")
        for ii, jj, vv in zip(i_lo, j_lo, x_lo):
            fp.write(f"{ii} {jj} {vv:.16g}\n")
    return filename


__all__ = [name for name in globals().keys() if name.startswith("inla_") or name in ("match_arg_vector", "pd_isna")]

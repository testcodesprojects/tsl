# section.py
# Python implementation of INLA "section.R" writers.
# This module aims to mirror structure and output intent of the R code.
# Assumes callers pass the same kind of structured arguments the R code expects.

from __future__ import annotations

import os
import re
import io
import sys
import math
import json
import time
import uuid
import shutil
import errno
import struct
import types
import random
import string
import pickle
import fnmatch
import platform
import inspect
import datetime as _dt
from typing import Any, Dict, List, Tuple, Optional, Sequence, Mapping, Union

from .control_defaults import control_inla as control_inla_default, control_vb as control_vb_default
from .models import INLAModels as _INLAModels
from .pc_bym import pc_bym_phi, pc_bym_Q
from .read_graph import inla_read_graph, InlaGraph

try:
    import numpy as np
except Exception as _e:
    raise RuntimeError("section.py requires numpy.") from _e

# SciPy is optional. If available, it improves handling of sparse matrices.
try:
    import scipy.sparse as sp
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

_MODELS_DB = _INLAModels()


# ======================================================================================
# Small compatibility layer for helpers used by the original R code
# ======================================================================================

def _writeln(filepath: str, text: str) -> None:
    """Append a single line to a file (creating it if needed)."""
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(text)

def inla_ifelse(cond: bool, a: Any, b: Any) -> Any:
    return a if bool(cond) else b

def inla_one_of(x: Any, choices: Sequence[Any]) -> bool:
    return any(x == c for c in choices)

def inla_paste(parts: Sequence[Any]) -> str:
    return "".join(str(p) for p in parts)

def inla_strcasecmp(a: Optional[str], b: Optional[str]) -> bool:
    if a is None or b is None:
        return False
    return str(a).strip().lower() == str(b).strip().lower()

def inla_trim(s: str) -> str:
    return re.sub(r"^\s+|\s+$", "", str(s))

def inla_graph_to_adjacency(graph: Any) -> np.ndarray:
    """Convert graph (file path, InlaGraph, or matrix) to adjacency matrix.

    Mirrors R-INLA's inla.graph2matrix function.
    """
    # If it's a string (file path), read the graph first
    if isinstance(graph, str):
        graph = inla_read_graph(graph)

    # If it's an InlaGraph object, convert to adjacency matrix
    if isinstance(graph, InlaGraph):
        n = graph.n
        # Build adjacency matrix from neighbor lists
        adj = np.zeros((n, n), dtype=float)
        for i in range(n):
            adj[i, i] = 1  # Diagonal (self-connection)
            for j in graph.nbs[i]:  # j is 1-based in InlaGraph
                adj[i, j - 1] = 1  # Convert to 0-based
                adj[j - 1, i] = 1  # Symmetric
        return adj

    # If it's already a matrix-like object
    if _HAVE_SCIPY and hasattr(graph, 'toarray'):
        return np.asarray(graph.toarray(), dtype=float)
    return np.asarray(graph, dtype=float)

def inla_trim_family(s: str) -> str:
    """Trim & normalize family/prior name the same way the R code would before writing."""
    return inla_trim(s)

def inla_namefix(s: str) -> str:
    """Make section names safe: drop spaces, replace non-alnum with '_' preserving ':' and '.'."""
    s = str(s)
    s = s.strip()
    s = re.sub(r"[ \t\r\n]+", "_", s)
    # Keep colon and dot; map everything else that is not word char/[:.] to '_'
    s = re.sub(r"[^\w:\.]+", "_", s)
    return s

def inla_text2vector(text: Union[str, Sequence[float], np.ndarray]) -> np.ndarray:
    """R: inla.text2vector; returns numeric vector."""
    if isinstance(text, str):
        # Split on space or comma or tab (one or more)
        parts = re.split(r"[ ,\t]+", text.strip())
        parts = [p for p in parts if p != ""]
        vec = [float(p) for p in parts]
        return np.asarray(vec, dtype=float)
    if isinstance(text, np.ndarray):
        return text.astype(float).ravel()
    try:
        return np.asarray(list(text), dtype=float).ravel()  # type: ignore[arg-type]
    except Exception:
        raise ValueError("inla_text2vector: unsupported input type.")

def inla_secsep(name: str | None = None) -> str:
    if not name:
        return "!"
    # R keeps the tag verbatim (even "(Intercept)")
    return f"!{name}!"

def inla_numlen(x: int) -> int:
    return max(1, int(math.ceil(math.log10(max(1, x+1)))))

def inla_num(i: Union[int, Sequence[int]], width: Optional[int] = None) -> Union[str, List[str]]:
    def fmt(x: int) -> str:
        if width is None:
            return str(x)
        return f"{x:0{width}d}"
    if isinstance(i, (list, tuple, np.ndarray)):
        return [fmt(int(v)) for v in i]  # type: ignore[list-item]
    return fmt(int(i))

def match_arg(x: Optional[str], choices: Sequence[str], several_ok: bool=False) -> str:
    """
    Simulate R's match.arg: case-insensitive match against choices; default to the first if x is None or 'default'.
    """
    if x is None:
        return choices[0]
    x_low = x.strip().lower()
    lowered = [c.lower() for c in choices]
    if x_low in lowered:
        return choices[lowered.index(x_low)]
    # Partial match (prefix)
    matches = [c for c in choices if c.lower().startswith(x_low)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1 and several_ok:
        return matches[0]
    # Fallback: allow "default"
    if x_low == "default":
        return choices[0]
    raise ValueError(f"match_arg: '{x}' not in {choices}")

def inla_getOption(name: str) -> Any:
    """Minimal default options used in this file."""
    defaults = {
        "scale.model.default": True,
        "smtp": "default",         # band, taucs, pardiso, stiles, default
        "numa": False,
        "internal.opt": True,
        "save.memory": False,
    }
    return defaults.get(name)

def inla_os_type() -> str:
    sysplat = sys.platform.lower()
    mach = platform.machine().lower()
    if sysplat.startswith("win"):
        return "windows"
    if sysplat.startswith("linux"):
        return "linux"
    if sysplat == "darwin":
        if mach in ("arm64", "aarch64"):
            return "mac.arm64"
        return "mac"
    return "unknown"

def inla_os_32or64bit() -> str:
    return "64" if sys.maxsize > 2**32 else "32"

def inla_version(which: str="version") -> str:
    # Provide a minimal version/date string
    if which == "version":
        return "py-inla-sections 0.1.0"
    if which == "date":
        return _dt.datetime.utcnow().strftime("%Y-%m-%d")
    return "unknown"

def inla_enabled_INLAjoint_features() -> bool:
    # Placeholder: joint features not enabled here.
    return False

def inla_reorderings_name2code(name: str) -> int:
    """Mapping names to codes; keep simple."""
    table = {
        "default": -1,
        "amd": 1,
        "metis": 2,
        "mmd": 3,
        "natural": 0,
    }
    low = name.strip().lower()
    return table.get(low, -1)

def inla_reorderings_code2name(code: int) -> str:
    rev = {
        -1: "default",
        0: "natural",
        1: "amd",
        2: "metis",
        3: "mmd",
    }
    return rev.get(int(code), "default")

def inla_set_control_fixed_default() -> Dict[str, Any]:
    return {
        "mean": 0.0,
        # Match R defaults seen in example outputs
        "prec": 1e-3,
        "mean.intercept": 0.0,
        "prec.intercept": 0.0,
    }

def inla_set_control_family_default() -> Dict[str, Any]:
    return {
        "cenpoisson.I": np.array([0, 0], dtype=int),
    }

def inla_set_control_inla_default() -> Dict[str, Any]:
    return {
        "control.vb": {
            "strategy": ["default", "diag", "full"],  # choices
            "hessian.strategy": ["default", "always", "periodic"],
        }
    }

def inla_set_control_compute_default() -> Dict[str, Any]:
    return {
        "control.gcpo": {
            "strategy": ["single", "joint"],
        }
    }

def _extract_link_model_value(link_val: Any) -> Optional[str]:
    """Extract the link model string from either string or dict format.

    Handles both formats:
    - String format: 'neglog' -> 'neglog'
    - Dict format: {'model': 'neglog'} -> 'neglog'

    Returns None if link_val is None or cannot be extracted.
    """
    if link_val is None:
        return None
    if isinstance(link_val, dict):
        # Extract 'model' key if present
        model_val = link_val.get("model")
        if model_val is not None:
            return str(model_val).strip()
        return None
    return str(link_val).strip()


def inla_model_validate_link_simple_function(family: str, link_simple: Optional[str]) -> Optional[str]:
    """Return normalized link.simple string if supported for the given family; otherwise None."""
    if link_simple is None:
        return None
    fam = (family or "").strip()
    candidate = str(link_simple).strip()
    if candidate == "":
        return None

    if fam:
        try:
            return _MODELS_DB.validate_link_simple_function(fam, candidate)
        except Exception:
            pass
    return candidate.lower()

def inla_model_validate_link_function(family: str, model: Any) -> str:
    """Return a normalized link model string, mirroring R-INLA defaults.

    Handles both string format ('neglog') and dict format ({'model': 'neglog'}).
    """
    fam = (family or "").strip()
    # Handle dict format: {'model': 'neglog'}
    extracted = _extract_link_model_value(model)
    candidate = "default" if extracted is None else extracted or "default"

    if fam:
        try:
            # INLAModels resolves "default" to the concrete link for the family.
            return _MODELS_DB.validate_link_function(fam, candidate).strip()
        except Exception:
            # Fall back to local heuristics when the registry lacks details.
            fam_lower = fam.lower()
            if inla_strcasecmp(candidate, "default"):
                if fam_lower == "poisson":
                    return "log"
                if fam_lower == "gaussian":
                    return "identity"
                if fam_lower == "binomial":
                    return "logit"
                if fam_lower in {"gamma", "exponential", "weibull"}:
                    return "log"
            return candidate.lower()

    if inla_strcasecmp(candidate, "default"):
        return "default"
    return candidate.lower()

def inla_model_properties(model: str, kind: str="latent", stop_on_error: bool=True) -> Dict[str, Any]:
    """
    Returns properties for a given latent model.
    Uses INLAModels to get full model definition including constr, nrow.ncol, etc.
    """
    from pyinla.models import INLAModels
    models = INLAModels()
    props = models.get_model_properties(model, kind, stop_on_error=stop_on_error)
    if props is None:
        # Fallback for unknown models
        nrow_ncol_models = {"matern2d", "matern2dx2part0", "matern2dx2p1", "rw2d", "rw2diid"}
        return {"nrow.ncol": (model in nrow_ncol_models)}
    # Normalize key names: 'nrow_ncol' -> 'nrow.ncol'
    result = dict(props)
    if 'nrow_ncol' in result:
        result['nrow.ncol'] = result.pop('nrow_ncol')
    return result

def inla_dir_create(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def inla_tempfile(tmpdir: Optional[str] = None, suffix: str = "") -> str:
    if tmpdir is None:
        tmpdir = os.getcwd()
    inla_dir_create(tmpdir)
    uid = uuid.uuid4().hex
    return os.path.join(tmpdir, f"inla_tmp_{int(time.time()*1e6)}_{uid}{suffix}")

def _ensure_matrix(x: Any) -> np.ndarray:
    """Coerce to a 2D numpy array of dtype float."""
    if x is None:
        raise ValueError("matrix is None")
    if _HAVE_SCIPY and sp.issparse(x):
        return x.toarray().astype(float)
    arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("Expected matrix-like (2D).")
    return arr.astype(float)

def _is_sparse(x: Any) -> bool:
    return _HAVE_SCIPY and sp.issparse(x)


def _ensure_sparse_or_matrix(x: Any):
    """
    Validate and return a matrix, preserving sparsity if input is sparse.

    Returns scipy.sparse matrix if input is sparse (and scipy available),
    otherwise returns a 2D numpy array.
    """
    if x is None:
        raise ValueError("matrix is None")
    if _HAVE_SCIPY and sp.issparse(x):
        # Preserve sparse, just ensure it's in a usable format
        if x.ndim != 2:
            raise ValueError("Expected matrix-like (2D).")
        return x.astype(float)
    arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("Expected matrix-like (2D).")
    return arr.astype(float)


def _sparse_diag(n: int, dtype=float):
    """Create a sparse identity matrix if scipy available, else dense."""
    if _HAVE_SCIPY:
        return sp.eye(n, dtype=dtype, format='csr')
    return np.eye(n, dtype=dtype)


def _sparse_zeros(m: int, n: int, dtype=float):
    """Create a sparse zero matrix if scipy available, else dense."""
    if _HAVE_SCIPY:
        return sp.csr_matrix((m, n), dtype=dtype)
    return np.zeros((m, n), dtype=dtype)


def _sparse_hstack(blocks):
    """Horizontally stack matrices, using sparse ops if any input is sparse."""
    if _HAVE_SCIPY and any(sp.issparse(b) for b in blocks):
        return sp.hstack(blocks, format='csr')
    return np.hstack(blocks)


def _sparse_vstack(blocks):
    """Vertically stack matrices, using sparse ops if any input is sparse."""
    if _HAVE_SCIPY and any(sp.issparse(b) for b in blocks):
        return sp.vstack(blocks, format='csr')
    return np.vstack(blocks)


def _sparse_block(block_list):
    """
    Build a block matrix from a list of lists, preserving sparsity.
    Similar to np.block but returns sparse if any block is sparse.
    """
    if not _HAVE_SCIPY:
        return np.block(block_list)

    # Check if any element is sparse
    has_sparse = False
    for row in block_list:
        for b in row:
            if sp.issparse(b):
                has_sparse = True
                break
        if has_sparse:
            break

    if has_sparse:
        rows = [sp.hstack(row, format='csr') for row in block_list]
        return sp.vstack(rows, format='csr')
    return np.block(block_list)

def inla_write_fmesher_file(obj: Any, filename: str, debug: bool=False) -> None:
    """Binary writer mirroring R's inla.write.fmesher.file for dense matrices."""
    version = 0
    if _HAVE_SCIPY and sp.issparse(obj):
        M = obj.tocoo()
        nrow, ncol = M.shape
        datatype = 1  # sparse
        valuetype = 0 if np.issubdtype(M.data.dtype, np.integer) else 1
        matrixtype = 0
        storagetype = 1
        elems = int(M.nnz)
        header = np.array([version, elems, nrow, ncol, datatype, valuetype, matrixtype, storagetype], dtype=np.int32)
        with open(filename, "wb") as fp:
            fp.write(np.int32([len(header)]).tobytes())
            fp.write(header.tobytes())
            fp.write((M.row.astype(np.int32)).tobytes(order="C"))
            fp.write((M.col.astype(np.int32)).tobytes(order="C"))
            if valuetype == 0:
                fp.write(M.data.astype(np.int32).tobytes(order="C"))
            else:
                fp.write(M.data.astype(np.float64).tobytes(order="C"))
        return

    arr = np.asarray(obj)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("inla_write_fmesher_file: only vectors/matrices are supported.")
    nrow, ncol = arr.shape
    datatype = 0  # dense
    valuetype = 0 if np.issubdtype(arr.dtype, np.integer) else 1
    matrixtype = 0
    storagetype = 1
    elems = nrow * ncol
    header = np.array([version, elems, nrow, ncol, datatype, valuetype, matrixtype, storagetype], dtype=np.int32)
    with open(filename, "wb") as fp:
        fp.write(np.int32([len(header)]).tobytes())
        fp.write(header.tobytes())
        if valuetype == 0:
            fp.write(arr.astype(np.int32).T.tobytes(order="C"))
        else:
            fp.write(arr.astype(np.float64).T.tobytes(order="C"))

def inla_write_graph(graph: Any, filename: Optional[str] = None) -> str:
    """
    Write a graph file in INLA format. Accepts:
      - str: path to existing .graph file (will be copied/returned as-is)
      - dict {node: [neighbors]} (1-indexed)
      - scipy sparse matrix (adjacency matrix, 0-indexed)
      - numpy 2D array (adjacency matrix, 0-indexed)
      - iterable of (u,v) tuples (edge list)
      - 2-column numpy array (edge list)

    INLA graph format:
      Line 1: n (number of nodes)
      Following lines: node_id num_neighbors neighbor1 neighbor2 ...
      (node IDs are 1-indexed in the file)

    Returns the filename.
    """
    if filename is None:
        filename = inla_tempfile()

    # If graph is a string or Path to an existing file, read and rewrite with standard format
    if isinstance(graph, (str, os.PathLike)):
        graph_path = str(graph)  # Convert Path to string if needed
        if os.path.isfile(graph_path):
            # Read the graph file and rewrite with R-INLA standard format (trailing spaces)
            with open(graph_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            with open(filename, "w", encoding="utf-8") as f:
                for line in lines:
                    line = line.strip()
                    if line:  # Skip empty lines
                        # R-INLA adds extra space for nodes with 0 neighbors
                        parts = line.split()
                        if len(parts) >= 2 and parts[1] == '0':
                            f.write(f"{line}  \n")  # Two trailing spaces for 0-neighbor nodes
                        else:
                            f.write(f"{line} \n")  # One trailing space otherwise
            return filename
        else:
            raise ValueError(f"Graph file not found: {graph_path}")

    # Check if it's a scipy sparse matrix
    if _HAVE_SCIPY and sp.issparse(graph):
        # Convert to adjacency list format
        # Sparse matrix is 0-indexed, INLA uses 1-indexed
        n = graph.shape[0]
        coo = graph.tocoo()
        adj_list: Dict[int, List[int]] = {i: [] for i in range(n)}
        for i, j, val in zip(coo.row, coo.col, coo.data):
            if val != 0 and i != j:  # Skip self-loops and zero entries
                adj_list[int(i)].append(int(j))

        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"{n} \n")  # R-INLA adds trailing space
            for node in range(n):
                neighbors = sorted(set(adj_list[node]))  # Remove duplicates, sort
                # Convert to 1-indexed for INLA
                neighbor_str = " ".join(str(nb + 1) for nb in neighbors)
                f.write(f"{node + 1} {len(neighbors)} {neighbor_str} \n")  # R-INLA adds trailing space
        return filename

    # Check if it's a dense numpy array (adjacency matrix)
    if isinstance(graph, np.ndarray) and graph.ndim == 2 and graph.shape[0] == graph.shape[1]:
        n = graph.shape[0]
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"{n} \n")  # R-INLA adds trailing space
            for node in range(n):
                neighbors = [j for j in range(n) if graph[node, j] != 0 and node != j]
                neighbor_str = " ".join(str(nb + 1) for nb in neighbors)
                f.write(f"{node + 1} {len(neighbors)} {neighbor_str} \n")  # R-INLA adds trailing space
        return filename

    # Handle dict format {node: [neighbors]} - assume already 1-indexed
    if isinstance(graph, dict):
        nodes = sorted(graph.keys())
        n = max(nodes) if nodes else 0
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"{n} \n")  # R-INLA adds trailing space
            for node in range(1, n + 1):
                neighbors = sorted(set(graph.get(node, [])))
                neighbor_str = " ".join(str(nb) for nb in neighbors)
                f.write(f"{node} {len(neighbors)} {neighbor_str} \n")  # R-INLA adds trailing space
        return filename

    # Handle edge list format
    edges: List[Tuple[int,int]] = []
    arr = np.asarray(list(graph))
    if arr.ndim == 1 and len(arr) == 2:
        edges.append((int(arr[0]), int(arr[1])))
    elif arr.ndim == 2 and arr.shape[1] == 2:
        for row in arr:
            edges.append((int(row[0]), int(row[1])))
    else:
        raise ValueError("Unsupported graph type for inla_write_graph.")

    # Convert edge list to adjacency list format
    if edges:
        all_nodes = set()
        for u, v in edges:
            all_nodes.add(u)
            all_nodes.add(v)
        n = max(all_nodes)
        adj_list_edges: Dict[int, List[int]] = {i: [] for i in range(1, n + 1)}
        for u, v in edges:
            adj_list_edges[u].append(v)

        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"{n} \n")  # R-INLA adds trailing space
            for node in range(1, n + 1):
                neighbors = sorted(set(adj_list_edges.get(node, [])))
                neighbor_str = " ".join(str(nb) for nb in neighbors)
                f.write(f"{node} {len(neighbors)} {neighbor_str} \n")  # R-INLA adds trailing space
    else:
        with open(filename, "w", encoding="utf-8") as f:
            f.write("0 \n")  # R-INLA adds trailing space

    return filename

def _np_to_int32(x: Sequence[int]) -> np.ndarray:
    return np.asarray(x, dtype=np.int32)

def _np_to_float64(x: Sequence[float]) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


# ======================================================================================
# 1) Parse B-matrix (coeff/var split)
# ======================================================================================

def inla_parse_Bmatrix(mat: Union[np.ndarray, Sequence[Sequence[str]]]) -> Dict[str, np.ndarray]:
    """
    Given a string matrix with entries like 'a*2', '2.3*x', '-2 * b', returns
    A: numeric coefficients matrix
    B: variable-name strings matrix (np.object_) or special token '<NO:VAR>'
    """
    NO_VAR = "<NO:VAR>"

    def remove_space(expr: str) -> str:
        return re.sub(r"[ \t]+", "", expr)

    def extract_coef(expr: str) -> float:
        a = remove_space(str(expr))
        if len(a) == 0:
            return 0.0
        parts = re.split(r"\*", a)
        if len(parts) == 0:
            return 1.0
        if len(parts) == 1:
            try:
                return float(parts[0])
            except Exception:
                return 1.0
        # two or more factors: return the numeric one among the first two tokens
        for i in range(min(2, len(parts))):
            try:
                return float(parts[i])
            except Exception:
                pass
        return 1.0

    def extract_var(expr: str) -> str:
        a = remove_space(str(expr))
        parts = re.split(r"\*", a)
        if len(parts) == 0:
            return NO_VAR
        if len(parts) == 1:
            try:
                float(parts[0])
                return NO_VAR
            except Exception:
                return parts[0]
        for i in range(min(2, len(parts))):
            try:
                float(parts[i])
            except Exception:
                return parts[i]
        return NO_VAR

    arr = np.asarray(mat, dtype=object)
    r, c = arr.shape if arr.ndim == 2 else (arr.size, 1)
    # Flatten, apply, then reshape
    flat = arr.ravel()
    A = np.array([extract_coef(x) for x in flat], dtype=float).reshape(r, c)
    B = np.array([extract_var(x) for x in flat], dtype=object).reshape(r, c)
    return {"A": A, "B": B}

def inla_parse_Bmatrix_test() -> Dict[str, np.ndarray]:
    mat = np.array([["a*2", "-3*b", ".2*x", "0.213 * d"],
                    ["e * -2.34", "   2.2 ", "x", ""]], dtype=object)
    return inla_parse_Bmatrix(mat)


# ======================================================================================
# 2) Simple primitives
# ======================================================================================

def inla_write_boolean_field(tag: str, val: Optional[bool], file: str) -> None:
    """Write 'tag = 1' or 'tag = 0' if val is not None."""
    if val is None:
        return
    _writeln(file, f"{tag} = {1 if bool(val) else 0}\n")


_CONTROL_INLA_KEY_ALIASES: Dict[str, str] = {
    "diff.log.dens": "diff_logdens",
    "print.joint.hyper": "print_joint_hyper",
    "force.diagonal": "force_diagonal",
    "skip.configurations": "skip_configurations",
    "adjust.weights": "adjust_weights",
    "lincomb.derived.correlation.matrix": "lincomb_derived_correlation_matrix",
    "numint.maxfeval": "numint_maxfeval",
    "numint.relerr": "numint_relerr",
    "numint.abserr": "numint_abserr",
    "b.strategy": "b_strategy",
    "step.len": "step_len",
    "step.factor": "step_factor",
    "global.node.factor": "global_node_factor",
    "global.node.degree": "global_node_degree",
    "stupid.search.max.iter": "stupid_search_max_iter",
    "stupid.search.factor": "stupid_search_factor",
    "num.gradient": "num_gradient",
    "num.hessian": "num_hessian",
    "optimise.strategy": "optimise_strategy",
    "use.directions": "use_directions",
    "constr.marginal.diagonal": "constr_marginal_diagonal",
    "improved.simplified.laplace": "improved_simplified_laplace",
    "parallel.linesearch": "parallel_linesearch",
    "compute.initial.values": "compute_initial_values",
    "hessian.correct.skewness.only": "hessian_correct_skewness_only",
    "control.vb": "control_vb",
    "int.strategy": "int_strategy",
    "int.design": "int_design",
    "diff.logdens": "diff_logdens",
    "print_joint_hyper": "print_joint_hyper",
    "adapt.hessian.mode": "adapt_hessian_mode",
    "adapt.hessian.max.trials": "adapt_hessian_max_trials",
    "adapt.hessian.scale": "adapt_hessian_scale",
    "cpo.diff": "cpo_diff",
}

_CONTROL_VB_KEY_ALIASES: Dict[str, str] = {
    "f.enable.limit": "f_enable_limit",
    "iter.max": "iter_max",
    "hessian.update": "hessian_update",
    "hessian.strategy": "hessian_strategy",
}


def _key_to_underscore(key: str, aliases: Dict[str, str]) -> str:
    if key in aliases:
        return aliases[key]
    if "." in key:
        return key.replace(".", "_")
    return key


def _normalize_control_vb_dict(spec: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if spec is None:
        return {}
    result: Dict[str, Any] = {}
    for k, v in spec.items():
        if k == "__control__":
            continue
        key_u = _key_to_underscore(k, _CONTROL_VB_KEY_ALIASES)
        result[key_u] = v
    return result


def _normalize_control_inla_dict(spec: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if spec is None:
        return {}
    result: Dict[str, Any] = {}
    for k, v in spec.items():
        if k == "__control__":
            continue
        key_u = _key_to_underscore(k, _CONTROL_INLA_KEY_ALIASES)
        if key_u == "control_vb" and isinstance(v, dict):
            result.setdefault("control_vb", {})
            result["control_vb"].update(_normalize_control_vb_dict(v))
        else:
            result[key_u] = v
    return result


def _format_number(val: Any) -> str:
    if isinstance(val, bool):
        return "1" if val else "0"
    if isinstance(val, int) or (isinstance(val, float) and val.is_integer()):
        return str(int(val))
    if isinstance(val, float):
        if math.isnan(val):
            return "NaN"
        if math.isinf(val):
            return "Inf" if val > 0 else "-Inf"
        text = f"{val:.5f}"
        if abs(val) >= 5e-6:
            text = text.rstrip("0").rstrip(".")
            if text == "-0":
                return "0"
            return text
        # Fall back to scientific-style formatting for very small numbers
        return format(val, ".6g")
    return str(val)


def _default_random_hyper(model: Optional[str]) -> List[Dict[str, Any]]:
    if model is None:
        return []
    model_lower = str(model).lower()
    if model_lower == "iid":
        return [{
            "initial": 4.0,
            "fixed": False,
            "hyperid": 1001,
            "prior": "loggamma",
            "param": [1.0, 5e-05],
            "to.theta": "function (x) <<NEWLINE>>log(x)",
            "from.theta": "function (x) <<NEWLINE>>exp(x)",
        }]
    if model_lower == "clinear":
        # clinear uses range-based transformation; R-INLA generates the functions dynamically
        # We use REPLACE.ME tokens that get substituted with actual range values
        return [{
            "initial": 1.0,
            "fixed": False,
            "hyperid": 37001,
            "prior": "normal",
            "param": [1.0, 10.0],
            "to.theta": (
                "function (x, REPLACE.ME.low, REPLACE.ME.high) <<NEWLINE>>"
                "{<<NEWLINE>>"
                "    if (all(is.infinite(c(low, high))) || low == high) {<<NEWLINE>>"
                "        stopifnot(low < high)<<NEWLINE>>"
                "        return(x)<<NEWLINE>>"
                "    }<<NEWLINE>>"
                "    else if (all(is.finite(c(low, high)))) {<<NEWLINE>>"
                "        stopifnot(low < high)<<NEWLINE>>"
                "        return(log(-(low - x)/(high - x)))<<NEWLINE>>"
                "    }<<NEWLINE>>"
                "    else if (is.finite(low) && is.infinite(high) && high > low) {<<NEWLINE>>"
                "        return(log(x - low))<<NEWLINE>>"
                "    }<<NEWLINE>>"
                "    else {<<NEWLINE>>"
                "        stop(\"Condition not yet implemented\")<<NEWLINE>>"
                "    }<<NEWLINE>>"
                "}"
            ),
            "from.theta": (
                "function (x, REPLACE.ME.low, REPLACE.ME.high) <<NEWLINE>>"
                "{<<NEWLINE>>"
                "    if (all(is.infinite(c(low, high))) || low == high) {<<NEWLINE>>"
                "        stopifnot(low < high)<<NEWLINE>>"
                "        return(x)<<NEWLINE>>"
                "    }<<NEWLINE>>"
                "    else if (all(is.finite(c(low, high)))) {<<NEWLINE>>"
                "        stopifnot(low < high)<<NEWLINE>>"
                "        return(low + exp(x)/(1 + exp(x)) * (high - low))<<NEWLINE>>"
                "    }<<NEWLINE>>"
                "    else if (is.finite(low) && is.infinite(high) && high > low) {<<NEWLINE>>"
                "        return(low + exp(x))<<NEWLINE>>"
                "    }<<NEWLINE>>"
                "    else {<<NEWLINE>>"
                "        stop(\"Condition not yet implemented\")<<NEWLINE>>"
                "    }<<NEWLINE>>"
                "}"
            ),
        }]
    return []

def inla_family_section(*args, **kwargs) -> None:
    """Wrapper: same as R — alias of inla_data_section."""
    inla_data_section(*args, **kwargs)


# ======================================================================================
# 3) Hyperparameters writer
# ======================================================================================

def _callable_to_r_function(fn: Any, label: Optional[str] = None) -> Optional[str]:
    """
    Attempt to convert a Python callable into the canonical R-style transform string,
    e.g. ``function (x) <<NEWLINE>>log(x)``. Falls back to None when we cannot
    reliably recover the expression.
    """
    if fn is None:
        return None
    if isinstance(fn, str):
        text = fn.strip()
        if text == "" or text.lower() == "none":
            return None
        return text
    if not callable(fn):
        text = str(fn).strip()
        if text.lower() == "none":
            return None
        return text

    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        src = ""

    expr: Optional[str] = None
    args_str: str = "x"  # Default to simple (x) signature
    if src:
        # Look for a lambda definition with full signature
        # Pattern captures: lambda <args>: <body>
        pattern = r"lambda\s+([^:]+):\s*(.*)"
        if label:
            label_pattern = re.escape(label) + r"['\"]?\s*:\s*(lambda\s+[^:]+:\s*.*)"
            m = re.search(label_pattern, src)
            if m:
                lambda_segment = m.group(1)
                m = re.search(pattern, lambda_segment)
        else:
            m = re.search(pattern, src)
        if m:
            args_str = m.group(1).strip()
            expr = m.group(2)
        else:
            # Look for a return statement inside a def
            returns = re.findall(r"return\s+(.*)", src)
            if returns:
                expr = returns[-1]
    if expr is None:
        return None

    # Drop trailing tokens that belong to subsequent dictionary keys
    expr = re.split(r",\s*(['\"]\w+['\"]|\})", expr)[0]
    expr = expr.strip()
    if expr.endswith(","):
        expr = expr[:-1].strip()

    if not expr:
        return None

    expr = expr.replace("np.", "").replace("math.", "")
    expr = re.sub(r"\s+", " ", expr).strip()
    expr = expr.rstrip(";")

    if expr.lower() == "none":
        return None

    # Convert Python underscore naming to R dot naming in both args and expr
    args_str = args_str.replace("_", ".")
    expr = expr.replace("_", ".")

    # Normalize single-variable function argument name to 'x' to match R-INLA
    # This handles cases like 'lambda z: exp(z)' -> 'function (x) exp(x)'
    simple_arg_match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)$', args_str.strip())
    if simple_arg_match:
        old_var = simple_arg_match.group(1)
        if old_var != 'x':
            # Replace the variable name in the expression with 'x'
            # Use word boundaries to avoid partial replacements
            expr = re.sub(r'\b' + re.escape(old_var) + r'\b', 'x', expr)
            args_str = 'x'

    # Normalize spacing around '=' in function arguments to match R-INLA style
    # Convert 'arg=val' to 'arg = val'
    args_str = re.sub(r"(\w+)\s*=\s*", r"\1 = ", args_str)

    # Normalize expression to match R-INLA's compact style
    # Remove spaces around '/' to produce 'a/b' instead of 'a / b'
    expr = re.sub(r"\s*/\s*", "/", expr)

    return f"function ({args_str}) <<NEWLINE>>{expr}"


def _func_to_source(fn: Any) -> str:
    """
    Convert a Python callable to a source-like string to embed in the .ini.
    If we cannot retrieve source, fall back to repr.
    """
    if isinstance(fn, str):
        return fn
    if not callable(fn):
        return str(fn)
    try:
        import inspect
        src = inspect.getsource(fn)
        # make single-line-ish (INI-friendly)
        src = src.replace("\r", "").replace("\n", "\\n")
        return src
    except Exception:
        return repr(fn)

def inla_write_hyper(hyper: Optional[List[Dict[str, Any]]],
                     file: str,
                     prefix: str = "",
                     data_dir: Optional[str] = None,
                     ngroup: int = -1,
                     low: float = float("-inf"),
                     high: float = float("inf")) -> List[Dict[str, Any]]:
    """
    Write hyperparameter blocks (list of dicts) to the INI.
    Each element k expects keys:
      - initial (float), fixed (bool), hyperid (int)
      - prior (string, or 'rprior:'-like; Python: keep it string)
      - param (sequence of floats)
      - to.theta / from.theta (callable or str) – written as stringified code
    Returns the (possibly modified) hyper list.
    """
    if not hyper:
        return []

    if data_dir is None:
        data_dir = os.getcwd()

    for k, hk in enumerate(hyper, start=1):
        suff = "" if len(hyper) == 1 else str(k - 1)
        initial_val = hk.get('initial')
        if initial_val is not None:
            _writeln(file, f"{prefix}initial{suff} = {_format_number(initial_val)}\n")
        fixed_val = hk.get('fixed')
        if fixed_val is not None:
            _writeln(file, f"{prefix}fixed{suff} = {_format_number(fixed_val)}\n")
        hyperid_val = hk.get('hyperid', 0)
        _writeln(file, f"{prefix}hyperid{suff} = {_format_number(hyperid_val)}\n")

        tmp_prior = str(hk.get("prior", "") or "")
        # Normalize whitespace and trailing semicolons like R
        tmp_prior = tmp_prior.replace("\n", "")
        tmp_prior = re.sub(r"^[ \t]+", "", tmp_prior)
        tmp_prior = re.sub(r";*[ \t]*$", "", tmp_prior)
        tmp_prior = re.sub(r"return[ \t]+\(", "return(", tmp_prior)

        prior_lower = tmp_prior.lower()
        if prior_lower == "pc.prec":
            tmp_prior = "pcprec"
            prior_lower = "pcprec"
        elif prior_lower == "pc.cor0":
            tmp_prior = "pccor0"
            prior_lower = "pccor0"
        elif prior_lower == "pc.cor1":
            tmp_prior = "pccor1"
            prior_lower = "pccor1"
        elif prior_lower == "pc.sn":
            tmp_prior = "pcsn"
            prior_lower = "pcsn"
        elif prior_lower == "pc.dof":
            tmp_prior = "pcdof"
            prior_lower = "pcdof"
        elif prior_lower == "pc.alphaw":
            tmp_prior = "pcalphaw"
            prior_lower = "pcalphaw"
        elif prior_lower == "pc.mgamma":
            tmp_prior = "pcmgamma"
            prior_lower = "pcmgamma"
        if tmp_prior.lower().startswith("table:"):
            tab = tmp_prior[len("table:"):].strip()
            # Check if this is already a file path reference (generated by bym2 handler)
            if tab.startswith("$inladatadir/") or tab.startswith(data_dir):
                # Already a file path - use it directly
                _writeln(file, f"{prefix}prior{suff} = table: {tab}\n")
            else:
                # Inline numbers - parse and write to file
                nums = [float(x) for x in re.split(r"[ \t\n\r]+", tab) if x.strip() != ""]
                nxy = len(nums) // 2
                xx = np.asarray(nums[:nxy], dtype=float)
                yy = np.asarray(nums[nxy:2*nxy], dtype=float)
                xy = np.column_stack([xx, yy])
                file_xy = inla_tempfile(tmpdir=data_dir)
                inla_write_fmesher_file(xy, filename=file_xy)
                fnm = file_xy.replace(data_dir, "$inladatadir")
                _writeln(file, f"{prefix}prior{suff} = table: {fnm}\n")
        else:
            # Normal prior string
            tmp_prior = inla_trim_family(tmp_prior)
            _writeln(file, f"{prefix}prior{suff} = {tmp_prior}\n")

        params = hk.get("param", [])
        # Normalize scalar param to list
        if params is not None and not isinstance(params, (list, tuple)):
            params = [params]
        if params:
            formatted_params = " ".join(_format_number(float(p)) for p in params)
        else:
            formatted_params = ""
        _writeln(file, f"{prefix}parameters{suff} = {formatted_params}\n")

        # handle PCGEVTAIL special case (keep parity in interface; no-op here)
        if prior_lower in ("pcgevtail", "pcegptail") and len(params) >= 3:
            low = float(params[1])
            high = float(params[2])

        # Embed the transform functions as "source"
        to_expr_raw = hk.get("to.theta")
        if to_expr_raw is None:
            to_expr = _callable_to_r_function(hk.get("to_theta"), label="to_theta")
        else:
            to_expr = _callable_to_r_function(to_expr_raw)

        from_expr_raw = hk.get("from.theta")
        if from_expr_raw is None:
            from_expr = _callable_to_r_function(hk.get("from_theta"), label="from_theta")
        else:
            from_expr = _callable_to_r_function(from_expr_raw)

        to_t = to_expr or ""
        from_t = from_expr or ""

        # Helper to format float for R (converts Python's inf to R's Inf, integers without .0)
        def _r_float(val):
            if math.isinf(val):
                return "Inf" if val > 0 else "-Inf"
            fval = float(val)
            # Output integers without decimal point to match R style
            if fval == int(fval):
                return str(int(fval))
            return str(fval)

        # Substitute tokens like R
        repl = {
            "REPLACE.ME.ngroup": f"ngroup={int(ngroup)}",
            "REPLACE.ME.low": f"low={_r_float(low)}",
            "REPLACE.ME.high": f"high={_r_float(high)}",
        }
        for krep, vrep in repl.items():
            to_t = to_t.replace(krep, vrep)
            from_t = from_t.replace(krep, vrep)

        if to_t.strip() != "":
            _writeln(file, f"{prefix}to.theta{suff} = {to_t}\n")
        if from_t.strip() != "":
            _writeln(file, f"{prefix}from.theta{suff} = {from_t}\n")

        # In-memory replacement as in R (best-effort):
        # Wrap original callables by binding the replaced defaults
        def _bind_low_high_ngroup(fn, _low=low, _high=high, _ng=ngroup):
            if not callable(fn):
                return fn
            def wrapped(*args, **kwargs):
                kwargs.setdefault("low", _low)
                kwargs.setdefault("high", _high)
                kwargs.setdefault("ngroup", _ng)
                return fn(*args, **kwargs)
            return wrapped

        hk["from.theta"] = _bind_low_high_ngroup(hk.get("from.theta"))
        hk["to.theta"] = _bind_low_high_ngroup(hk.get("to.theta"))

    return hyper


# ======================================================================================
# 4) Data / Family section
# ======================================================================================

def inla_data_section(file: str,
                      family: str,
                      file_data: str,
                      file_weights: str,
                      file_attr: str,
                      file_lp_scale: str,
                      control: Dict[str, Any],
                      i_family: str = "",
                      link_covariates: Optional[Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]] = None,
                      data_dir: Optional[str] = None) -> None:
    """
    Write the 'INLA.Data' section (aka 'family' section).
    """
    if data_dir is None:
        data_dir = os.getcwd()

    _writeln(file, f"{inla_secsep()}INLA.Data{i_family}{inla_secsep()}\n")
    _writeln(file, "type = data\n")
    _writeln(file, f"likelihood = {family}\n")
    _writeln(file, f"filename = {file_data}\n")
    _writeln(file, f"weights = {file_weights}\n")
    _writeln(file, f"attributes = {file_attr}\n")
    _writeln(file, f"lpscale = {file_lp_scale}\n")

    _writeln(file, "variant = " + str(int(control.get("variant", 0))) + "\n")

    if inla_one_of(family, ["cenpoisson", "zeroinflatedcenpoisson0", "zeroinflatedcenpoisson1"]):
        # Note: keys may be normalized (dots → underscores), so check both forms
        interval = control.get("cenpoisson.I") or control.get("cenpoisson_I") or inla_set_control_family_default()["cenpoisson.I"]
        interval = np.asarray(interval, dtype=int).ravel()
        if interval.size != 2:
            raise ValueError("cenpoisson.I must be length 2.")
        interval = np.sort(np.maximum(0, interval))
        if not np.isfinite(interval).all():
            raise ValueError("cenpoisson.I must be finite.")
        _writeln(file, f"cenpoisson.I = {int(interval[0])} {int(interval[1])}\n")

    if inla_one_of(family, ["beta"]):
        # Note: keys may be normalized (dots → underscores), so check both forms
        c_val = control.get("beta.censor.value") or control.get("beta_censor_value") or 0.0
        if c_val < 0 or c_val >= 0.5:
            raise ValueError("beta.censor.value must be in [0, 1/2).")
        _writeln(file, f"beta.censor.value = {float(c_val)}\n")

    # Quantile (now only via control.link)
    if control.get("quantile", None) is not None:
        raise ValueError("control.family(list(quantile=...)) is disabled; use control.link(list(quantile=...)).")
    # Note: keys may be normalized (dots → underscores), so check both forms
    ctrl_link_quantile = control.get("control.link") or control.get("control_link") or {}
    quantile = ctrl_link_quantile.get("quantile", None)
    q_val = -1
    if isinstance(quantile, (int, float)):
        if not (0.0 < float(quantile) < 1.0):
            raise ValueError("quantile must be numeric in (0,1).")
        q_val = float(quantile)
    _writeln(file, f"quantile = {q_val}\n")

    if inla_one_of(family, ["gev"]):
        # Note: keys may be normalized (dots → underscores), so check both forms
        v = control.get("gev.scale.xi") or control.get("gev_scale_xi") or 0.01
        _writeln(file, f"gev.scale.xi = {float(v)}\n")

    if inla_one_of(family, ["bgev"]):
        # Note: keys may be normalized (dots → underscores), so check both forms
        c_bgev = control.get("control.bgev") or control.get("control_bgev") or {}
        for k, v in c_bgev.items():
            vv = np.asarray(v).astype(float).ravel()
            _writeln(file, f"bgev.{k} = {' '.join(str(x) for x in vv)}\n")

    inla_write_hyper(control.get("hyper"), file=file, data_dir=data_dir)

    # link.simple (optional)
    # Note: keys may be normalized (dots → underscores), so check both forms
    link_simple = inla_model_validate_link_simple_function(family, control.get("link.simple") or control.get("link_simple"))
    if link_simple is not None:
        _writeln(file, f"link.simple = {link_simple}\n")

    # Backward-compat: control$link → control.link$model
    # Note: keys may be normalized (dots → underscores), so check both forms
    ctrl_link = dict(control.get("control.link") or control.get("control_link") or {})
    if "link" in control and (control["link"] is not None) and not inla_strcasecmp(control["link"], "default"):
        if "model" in ctrl_link and not inla_strcasecmp(ctrl_link["model"], "default"):
            raise ValueError("Both control.family$link (OBSOLETE) and control.family$control.link$model are set.")
        ctrl_link["model"] = control["link"]

    lmod = ctrl_link.get("model", None)
    lmod = inla_model_validate_link_function(family, lmod)
    ord_val = ctrl_link.get("order", None)
    _writeln(file, f"link.model = {lmod}\n")

    if inla_one_of(lmod, ["special1"]):
        if ord_val is None:
            raise ValueError(f"For link-model {lmod}, 'order' must be specified.")
        _writeln(file, f"link.order = {int(ord_val)}\n")
        # Special reshape for theta2 if present
        # (Python version leaves hyper re-shaping to the caller; no automatic rewrite here.)
    else:
        if ord_val is not None:
            raise ValueError(f"For link-model {lmod}, 'order' must be NULL/None.")

    variant = ctrl_link.get("variant", None)
    if inla_one_of(lmod, ["logoffset"]):
        if variant is None or int(variant) not in (0, 1):
            raise ValueError(f"For link-model {lmod}, 'variant' must be 0 or 1.")
        _writeln(file, f"link.variant = {int(variant)}\n")
    else:
        if variant is not None:
            raise ValueError(f"For link-model {lmod}, 'variant' is not used.")

    a_val = ctrl_link.get("a", None)
    if inla_one_of(lmod, ["loga"]):
        if (a_val is None) or (not isinstance(a_val, (int, float))) or not (0 < float(a_val) <= 1.0):
            raise ValueError(f"For link-model {lmod}, 'a' must be numeric and 0 < a <= 1.")
        _writeln(file, f"link.a = {float(a_val)}\n")

    inla_write_hyper(ctrl_link.get("hyper"), file=file, prefix="link.", data_dir=os.path.dirname(file))

    if link_covariates is not None:
        M = _ensure_matrix(link_covariates)
        file_link_cov = inla_tempfile(tmpdir=data_dir)
        inla_write_fmesher_file(M, filename=file_link_cov)
        file_link_cov = file_link_cov.replace(data_dir, "$inladatadir")
        _writeln(file, f"link.covariates = {file_link_cov}\n")

    # mix
    # Note: keys may be normalized (dots → underscores), so check both forms
    ctrl_mix = control.get("control.mix") or control.get("control_mix") or {}
    inla_write_boolean_field("mix.use", bool(ctrl_mix.get("model")) if ctrl_mix else False, file)
    if ctrl_mix and ctrl_mix.get("model"):
        _writeln(file, f"mix.model = {ctrl_mix['model']}\n")
        npoints = int(ctrl_mix.get("npoints", 5))
        if npoints < 5:
            raise ValueError("mix.npoints must be >= 5")
        _writeln(file, f"mix.npoints = {npoints}\n")
        integrator = match_arg(ctrl_mix.get("integrator", "default"),
                               ["default", "quadrature", "simpson"])
        _writeln(file, f"mix.integrator = {integrator}\n")
        inla_write_hyper(ctrl_mix.get("hyper"), file=file, prefix="mix.", data_dir=os.path.dirname(file))

    # pom
    if inla_one_of(family, ["pom"]):
        # Note: keys may be normalized (dots → underscores), so check both forms
        cpom = control.get("control.pom") or control.get("control_pom") or {}
        if cpom.get("cdf") is not None:
            if cpom["cdf"] not in ("logit", "probit"):
                raise ValueError("pom.cdf must be 'logit' or 'probit'")
            _writeln(file, f"pom.cdf = {cpom['cdf']}\n")
        if cpom.get("fast") is not None:
            inla_write_boolean_field("pom.fast.probit", bool(cpom["fast"]), file)

    # sem
    if inla_one_of(family, ["sem"]):
        # Note: keys may be normalized (dots → underscores), so check both forms
        ctrl_sem = control.get("control.sem") or control.get("control_sem") or {}
        if not ctrl_sem or "B" not in ctrl_sem:
            raise ValueError("control.sem$B must be provided for 'sem'.")
        AB = inla_parse_Bmatrix(ctrl_sem["B"])
        A = np.asarray(AB["A"])
        B = np.asarray(AB["B"], dtype=object)
        if A.shape[0] != A.shape[1] or B.shape[0] != B.shape[1] or A.shape != B.shape:
            raise ValueError("SEM: A and B must be square and same size.")
        file_B = inla_tempfile(tmpdir=data_dir)
        with open(file_B, "w", encoding="utf-8") as f:
            f.write(str(A.shape[0]) + "\n")
            # Write flattened A and B column-wise as in R code
            flatA = A.ravel(order="F")
            flatB = B.ravel(order="F")
            for a, b in zip(flatA, flatB):
                f.write(str(float(a)) + "\n")
                f.write(str(b) + "\n")
        fnm = file_B.replace(data_dir, "$inladatadir")
        _writeln(file, f"control.sem.b = {fnm}\n")

        idx = int(ctrl_sem.get("idx", 1))
        if idx <= 0 or idx > A.shape[0]:
            raise ValueError("control.sem$idx out of bounds.")
        _writeln(file, f"control.sem.idx = {idx-1}\n")  # C-index

    # cloglike
    if inla_one_of(family, ["cloglike"]):
        clog = control.get("cloglike")
        if clog is None or "cloglike" not in clog:
            raise ValueError("control$cloglike must be provided for 'cloglike'.")
        os_type = inla_os_type()
        if os_type == "windows":
            suffix = ".dll"
        elif os_type == "linux":
            suffix = ".so"
        elif os_type in ("mac", "mac.arm64"):
            suffix = ".dylib"
        else:
            raise RuntimeError("Unsupported platform for shared libraries.")
        shlib = inla_copy_file_for_section(clog["cloglike"]["shlib"], data_dir, suffix=suffix)
        _writeln(file, f"cloglike.shlib = {shlib}\n")
        _writeln(file, f"cloglike.model = {clog['cloglike']['model']}\n")
        inla_write_boolean_field("cloglike.debug", clog["cloglike"].get("debug"), file)

        # data block: ints/doubles/characters/matrices/smatrices
        data = clog["cloglike"]["data"]
        file_bin = inla_tempfile(tmpdir=data_dir)
        with open(file_bin, "wb") as fd:
            # ints
            ints = data.get("ints", {})
            fd.write(struct.pack("<i", len(ints)))
            for name, arr in ints.items():
                arr = _np_to_int32(arr)
                fd.write(struct.pack("<i", len(name)))
                fd.write(name.encode("utf-8"))
                fd.write(struct.pack("<i", len(arr)))
                fd.write(arr.astype(np.int32).tobytes(order="C"))
            # doubles
            doubles = data.get("doubles", {})
            fd.write(struct.pack("<i", len(doubles)))
            for name, arr in doubles.items():
                arr = _np_to_float64(arr)
                fd.write(struct.pack("<i", len(name)))
                fd.write(name.encode("utf-8"))
                fd.write(struct.pack("<i", len(arr)))
                fd.write(arr.astype(np.float64).tobytes(order="C"))
            # characters
            chars = data.get("characters", {})
            fd.write(struct.pack("<i", len(chars)))
            for name, s in chars.items():
                s = str(s)
                fd.write(struct.pack("<i", len(name)))
                fd.write(name.encode("utf-8"))
                fd.write(struct.pack("<i", len(s)))
                fd.write(s.encode("utf-8"))
            # matrices
            mats = data.get("matrices", {})
            fd.write(struct.pack("<i", len(mats)))
            for name, mat in mats.items():
                M = _ensure_matrix(mat)
                fd.write(struct.pack("<i", len(name)))
                fd.write(name.encode("utf-8"))
                fd.write(struct.pack("<i", int(M.shape[0])))
                fd.write(struct.pack("<i", int(M.shape[1])))
                fd.write(M.astype(np.float64).ravel(order="C").tobytes(order="C"))
            # smatrices (COO)
            smats = data.get("smatrices", {})
            fd.write(struct.pack("<i", len(smats)))
            for name, sm in smats.items():
                if _HAVE_SCIPY and sp.issparse(sm):
                    sm = sm.tocoo()
                    r, c = sm.shape
                    nnz = int(sm.nnz)
                    fd.write(struct.pack("<i", len(name)))
                    fd.write(name.encode("utf-8"))
                    fd.write(struct.pack("<i", r))
                    fd.write(struct.pack("<i", c))
                    fd.write(struct.pack("<i", nnz))
                    fd.write(_np_to_int32(sm.row + 1).tobytes(order="C"))
                    fd.write(_np_to_int32(sm.col + 1).tobytes(order="C"))
                    fd.write(sm.data.astype(np.float64).tobytes(order="C"))
                else:
                    raise ValueError("smatrices require scipy.sparse COO inputs.")
        file_bin2 = file_bin.replace(data_dir, "$inladatadir")
        _writeln(file, f"cloglike.data = {file_bin2}\n")

    _writeln(file, "\n")


# ======================================================================================
# 5) f-field (random effect / latent) section
# ======================================================================================

def _diag(n: int, val: float=1.0) -> np.ndarray:
    return np.eye(int(n), dtype=float) * float(val)

def _zeros(r: int, c: int) -> np.ndarray:
    return np.zeros((int(r), int(c)), dtype=float)

def inla_copy_file_for_section(filename: Optional[str], data_dir: str, suffix: str="") -> Optional[str]:
    if not filename:
        return None
    inla_dir_create(data_dir)
    new_f = inla_tempfile(tmpdir=data_dir, suffix=suffix)
    shutil.copy2(filename, new_f)
    return new_f.replace(data_dir, "$inladatadir")

def inla_copy_dir_for_section(dir_name: str, data_dir: str) -> str:
    new_dir = inla_tempfile(tmpdir=data_dir)
    inla_dir_create(new_dir)
    all_files = [os.path.join(dir_name, p) for p in os.listdir(dir_name)]
    for src in all_files:
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(new_dir, os.path.basename(src)), dirs_exist_ok=True)
        else:
            shutil.copy2(src, new_dir)
    return new_dir.replace(data_dir, "$inladatadir")

def inla_copy_dir_for_section_spde(prefix: str, data_dir: str) -> str:
    dir_name = os.path.dirname(prefix)
    file_prefix = os.path.basename(prefix)
    new_dir = inla_tempfile(tmpdir=data_dir)
    inla_dir_create(new_dir)
    files_to_copy = [os.path.join(dir_name, f) for f in os.listdir(dir_name)
                     if f.startswith(file_prefix)]
    for src in files_to_copy:
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(new_dir, os.path.basename(src)), dirs_exist_ok=True)
        else:
            shutil.copy2(src, new_dir)
    rdir = new_dir.replace(data_dir, "$inladatadir")
    # Add trailing dot so INLA finds files as prefix.B0, prefix.M0, etc.
    rprefix = f"{rdir}/{file_prefix}."
    # Remove originals created by f() (best effort)
    for src in files_to_copy:
        try:
            if os.path.isdir(src):
                shutil.rmtree(src, ignore_errors=True)
            else:
                os.remove(src)
        except Exception:
            pass
    return rprefix

def inla_ffield_section(file: str,
                        file_loc: Optional[str],
                        file_cov: str,
                        file_id_names: Optional[str],
                        n: int,
                        nrep: Optional[int],
                        ngroup: Optional[int],
                        file_extraconstr: Optional[str],
                        file_weights: Optional[str],
                        random_spec: Dict[str, Any],
                        results_dir: str,
                        only_hyperparam: bool,
                        data_dir: str) -> Dict[str, Any]:
    random_spec = dict(random_spec)

    # Get model properties early so we can use model-specific defaults
    prop = inla_model_properties(random_spec.get("model", ""), "latent")

    # Use model's constr default (e.g., True for rw1/rw2)
    random_spec.setdefault("constr", prop.get("constr", False))

    # BYM2 model requires constr=True (R-INLA does not support constr=FALSE for bym2)
    if inla_one_of(random_spec.get("model"), ["bym2"]) and not random_spec.get("constr", True):
        raise ValueError("The bym2 model requires constr=True. R-INLA does not support constr=FALSE for bym2.")

    # Set diagonal: 0.0001 for rw1/rw2/besag/etc. ONLY when constraint is on
    # When constr=False, diagonal should be 0.0 (matrix is full rank without constraint)
    if random_spec.get("diagonal") is None:
        if (inla_one_of(random_spec.get("model"), ["rw1", "rw2", "besag", "bym", "bym2", "besag2", "rw2d", "rw2diid", "seasonal"])
                and random_spec.get("constr", False)):
            random_spec["diagonal"] = 0.0001
        else:
            random_spec["diagonal"] = 0.0
    if not random_spec.get("hyper"):
        random_spec["hyper"] = _default_random_hyper(random_spec.get("model"))

    # Use 'id' for the section label
    label = random_spec.get("id", "")

    _writeln(file, f"{inla_secsep(label)}\n")
    _writeln(file, f"dir = {results_dir}\n")
    _writeln(file, "type = ffield\n")
    _writeln(file, f"model = {random_spec.get('model')}\n")
    if random_spec.get("same.as") is not None:
        _writeln(file, f"same.as = {random_spec['same.as']}\n")
    _writeln(file, f"covariates = {file_cov}\n")
    if file_id_names is not None:
        _writeln(file, f"id.names = {file_id_names}\n")
    if random_spec.get("diagonal") is not None:
        _writeln(file, f"diagonal = {random_spec['diagonal']}\n")
    inla_write_boolean_field("constraint", random_spec.get("constr"), file)
    inla_write_boolean_field("si", random_spec.get("si"), file)
    if file_extraconstr is not None:
        _writeln(file, f"extraconstraint = {file_extraconstr}\n")
    if random_spec.get("weights") is not None and file_weights is not None:
        _writeln(file, f"weights = {file_weights}\n")
    if random_spec.get("spde.prefix") is not None:
        fnm = inla_copy_dir_for_section_spde(random_spec["spde.prefix"], data_dir)
        _writeln(file, f"spde.prefix = {fnm}\n")
    if random_spec.get("spde2.prefix") is not None:
        fnm = inla_copy_dir_for_section_spde(random_spec["spde2.prefix"], data_dir)
        _writeln(file, f"spde2.prefix = {fnm}\n")
        if random_spec.get("spde2.transform") is not None:
            _writeln(file, f"spde2.transform = {random_spec['spde2.transform']}\n")
    if random_spec.get("spde3.prefix") is not None:
        fnm = inla_copy_dir_for_section_spde(random_spec["spde3.prefix"], data_dir)
        _writeln(file, f"spde3.prefix = {fnm}\n")
        if random_spec.get("spde3.transform") is not None:
            _writeln(file, f"spde3.transform = {random_spec['spde3.transform']}\n")
    if inla_one_of(random_spec.get("model"), ["copy", "scopy"]):
        if random_spec.get("of") is not None:
            _writeln(file, f"of = {random_spec['of']}\n")
    if inla_one_of(random_spec.get("model"), ["copy", "scopy", "sigm", "revsigm", "log1exp", "fgn", "intslope"]):
        if random_spec.get("precision") is not None:
            _writeln(file, f"precision = {random_spec['precision']}\n")

    if inla_one_of(random_spec.get("model"), ["clinear", "copy", "mec", "meb"]):
        rng = random_spec.get("range")
        if rng is None:
            rng = (0, 0)
        # Format values for R (convert inf to Inf, integers without .0)
        def _r_float_val(v):
            v = float(v)
            if math.isinf(v):
                return "Inf" if v > 0 else "-Inf"
            # Output integers without decimal point to match R style
            if v == int(v):
                return str(int(v))
            return str(v)
        _writeln(file, f"range.low  = {_r_float_val(rng[0])}\n")
        _writeln(file, f"range.high = {_r_float_val(rng[1])}\n")

    if inla_one_of(random_spec.get("model"), ["rw1", "rw2", "besag", "bym", "bym2", "besag2", "rw2d", "rw2diid", "seasonal"]):
        # bym2 defaults to scale.model=True, others default to False
        default_scale = True if inla_one_of(random_spec.get("model"), ["bym2"]) else False
        sc = random_spec.get("scale.model", default_scale)
        _writeln(file, f"scale.model = {1 if bool(sc) else 0}\n")

    if inla_one_of(random_spec.get("model"), ["besag", "bym", "bym2", "besag2"]):
        # For BYM2, R-INLA REQUIRES adjust.for.con.comp=TRUE (not optional)
        # For other models, default to True to match R-INLA behavior
        if inla_one_of(random_spec.get("model"), ["bym2"]):
            adj = True  # R-INLA requires TRUE for BYM2
        else:
            adj = random_spec.get("adjust.for.con.comp", True)
        _writeln(file, f"adjust.for.con.comp = {1 if bool(adj) else 0}\n")

    # Handle adaptive PC prior for bym2 phi parameter
    if inla_one_of(random_spec.get("model"), ["bym2"]):
        hyper_list = random_spec.get("hyper", [])
        graph = random_spec.get("graph")

        if hyper_list and len(hyper_list) >= 2 and graph is not None:
            phi_hyper = hyper_list[1]  # theta2 is the phi parameter
            phi_prior = str(phi_hyper.get("prior", "")).lower()
            # Generate table if prior is "pc" (the default adaptive prior)
            # R-INLA generates the table even when phi is fixed
            if phi_prior == "pc":
                # Get prior parameters (u, alpha) - defaults from R-INLA
                phi_param = phi_hyper.get("param", [0.5, 0.5])
                u_val = float(phi_param[0]) if len(phi_param) > 0 else 0.5
                alpha_val = float(phi_param[1]) if len(phi_param) > 1 else 0.5
                # Generate the PC prior table for phi based on the graph
                try:
                    # Convert graph to adjacency matrix (handles file paths, InlaGraph, or matrices)
                    graph_arr = inla_graph_to_adjacency(graph)
                    table_result = pc_bym_phi(
                        graph=graph_arr,
                        alpha=alpha_val,
                        u=u_val,
                        return_as_table=True,
                        adjust_for_con_comp=True  # R-INLA requires TRUE for BYM2
                    )
                    # Convert to table: format - write theta and log prior values
                    theta_grid = table_result["theta"]
                    logprior = table_result["logprior"]
                    # INLA expects (theta, log_prior) pairs (log density, not raw density)
                    # Create table string in the format INLA expects
                    table_data = np.column_stack([theta_grid, logprior])
                    file_table = inla_tempfile(tmpdir=data_dir)
                    inla_write_fmesher_file(table_data, filename=file_table)
                    fnm_table = file_table.replace(data_dir, "$inladatadir")
                    # Update the phi hyper to use table prior
                    phi_hyper["prior"] = f"table: {fnm_table}"
                    phi_hyper["param"] = []  # No params needed for table prior
                except Exception as e:
                    # Fall back to "none" if table generation fails (matches R-INLA behavior)
                    phi_hyper["prior"] = "none"
                    phi_hyper["param"] = []
                    import warnings
                    warnings.warn(f"Could not generate PC prior table for bym2 phi: {e}. Using prior='none'.")

    rng = random_spec.get("range")
    low = float(rng[0]) if rng is not None else float("-inf")
    high = float(rng[1]) if rng is not None else float("inf")
    random_spec["hyper"] = inla_write_hyper(random_spec.get("hyper"), file=file, data_dir=data_dir,
                                            ngroup=(ngroup if ngroup is not None else -1), low=low, high=high)

    if inla_model_properties(random_spec.get("model"), "latent").get("nrow.ncol", False):
        _writeln(file, f"nrow = {int(random_spec.get('nrow'))}\n")
        _writeln(file, f"ncol = {int(random_spec.get('ncol'))}\n")
        if random_spec.get("bvalue") is not None:
            _writeln(file, f"bvalue = {random_spec['bvalue']}\n")
        if inla_one_of(random_spec.get("model"), ["matern2d", "matern2dx2part0", "matern2dx2p1"]):
            if random_spec.get("nu") is not None:
                _writeln(file, f"nu = {random_spec['nu']}\n")
    else:
        _writeln(file, f"n = {int(n)}\n")

    _writeln(file, f"nrep = {int(1 if nrep is None else nrep)}\n")

    if ngroup is not None and ngroup > 1:
        _writeln(file, f"ngroup = {int(ngroup)}\n")
        if not inla_one_of(random_spec.get("model"), ["copy"]):
            # Ensure control.group exists in random_spec
            if random_spec.get("control.group") is None:
                random_spec["control.group"] = {}
            cg = random_spec.get("control.group") or {}
            group_model = cg.get("model", "iid")  # Default to 'iid' like R-INLA
            _writeln(file, f"group.model = {group_model}\n")
            # Write group.cyclic (default False)
            group_cyclic = cg.get("cyclic", False)
            _writeln(file, f"group.cyclic = {1 if group_cyclic else 0}\n")
            # Write group.scale.model (default from option)
            scg = cg.get("scale.model", inla_getOption("scale.model.default"))
            _writeln(file, f"group.scale.model = {1 if scg else 0}\n")
            # Write group.adjust.for.con.comp (default True)
            group_adj = cg.get("adjust.for.con.comp", True)
            _writeln(file, f"group.adjust.for.con.comp = {1 if group_adj else 0}\n")
            if inla_one_of(cg.get("model"), ["ar"]):
                p = int(cg.get("order", 0))
                _writeln(file, f"group.order = {p}\n")
            if inla_one_of(cg.get("model"), ["besag"]):
                g = cg.get("graph")
                if g is None:
                    raise ValueError("control.group$graph must be provided for group besag.")
                gfile = inla_write_graph(g, filename=inla_tempfile())
                fnm = inla_copy_file_for_section(gfile, data_dir)
                try:
                    os.remove(gfile)
                except Exception:
                    pass
                _writeln(file, f"group.graph = {fnm}\n")
            else:
                if cg.get("graph") is not None:
                    raise ValueError("group.graph must be NULL unless model is 'besag'.")

            # Get group hyperparameters - use defaults from group model if not specified
            group_hyper = cg.get("hyper")
            if not group_hyper:
                # Get default hyperparameters for the group model
                group_model_props = inla_model_properties(group_model, "group", stop_on_error=False)
                if group_model_props and group_model_props.get("hyper"):
                    hyper_dict = group_model_props.get("hyper")
                    # Convert from dict format {'theta': {...}, 'theta2': {...}} to list format [{...}, {...}]
                    if isinstance(hyper_dict, dict):
                        # Check if values are dicts (expected) or something else
                        values = list(hyper_dict.values())
                        if values and isinstance(values[0], dict):
                            group_hyper = values
                        else:
                            # Unexpected format, skip
                            group_hyper = None
                    else:
                        group_hyper = hyper_dict

            if group_hyper:
                # Ensure it's a list of dicts
                if not isinstance(group_hyper, list):
                    group_hyper = [group_hyper] if isinstance(group_hyper, dict) else []
                # Filter out any non-dict items
                group_hyper = [h for h in group_hyper if isinstance(h, dict)]
                if group_hyper:
                    random_spec["control.group"]["hyper"] = inla_write_hyper(group_hyper, file=file, prefix="group.", data_dir=data_dir, ngroup=ngroup)
                else:
                    random_spec["control.group"]["hyper"] = []
            else:
                random_spec["control.group"]["hyper"] = []

    # scopy
    if inla_one_of(random_spec.get("model"), ["scopy"]):
        ctrl = random_spec.get("control.scopy", {})
        _writeln(file, f"scopy.n = {int(ctrl.get('n'))}\n")
        # covariate vector
        z = np.asarray(ctrl.get("covariate"), dtype=float).ravel()
        file_z = inla_tempfile(tmpdir=data_dir)
        inla_write_fmesher_file(z.reshape(-1, 1), filename=file_z)
        _writeln(file, f"scopy.covariate = {file_z.replace(data_dir, '$inladatadir')}\n")
        # W definition (simple identity fallback)
        nW = int(ctrl.get("n"))
        W = _diag(nW, 1.0)
        file_W = inla_tempfile(tmpdir=data_dir)
        inla_write_fmesher_file(W, filename=file_W)
        _writeln(file, f"scopy.W = {file_W.replace(data_dir, '$inladatadir')}\n")

    if random_spec.get("cyclic") is not None:
        _writeln(file, f"cyclic = {1 if random_spec['cyclic'] else 0}\n")
    if random_spec.get("season.length") is not None:
        _writeln(file, f"season = {random_spec['season.length']}\n")
    if random_spec.get("graph") is not None:
        gfile = inla_write_graph(random_spec["graph"], filename=inla_tempfile())
        fnm = inla_copy_file_for_section(gfile, data_dir)
        try: os.remove(gfile)
        except Exception: pass
        _writeln(file, f"graph = {fnm}\n")

    rankdef_value = random_spec.get("rankdef")
    if rankdef_value is not None:
        _writeln(file, f"rankdef = {int(rankdef_value)}\n")
    elif inla_one_of(random_spec.get("model"), ["besag", "bym2"]) and random_spec.get("graph") is not None:
        _writeln(file, "rankdef = 1\n")

    if file_loc is not None:
        _writeln(file, f"locations = {file_loc}\n")

    # z-model (sparse-aware)
    if inla_one_of(random_spec.get("model"), ["z"]):
        Z = _ensure_sparse_or_matrix(random_spec["Z"])
        Z_n, Z_m = Z.shape
        prec = float(random_spec.get("precision", 1e8))  # R-INLA default is 1e8

        # Check if we should use sparse operations
        use_sparse = _is_sparse(Z)

        if use_sparse:
            # Sparse path: build A and B using scipy.sparse operations
            tZ = Z.T.tocsr()
            I_n = _sparse_diag(Z_n)
            tZZ = tZ @ Z  # sparse @ sparse = sparse

            # A = prec * [[I_n, -Z], [-Z^T, Z^T @ Z]]
            top = sp.hstack([I_n, -Z], format='csr')
            bot = sp.hstack([-tZ, tZZ], format='csr')
            A = prec * sp.vstack([top, bot], format='csr')

            # C matrix (Cmatrix or identity)
            Cmat = random_spec.get("Cmatrix", None)
            if Cmat is None:
                Cm = _sparse_diag(Z_m)
            else:
                Cm = _ensure_sparse_or_matrix(Cmat)

            # B = [[0, 0], [0, Cm]]
            Z_n_n = _sparse_zeros(Z_n, Z_n)
            Z_n_m = _sparse_zeros(Z_n, Z_m)
            Z_m_n = _sparse_zeros(Z_m, Z_n)
            B = sp.vstack([
                sp.hstack([Z_n_n, Z_n_m], format='csr'),
                sp.hstack([Z_m_n, Cm], format='csr')
            ], format='csr')
        else:
            # Dense path: original implementation
            tZ = Z.T
            # A matrix: [[I_n, -Z], [-Z^T, Z^T @ Z]]
            top = np.hstack([_diag(Z_n), -Z])
            bot = np.hstack([-tZ, tZ @ Z])
            A = prec * np.vstack([top, bot])
            # C matrix
            Cmat = random_spec.get("Cmatrix", None)
            if Cmat is None:
                Cm = _diag(Z_m)
            else:
                Cm = _ensure_matrix(Cmat)
            B = np.vstack([
                np.hstack([_zeros(Z_n, Z_n), _zeros(Z_n, Z_m)]),
                np.hstack([_zeros(Z_m, Z_n), Cm])
            ])

        # Write dims
        _writeln(file, f"z.n = {Z_n}\n")
        _writeln(file, f"z.m = {Z_m}\n")
        # Write A (always sparse format for z-model - INLA expects sparse)
        if not _is_sparse(A):
            A = sp.csr_matrix(A)
        fA = inla_tempfile(tmpdir=data_dir)
        inla_write_fmesher_file(A, filename=fA)
        _writeln(file, f"z.Amatrix = {fA.replace(data_dir, '$inladatadir')}\n")
        # Write B (always sparse format for z-model - INLA expects sparse)
        if not _is_sparse(B):
            B = sp.csr_matrix(B)
        fB = inla_tempfile(tmpdir=data_dir)
        inla_write_fmesher_file(B, filename=fB)
        _writeln(file, f"z.Bmatrix = {fB.replace(data_dir, '$inladatadir')}\n")

    if inla_one_of(random_spec.get("model"), ["dmatern"]):
        file_l = inla_tempfile(tmpdir=data_dir)
        inla_write_fmesher_file(_ensure_matrix(random_spec["locations"]), filename=file_l)
        _writeln(file, f"dmatern.locations = {file_l.replace(data_dir, '$inladatadir')}\n")

    if inla_one_of(random_spec.get("model"), ["generic3"]):
        C = random_spec.get("Cmatrix")
        if not isinstance(C, (list, tuple)) or len(C) == 0:
            raise ValueError("generic3: Cmatrix must be a non-empty list")
        _writeln(file, f"generic3.n = {int(random_spec.get('n'))}\n")
        _writeln(file, f"generic3.m = {len(C)}\n")
        for k, M in enumerate(C, start=1):
            fA = inla_tempfile(tmpdir=data_dir)
            # Preserve sparsity - fmesher writer handles both sparse and dense
            inla_write_fmesher_file(_ensure_sparse_or_matrix(M), filename=fA)
            _writeln(file, f"generic3.Cmatrix.{k-1} = {fA.replace(data_dir, '$inladatadir')}\n")

    # Write Cmatrix generally if provided (z and generic3 handled above)
    # This handles generic0, generic1, generic2, and other models with Cmatrix
    if (not inla_one_of(random_spec.get("model"), ["z", "generic3"])) and (random_spec.get("Cmatrix") is not None):
        Cx = random_spec["Cmatrix"]
        if isinstance(Cx, str):
            fnm = inla_copy_file_for_section(Cx, data_dir)
            _writeln(file, f"Cmatrix = {fnm}\n")
        else:
            fC = inla_tempfile(tmpdir=data_dir)
            # Preserve sparsity - fmesher writer handles both sparse and dense
            inla_write_fmesher_file(_ensure_sparse_or_matrix(Cx), filename=fC)
            _writeln(file, f"Cmatrix = {fC.replace(data_dir, '$inladatadir')}\n")

    # slm (sparse-aware)
    if inla_one_of(random_spec.get("model"), ["slm"]):
        X = _ensure_sparse_or_matrix(random_spec["args.slm"]["X"])
        W = _ensure_sparse_or_matrix(random_spec["args.slm"]["W"])
        Qb = _ensure_sparse_or_matrix(random_spec["args.slm"]["Q.beta"])
        slm_n, slm_m = X.shape
        _writeln(file, f"slm.n = {slm_n}\n")
        _writeln(file, f"slm.m = {slm_m}\n")
        _writeln(file, f"slm.rho.min = {random_spec['args.slm']['rho.min']}\n")
        _writeln(file, f"slm.rho.max = {random_spec['args.slm']['rho.max']}\n")

        use_sparse = _is_sparse(X) or _is_sparse(W) or _is_sparse(Qb)

        if use_sparse:
            # A1 = [[I_n, -X^T], [-X, X^T @ X]]
            I_n = _sparse_diag(slm_n)
            XtX = X.T @ X
            A1 = _sparse_block([[I_n, -X.T], [-X, XtX]])
            # A2 = [[0, 0], [0, Qb]]
            A2 = _sparse_block([[_sparse_zeros(slm_n, slm_n), _sparse_zeros(slm_n, slm_m)],
                                [_sparse_zeros(slm_m, slm_n), Qb]])
            # B = [[-(W^T + W), X^T @ W], [W^T @ X, 0]]
            B = _sparse_block([[-(W.T + W), X.T @ W],
                               [W.T @ X, _sparse_zeros(slm_m, slm_m)]])
            # C = [[W^T @ W, 0], [0, 0]]
            C = _sparse_block([[W.T @ W, _sparse_zeros(slm_n, slm_m)],
                               [_sparse_zeros(slm_m, slm_n), _sparse_zeros(slm_m, slm_m)]])
        else:
            # Dense path
            A1 = np.block([[ _diag(slm_n), -X.T ],
                           [ -X,           X.T @ X ]])
            A2 = np.block([[ _zeros(slm_n, slm_n), _zeros(slm_n, slm_m)],
                           [ _zeros(slm_m, slm_n), Qb ]])
            B = np.block([[ -(W.T + W),    X.T @ W ],
                          [ W.T @ X,       _zeros(slm_m, slm_m)]])
            C = np.block([[ W.T @ W,         _zeros(slm_n, slm_m)],
                          [ _zeros(slm_m, slm_n), _zeros(slm_m, slm_m)]])

        fA1 = inla_tempfile(tmpdir=data_dir)
        inla_write_fmesher_file(A1, filename=fA1)
        _writeln(file, f"slm.A1matrix = {fA1.replace(data_dir, '$inladatadir')}\n")
        fA2 = inla_tempfile(tmpdir=data_dir)
        inla_write_fmesher_file(A2, filename=fA2)
        _writeln(file, f"slm.A2matrix = {fA2.replace(data_dir, '$inladatadir')}\n")
        fB = inla_tempfile(tmpdir=data_dir)
        inla_write_fmesher_file(B, filename=fB)
        _writeln(file, f"slm.Bmatrix = {fB.replace(data_dir, '$inladatadir')}\n")
        fC = inla_tempfile(tmpdir=data_dir)
        inla_write_fmesher_file(C, filename=fC)
        _writeln(file, f"slm.Cmatrix = {fC.replace(data_dir, '$inladatadir')}\n")

    # ar1c (sparse-aware)
    if inla_one_of(random_spec.get("model"), ["ar1c"]):
        Z = _ensure_sparse_or_matrix(random_spec["args.ar1c"]["Z"])
        Qbeta = _ensure_sparse_or_matrix(random_spec["args.ar1c"]["Q.beta"])
        ar1c_n, ar1c_m = Z.shape
        _writeln(file, f"ar1c.n = {ar1c_n}\n")
        _writeln(file, f"ar1c.m = {ar1c_m}\n")
        # Z (last row not used in R; we keep as-is)
        fZ = inla_tempfile(tmpdir=data_dir)
        inla_write_fmesher_file(Z, filename=fZ)
        _writeln(file, f"ar1c.Z = {fZ.replace(data_dir, '$inladatadir')}\n")
        # ZZ = Z^T @ Z (sparse @ sparse = sparse automatically)
        ZZ = Z.T @ Z
        fZZ = inla_tempfile(tmpdir=data_dir)
        inla_write_fmesher_file(ZZ, filename=fZZ)
        _writeln(file, f"ar1c.ZZ = {fZZ.replace(data_dir, '$inladatadir')}\n")
        # Qbeta
        fQ = inla_tempfile(tmpdir=data_dir)
        inla_write_fmesher_file(Qbeta, filename=fQ)
        _writeln(file, f"ar1c.Qbeta = {fQ.replace(data_dir, '$inladatadir')}\n")

    # compute flag
    if only_hyperparam or (not bool(random_spec.get("compute", True))):
        _writeln(file, "compute = 0\n")
    else:
        _writeln(file, "compute = 1\n")

    # scale (mec/meb/iid)
    if inla_one_of(random_spec.get("model"), ["mec", "meb", "iid"]):
        if random_spec.get("scale") is not None:
            sc = np.asarray(random_spec["scale"], dtype=float).ravel()
            idxs = np.arange(1, sc.size+1, dtype=int)
            M = np.column_stack([idxs-1, sc])
            fS = inla_tempfile(tmpdir=data_dir)
            inla_write_fmesher_file(M, filename=fS)
            _writeln(file, f"scale = {fS.replace(data_dir, '$inladatadir')}\n")
    else:
        if random_spec.get("scale") is not None:
            raise ValueError(f"Section [{label}]: option 'scale' is not used for model={random_spec.get('model')}.")

    # rgeneric (store Python object via pickle, keep name)
    if random_spec.get("model") == "rgeneric":
        file_rgen = inla_tempfile(tmpdir=data_dir)
        model_obj = random_spec["rgeneric"]["model"]
        with open(file_rgen, "wb") as fbin:
            pickle.dump(model_obj, fbin)
        fnm = file_rgen.replace(data_dir, "$inladatadir")
        model_name = f".inla.rgeneric.model.{random_spec['rgeneric'].get('Id','X')}"
        _writeln(file, f"rgeneric.file = {fnm}\n")
        _writeln(file, f"rgeneric.model = {model_name}\n")

    # cgeneric (shared lib + data)
    if random_spec.get("model") == "cgeneric":
        os_type = inla_os_type()
        if os_type == "windows":
            suffix = ".dll"
        elif os_type == "linux":
            suffix = ".so"
        elif os_type in ("mac", "mac.arm64"):
            suffix = ".dylib"
        else:
            raise RuntimeError("Unsupported platform for shared libraries.")
        model = random_spec["cgeneric"]["model"]
        shlib = inla_copy_file_for_section(model["shlib"], data_dir, suffix=suffix)
        _writeln(file, f"cgeneric.shlib = {shlib}\n")
        _writeln(file, f"cgeneric.model = {model['model']}\n")
        _writeln(file, f"cgeneric.n = {int(model['n'])}\n")
        inla_write_boolean_field("cgeneric.debug", model.get("debug"), file)
        inla_write_boolean_field("cgeneric.q", model.get(".q"), file)
        if model.get(".q"):
            _writeln(file, f"cgeneric.q.file = {model.get('.q.file')}\n")

        data = model.get("data", {})
        file_bin = inla_tempfile(tmpdir=data_dir)
        with open(file_bin, "wb") as fd:
            # ints
            ints = data.get("ints", {})
            fd.write(struct.pack("<i", len(ints)))
            for name, arr in ints.items():
                arr = _np_to_int32(arr)
                fd.write(struct.pack("<i", len(name)))
                fd.write(name.encode("utf-8"))
                fd.write(struct.pack("<i", len(arr)))
                fd.write(arr.astype(np.int32).tobytes(order="C"))
            # doubles
            doubles = data.get("doubles", {})
            fd.write(struct.pack("<i", len(doubles)))
            for name, arr in doubles.items():
                arr = _np_to_float64(arr)
                fd.write(struct.pack("<i", len(name)))
                fd.write(name.encode("utf-8"))
                fd.write(struct.pack("<i", len(arr)))
                fd.write(arr.astype(np.float64).tobytes(order="C"))
            # characters
            chars = data.get("characters", {})
            fd.write(struct.pack("<i", len(chars)))
            for name, s in chars.items():
                s = str(s)
                fd.write(struct.pack("<i", len(name)))
                fd.write(name.encode("utf-8"))
                fd.write(struct.pack("<i", len(s)))
                fd.write(s.encode("utf-8"))
            # matrices
            mats = data.get("matrices", {})
            fd.write(struct.pack("<i", len(mats)))
            for name, mat in mats.items():
                M = _ensure_matrix(mat)
                fd.write(struct.pack("<i", len(name)))
                fd.write(name.encode("utf-8"))
                fd.write(struct.pack("<i", int(M.shape[0])))
                fd.write(struct.pack("<i", int(M.shape[1])))
                fd.write(M.astype(np.float64).ravel(order="C").tobytes(order="C"))
            # smatrices
            smats = data.get("smatrices", {})
            fd.write(struct.pack("<i", len(smats)))
            for name, sm in smats.items():
                if _HAVE_SCIPY and sp.issparse(sm):
                    sm = sm.tocoo()
                    r, c = sm.shape
                    nnz = int(sm.nnz)
                    fd.write(struct.pack("<i", len(name)))
                    fd.write(name.encode("utf-8"))
                    fd.write(struct.pack("<i", r))
                    fd.write(struct.pack("<i", c))
                    fd.write(struct.pack("<i", nnz))
                    fd.write(_np_to_int32(sm.row + 1).tobytes(order="C"))
                    fd.write(_np_to_int32(sm.col + 1).tobytes(order="C"))
                    fd.write(sm.data.astype(np.float64).tobytes(order="C"))
                else:
                    raise ValueError("cgeneric.smatrices require scipy.sparse COO inputs.")
        _writeln(file, f"cgeneric.data = {file_bin.replace(data_dir, '$inladatadir')}\n")

    if inla_one_of(random_spec.get("model"), ["ar", "fgn", "fgn2", "iidkd"]):
        _writeln(file, f"order = {int(random_spec.get('order', 0))}\n")

    # vb.correct conversion
    vb = random_spec.get("vb.correct", True)
    if isinstance(vb, (list, tuple, np.ndarray)):
        vb = np.asarray(vb) - 1
        vb = np.asarray(vb, dtype=int)
        if (vb < 0).any():
            raise ValueError("vb.correct: converted to R->C indexing must be >= 0")
        vbs = " ".join(str(int(x)) for x in np.sort(vb))
    elif isinstance(vb, bool):
        vbs = "-1" if vb else "-2"
    else:
        # numeric scalar?
        try:
            ival = int(vb) - 1
            vbs = str(max(0, ival))
        except Exception:
            vbs = "-2"
    _writeln(file, f"vb.correct = {vbs}\n")
    _writeln(file, "\n")

    # A.local (sparse-aware)
    if random_spec.get("A.local") is not None:
        A_local = _ensure_sparse_or_matrix(random_spec["A.local"])
        # Check for nonzero entries (works for both sparse and dense)
        if _is_sparse(A_local):
            has_nonzero = A_local.nnz > 0
        else:
            has_nonzero = np.count_nonzero(A_local) > 0
        if has_nonzero:
            fA = inla_tempfile(tmpdir=data_dir)
            inla_write_fmesher_file(A_local, filename=fA)
            _writeln(file, f"A.local = {fA.replace(data_dir, '$inladatadir')}\n")

    return random_spec


# ======================================================================================
# 6) INLA Parameters section (control.inla)
# ======================================================================================


def inla_inla_section(file: str, inla_spec: Dict[str, Any], data_dir: str, inla_mode: str) -> None:
    defaults = _normalize_control_inla_dict(control_inla_default())
    user_spec = _normalize_control_inla_dict(inla_spec or {})

    merged: Dict[str, Any] = {k: v for k, v in defaults.items() if k != "control_vb"}
    merged.update({k: v for k, v in user_spec.items() if k != "control_vb"})

    vb_defaults = _normalize_control_vb_dict(control_vb_default())
    vb_defaults.update(_normalize_control_vb_dict(defaults.get("control_vb", {})))
    vb_defaults.update(_normalize_control_vb_dict(user_spec.get("control_vb", {})))
    merged["control_vb"] = vb_defaults

    tol = float(merged.get("tolerance", 0.005))
    if merged.get("tolerance_f") is None:
        merged["tolerance_f"] = tol * 0.4
    if merged.get("tolerance_g") is None:
        merged["tolerance_g"] = tol
    if merged.get("tolerance_x") is None:
        merged["tolerance_x"] = tol * 0.2
    if merged.get("tolerance_step") is None:
        merged["tolerance_step"] = tol / 1000.0

    _writeln(file, f"{inla_secsep('INLA.Parameters')}\n")
    _writeln(file, "type = inla\n")

    int_strategy = merged.get("int_strategy")
    if int_strategy is not None:
        _writeln(file, f"int.strategy = {int_strategy}\n")

    if inla_one_of(int_strategy, ["user", "user.std", "user.expert"]):
        int_design = merged.get("int_design")
        if int_design is None:
            raise ValueError("control.inla: int.strategy requires 'int.design'.")
        fA = inla_tempfile(tmpdir=data_dir)
        inla_write_fmesher_file(_ensure_matrix(int_design), filename=fA)
        _writeln(file, f"int.design = {fA.replace(data_dir, '$inladatadir')}\n")

    strategy = merged.get("strategy")
    if strategy is not None:
        _writeln(file, f"strategy = {strategy}\n")

    _writeln(file, f"adaptive.max = {int(merged.get('adaptive_max', 25))}\n")
    inla_write_boolean_field("fast", merged.get("fast"), file)

    if merged.get("linear_correction") is not None:
        _writeln(file, f"linear.correction = {merged['linear_correction']}\n")
    if merged.get("h") is not None:
        _writeln(file, f"h = {merged['h']}\n")
    if merged.get("dz") is not None:
        _writeln(file, f"dz = {merged['dz']}\n")
    if merged.get("interpolator") is not None:
        _writeln(file, f"interpolator = {merged['interpolator']}\n")
    if merged.get("diff_logdens") is not None:
        _writeln(file, f"diff.log.dens = {merged['diff_logdens']}\n")

    if merged.get("print_joint_hyper", True):
        _writeln(file, "fp.hyperparam = joint.dat\n")

    _writeln(file, f"tolerance.f = {merged['tolerance_f']}\n")
    _writeln(file, f"tolerance.g = {merged['tolerance_g']}\n")
    _writeln(file, f"tolerance.x = {merged['tolerance_x']}\n")
    _writeln(file, f"tolerance.step = {merged['tolerance_step']}\n")

    inla_write_boolean_field("hessian.force.diagonal", merged.get("force_diagonal"), file)
    inla_write_boolean_field("skip.configurations", merged.get("skip_configurations"), file)
    inla_write_boolean_field("adjust.weights", merged.get("adjust_weights"), file)
    inla_write_boolean_field("lincomb.derived.correlation.matrix", merged.get("lincomb_derived_correlation_matrix"), file)

    if merged.get("lincomb_derived_only") is not None:
        raise ValueError("Option control.inla$lincomb.derived.only is disabled.")

    restart = merged.get("restart")
    if restart is not None and int(restart) >= 0:
        _writeln(file, f"restart = {int(restart)}\n")

    optimiser = merged.get("optimiser")
    if optimiser is not None:
        _writeln(file, f"optimiser = {optimiser}\n")

    if merged.get("verbose"):
        _writeln(file, "optpar.fp = stdout\n")
    else:
        _writeln(file, "## optpar.fp = stdout\n")

    reo = merged.get("reordering", "default")
    if isinstance(reo, dict):
        reo = reo.get("name", "default")
    if isinstance(reo, str):
        reo_code = inla_reorderings_name2code(reo)
    else:
        inla_reorderings_code2name(int(reo))
        reo_code = int(reo)
    _writeln(file, f"reordering = {reo_code}\n")

    if merged.get("cpo_diff") is not None:
        _writeln(file, f"cpo.diff = {merged['cpo_diff']}\n")

    if merged.get("npoints") is not None:
        _writeln(file, f"n.points = {int(merged['npoints'])}\n")
    if merged.get("cutoff") is not None:
        _writeln(file, f"cutoff = {merged['cutoff']}\n")

    inla_write_boolean_field("adapt.hessian.mode", merged.get("adapt_hessian_mode"), file)
    if merged.get("adapt_hessian_max_trials") is not None and int(merged["adapt_hessian_max_trials"]) >= 0:
        _writeln(file, f"adapt.hessian.max.trials = {int(merged['adapt_hessian_max_trials'])}\n")
    if merged.get("adapt_hessian_scale") is not None and float(merged["adapt_hessian_scale"]) >= 1.0:
        _writeln(file, f"adapt.hessian.scale = {int(merged['adapt_hessian_scale'])}\n")

    if merged.get("step_len") is not None:
        _writeln(file, f"step.len = {merged['step_len']}\n")

    stencil = merged.get("stencil")
    if stencil is not None:
        stencil = int(stencil)
        if stencil not in (5, 7, 9):
            raise ValueError("stencil must be 5, 7, or 9.")
        _writeln(file, f"stencil = {stencil}\n")
    else:
        _writeln(file, "stencil = 5\n")

    diagonal = merged.get("diagonal")
    if diagonal is not None and float(diagonal) >= 0.0:
        _writeln(file, f"diagonal = {float(diagonal)}\n")
    else:
        _writeln(file, "diagonal = 0\n")

    _writeln(file, f"numint.maxfeval = {int(merged.get('numint_maxfeval', 100000))}\n")
    _writeln(file, f"numint.relerr = {merged.get('numint_relerr', 1e-05)}\n")
    _writeln(file, f"numint.abserr = {merged.get('numint_abserr', 1e-06)}\n")
    _writeln(file, f"cmin = {_format_number(merged.get('cmin', float('-inf')))}\n")

    b_strategy = merged.get("b_strategy", "keep")
    if isinstance(b_strategy, str):
        b_strategy = b_strategy.strip().lower()
        b_out = 0 if b_strategy == "skip" else 1
    else:
        b_out = int(b_strategy)
    _writeln(file, f"b.strategy = {b_out}\n")

    _writeln(file, f"nr.step.factor = {merged.get('step_factor', -0.1)}\n")
    _writeln(file, f"global.node.factor = {merged.get('global_node_factor', 2.0)}\n")
    _writeln(file, f"global.node.degree = {int(merged.get('global_node_degree', 2**31 - 1))}\n")

    inla_write_boolean_field("stupid.search", merged.get("stupid_search"), file)
    _writeln(file, f"stupid.search.max.iter = {int(merged.get('stupid_search_max_iter', 1000))}\n")
    _writeln(file, f"stupid.search.factor = {merged.get('stupid_search_factor', 1.05)}\n")

    vb = merged.get("control_vb", {})
    vb_enable = vb.get("enable", "auto")
    if isinstance(vb_enable, str):
        vb_enable = (inla_mode == "compact")
    inla_write_boolean_field("control.vb.enable", vb_enable, file)
    inla_write_boolean_field("control.vb.verbose", vb.get("verbose", True), file)

    vb_strategy = str(vb.get("strategy", "mean")).lower()
    _writeln(file, f"control.vb.strategy = {vb_strategy}\n")
    _writeln(file, f"control.vb.hessian.update = {int(max(1, round(float(vb.get('hessian_update', 2)))))}\n")

    vb_hstr = str(vb.get("hessian_strategy", "default")).lower()
    if vb_hstr == "default":
        vb_hstr = "full"
    _writeln(file, f"control.vb.hessian.strategy = {vb_hstr}\n")

    lim = vb.get("f_enable_limit", (30, 25, 1024, 768))
    lim_arr = np.asarray(lim, dtype=float).ravel()
    if lim_arr.size < 4:
        lim_arr = np.pad(lim_arr, (0, 4 - lim_arr.size), constant_values=0.0)
    _writeln(file, f"control.vb.f.enable.limit.mean = {lim_arr[0]}\n")
    _writeln(file, f"control.vb.f.enable.limit.variance = {lim_arr[1]}\n")
    _writeln(file, f"control.vb.f.enable.limit.mean.max = {lim_arr[2]}\n")
    _writeln(file, f"control.vb.f.enable.limit.variance.max = {lim_arr[3]}\n")
    _writeln(file, f"control.vb.iter.max = {int(vb.get('iter_max', 25))}\n")
    _writeln(file, f"control.vb.emergency = {abs(float(vb.get('emergency', 25.0)))}\n")

    numgrad = str(merged.get("num_gradient", "central")).lower()
    numhess = str(merged.get("num_hessian", "central")).lower()
    optstr = str(merged.get("optimise_strategy", "smart")).lower()
    _writeln(file, f"num.gradient = {numgrad}\n")
    _writeln(file, f"num.hessian = {numhess}\n")
    _writeln(file, f"optimise.strategy = {optstr}\n")

    use_dir = merged.get("use_directions", True)
    use_dir_flag = use_dir if isinstance(use_dir, bool) else True
    inla_write_boolean_field("use.directions", use_dir_flag, file)
    if use_dir_flag and isinstance(use_dir, (np.ndarray, list)):
        fdir = inla_tempfile(tmpdir=data_dir)
        inla_write_fmesher_file(_ensure_matrix(use_dir), filename=fdir)
        _writeln(file, f"use.directions.matrix = {fdir.replace(data_dir, '$inladatadir')}\n")

    cmd = float(merged.get("constr_marginal_diagonal", math.sqrt(sys.float_info.epsilon)))
    if cmd > 0.0:
        _writeln(file, f"constr.marginal.diagonal = {format(cmd, '.6e')}\n")

    inla_write_boolean_field("improved.simplified.laplace", merged.get("improved_simplified_laplace"), file)
    inla_write_boolean_field("parallel.linesearch", merged.get("parallel_linesearch"), file)
    inla_write_boolean_field("compute.initial.values", merged.get("compute_initial_values"), file)
    inla_write_boolean_field("hessian.correct.skewness.only", merged.get("hessian_correct_skewness_only"), file)

    _writeln(file, "\n")


# ======================================================================================
# 7) Predictor section
# ======================================================================================

def inla_predictor_section(file: str,
                           n: int,
                           m: int,
                           predictor_spec: Dict[str, Any],
                           file_offset: Optional[str],
                           data_dir: str,
                           file_link_fitted_values: Optional[str]) -> None:
    _writeln(file, f"{inla_secsep('Predictor')}\n")
    _writeln(file, "type = predictor\n")
    _writeln(file, "dir = predictor\n")
    _writeln(file, f"n = {int(n)}\n")
    _writeln(file, f"m = {int(m)}\n")

    inla_write_boolean_field("fixed", predictor_spec.get("fixed"), file)
    # Default compute=1 to match common R output
    inla_write_boolean_field("compute", predictor_spec.get("compute", True), file)

    if predictor_spec.get("cdf") is not None:
        _writeln(file, f"cdf = {predictor_spec['cdf']}\n")
    if predictor_spec.get("quantiles") is not None:
        _writeln(file, f"quantiles = {predictor_spec['quantiles']}\n")
    if file_offset is not None:
        _writeln(file, f"offset = {file_offset}\n")
    if file_link_fitted_values is not None:
        _writeln(file, f"link.fitted.values = {file_link_fitted_values}\n")

    inla_write_hyper(predictor_spec.get("hyper"), file=file, data_dir=data_dir)

    # cross constraint
    cross = predictor_spec.get("cross")
    if cross is not None and len(cross) > 0:
        cross = np.asarray(cross).ravel()
        if cross.size != n + m:
            raise ValueError("Length of cross must equal n+m.")
        cross = cross.astype(object)
        # 1..ncross via factor-like mapping
        uniq = [u for u in np.unique(cross) if u is not None]
        mapping = {u: i+1 for i, u in enumerate(uniq)}
        encoded = np.array([mapping.get(v, 0) if v is not None else 0 for v in cross], dtype=int).reshape(-1, 1)
        fcr = inla_tempfile(tmpdir=data_dir)
        inla_write_fmesher_file(encoded, filename=fcr)
        _writeln(file, f"cross.constraint = {fcr.replace(data_dir, '$inladatadir')}\n")

    # A matrix (sparse-aware)
    if predictor_spec.get("A") is not None:
        A = predictor_spec["A"]
        if isinstance(A, str):
            # path to text triplets i j x
            with open(A, "r", encoding="utf-8") as fr:
                txt = fr.read()
            # Keep file as-is (assume correct)
            fA = inla_tempfile(tmpdir=data_dir)
            with open(fA, "w", encoding="utf-8") as fw:
                fw.write(txt)
            _writeln(file, f"A = {fA.replace(data_dir, '$inladatadir')}\n")
            # Cannot compute Aext from file-based A
            raise ValueError("Aext computation requires A as matrix; provide matrix for A.")
        else:
            A = _ensure_sparse_or_matrix(A)
            if A.shape != (m, n):
                raise ValueError("A must be m x n.")
            # Check for zero rows (works for both sparse and dense)
            if _is_sparse(A):
                row_sums = np.abs(A).sum(axis=1).A1  # .A1 converts to 1D array
            else:
                row_sums = np.abs(A).sum(axis=1)
            if np.any(row_sums == 0.0):
                nz = int(np.sum(row_sums == 0.0))
                raise ValueError(f"{nz} rows in control.predictor(list(A=A)) have only zeros; not allowed.")
            fA = inla_tempfile(tmpdir=data_dir)
            inla_write_fmesher_file(A, filename=fA)
            _writeln(file, f"A = {fA.replace(data_dir, '$inladatadir')}\n")

        # Aext = [[I, -A], [-A^T, A^T A]] (sparse-aware)
        if _is_sparse(A):
            I_m = _sparse_diag(m)
            AtA = A.T @ A
            Aext = _sparse_block([
                [I_m, -A],
                [-A.T, AtA]
            ])
        else:
            Aext = np.vstack([
                np.hstack([_diag(m), -A]),
                np.hstack([-A.T, A.T @ A])
            ])
        fAe = inla_tempfile(tmpdir=data_dir)
        inla_write_fmesher_file(Aext, filename=fAe)
        _writeln(file, f"Aext = {fAe.replace(data_dir, '$inladatadir')}\n")
        _writeln(file, f"AextPrecision = {float(predictor_spec.get('precision', 1.0))}\n")

    _writeln(file, "\n")


# ======================================================================================
# 8) lp.scale section
# ======================================================================================

def inla_lp_scale_section(file: str, contr: Dict[str, Any], data_dir: str, write_hyper: bool=True) -> None:
    _writeln(file, f"{inla_secsep('INLA.lp.scale')}\n")
    _writeln(file, "type = lp.scale\n")
    if write_hyper:
        inla_write_hyper(contr.get("hyper"), file=file, data_dir=data_dir)
    _writeln(file, "\n")


# ======================================================================================
# 9) Problem section (top-level model metadata)
# ======================================================================================

def inla_problem_section(file: str,
                         data_dir: str,
                         result_dir: str,
                         hyperpar: bool,
                         return_marginals: bool,
                         return_marginals_predictor: bool,
                         dic: bool,
                         cpo: bool,
                         gcpo: Dict[str, Any],
                         po: bool,
                         mlik: bool,
                         quantiles: Optional[Sequence[float]],
                         smtp: Optional[str],
                         q: bool,
                         openmp_strategy: Optional[str],
                         graph: bool,
                         config: Union[bool, str],
                         likelihood_info: Optional[bool],
                         internal_opt: Optional[bool],
                         save_memory: Optional[bool]) -> None:
    _writeln(file, "")
    _writeln(file, f"###  {inla_version('version')}\n")
    _writeln(file, f"###  Python {platform.python_version()} on {platform.platform()}\n")
    _writeln(file, f"###  {inla_os_type()} - {inla_os_32or64bit()}bit  {_dt.datetime.now()}\n")

    # Session info analogue
    _writeln(file, "\n### [[[start of output from sessionInfo()]]]\n")
    _writeln(file, f"###   sys.version: {sys.version.replace(os.linesep, ' ')}\n")
    _writeln(file, f"###   sys.executable: {sys.executable}\n")
    _writeln(file, f"###   cwd: {os.getcwd()}\n")
    _writeln(file, "### [[[end of output from sessionInfo()]]]\n\n")

    _writeln(file, f"inladatadir = {data_dir}\n")
    _writeln(file, f"inlaresdir = {result_dir}\n")
    _writeln(file, f"##inladatadir = {os.path.basename(data_dir)}\n")
    _writeln(file, f"##inlaresdir = {os.path.basename(result_dir)}-%d\n")

    # "INLA.libR" analogue
    _writeln(file, f"\n{inla_secsep('INLA.libR')}\n")
    _writeln(file, "type = libR\n")
    r_home = os.environ.get("R_HOME", "/usr/lib/R")
    _writeln(file, f"R_HOME = {r_home}\n")

    _writeln(file, f"\n{inla_secsep('INLA.Model')}\n")
    _writeln(file, "type = problem\n")
    _writeln(file, "dir = $inlaresdir\n")
    _writeln(file, f"rinla.version = {inla_version('version')}\n")
    _writeln(file, f"rinla.bdate = {inla_version('date')}\n")

    inla_write_boolean_field("return.marginals", return_marginals, file)
    inla_write_boolean_field("return.marginals.predictor", return_marginals_predictor, file)
    inla_write_boolean_field("hyperparameters", hyperpar, file)
    inla_write_boolean_field("cpo", cpo, file)
    inla_write_boolean_field("po", po, file)
    inla_write_boolean_field("dic", dic, file)
    inla_write_boolean_field("mlik", mlik, file)
    inla_write_boolean_field("q", q, file)
    inla_write_boolean_field("graph", graph, file)

    if internal_opt is None:
        internal_opt = inla_getOption("internal.opt")
    inla_write_boolean_field("internal.opt", internal_opt, file)

    if save_memory is None:
        save_memory = inla_getOption("save.memory")
    inla_write_boolean_field("save.memory", save_memory, file)

    config_lite = False
    if str(config) == "lite":
        config_lite = True
        config = True
    inla_write_boolean_field("config", bool(config), file)
    inla_write_boolean_field("config.lite", config_lite, file)
    if likelihood_info is not None:
        inla_write_boolean_field("likelihood.info", likelihood_info, file)

    # gcpo
    inla_write_boolean_field("gcpo.enable", gcpo.get("enable", False), file)
    inla_write_boolean_field("gcpo.verbose", gcpo.get("verbose", False), file)
    inla_write_boolean_field("gcpo.correct.hyperpar", gcpo.get("correct.hyperpar", True), file)
    inla_write_boolean_field("gcpo.remove.fixed", gcpo.get("remove.fixed", True), file)
    _writeln(file, f"gcpo.epsilon = {max(0.0, float(gcpo.get('epsilon', 5e-3)))}\n")
    _writeln(file, f"gcpo.prior.diagonal = {max(0.0, float(gcpo.get('prior.diagonal', 1e-4)))}\n")
    type_cv_raw = gcpo.get("type.cv", gcpo.get("type_cv", "single"))
    try:
        type_cv_label = match_arg(str(type_cv_raw).strip(), ["single", "joint"])
    except Exception:
        type_cv_label = "single"
    type_cv_code = 0 if type_cv_label.lower().startswith("single") else 1
    _writeln(file, f"gcpo.typecv = {type_cv_code}\n")

    if gcpo.get("keep") is not None and gcpo.get("remove") is not None:
        raise ValueError("control.gcpo$keep and $remove cannot be used at the same time.")
    if gcpo.get("keep") is not None:
        _writeln(file, f"gcpo.keep = {' '.join(str(int(x)) for x in gcpo['keep'])}\n")
    if gcpo.get("remove") is not None:
        _writeln(file, f"gcpo.remove = {' '.join(str(int(x)) for x in gcpo['remove'])}\n")

    # Strategy: default to 'posterior' to match R output when not specified
    strategy = gcpo.get("strategy", "posterior")
    _writeln(file, f"gcpo.strategy = {strategy}\n")

    # friends → groups or selection/group.selection/weights/friends
    groups = gcpo.get("groups")
    friends = gcpo.get("friends")
    if groups is not None and friends is not None:
        raise ValueError("Both friends and groups provided; only one can be used.")
    if groups is not None:
        # Normalize groups into unique-sorted lists (1-based indices)
        glist = []
        for i, g in enumerate(groups, start=1):
            if isinstance(g, dict) and list(g.keys()) == ["idx", "corr"]:
                g = list(g["idx"])
            g = [int(x) for x in (g or []) if int(x) > 0]
            g = sorted(set(g+[i]))
            glist.append(g)
        file_groups = inla_tempfile(tmpdir=data_dir)
        with open(file_groups, "wb") as fb:
            fb.write(struct.pack("<i", len(glist)))
            total_len = len(glist) + sum(len(g) for g in glist if g)
            fb.write(struct.pack("<i", total_len))
            for g in glist:
                fb.write(struct.pack("<i", len(g)))
                if g:
                    fb.write(_np_to_int32([x-1 for x in g]).tobytes(order="C"))
        _writeln(file, f"gcpo.groups = {file_groups.replace(data_dir, '$inladatadir')}\n")
    else:
        gsiz = int(gcpo.get("num.level.sets", -1))
        if gsiz <= 0:
            gsiz = -1
        _writeln(file, f"gcpo.num.level.sets = {gsiz}\n")
        gsiz_max = int(round(float(gcpo.get("size.max", 32))))
        if gsiz_max <= 0:
            gsiz_max = -1
        _writeln(file, f"gcpo.size.max = {gsiz_max}\n")
        if gcpo.get("selection") is not None:
            sel = sorted(set(int(x) for x in gcpo["selection"] if int(x) != 0))
            file_sel = inla_tempfile(tmpdir=data_dir)
            with open(file_sel, "wb") as fb:
                fb.write(struct.pack("<i", len(sel)))
                fb.write(_np_to_int32(sel).tobytes(order="C"))
            _writeln(file, f"gcpo.selection = {file_sel.replace(data_dir, '$inladatadir')}\n")
        if gcpo.get("group.selection") is not None:
            gsel = sorted(set(int(x)-1 for x in gcpo["group.selection"] if int(x) >= 1))
            file_gsel = inla_tempfile(tmpdir=data_dir)
            with open(file_gsel, "wb") as fb:
                fb.write(struct.pack("<i", len(gsel)))
                fb.write(_np_to_int32(gsel).tobytes(order="C"))
            _writeln(file, f"gcpo.group.selection = {file_gsel.replace(data_dir, '$inladatadir')}\n")
        if gcpo.get("weights") is not None:
            w = _np_to_float64(gcpo["weights"])
            file_w = inla_tempfile(tmpdir=data_dir)
            with open(file_w, "wb") as fb:
                fb.write(struct.pack("<i", len(w)))
                fb.write(w.tobytes(order="C"))
            _writeln(file, f"gcpo.weights = {file_w.replace(data_dir, '$inladatadir')}\n")
        if friends is not None:
            friends = [list(set(int(x)-1 for x in (f or []) if int(x) > 0)) for f in friends]
            file_fr = inla_tempfile(tmpdir=data_dir)
            with open(file_fr, "wb") as fb:
                fb.write(struct.pack("<i", len(friends)))
                for fr in friends:
                    fb.write(struct.pack("<i", len(fr)))
                    if fr:
                        fb.write(_np_to_int32(fr).tobytes(order="C"))
            _writeln(file, f"gcpo.friends = {file_fr.replace(data_dir, '$inladatadir')}\n")

    # smtp
    if smtp is None or not (isinstance(smtp, str) and len(smtp) > 0):
        smtp = inla_getOption("smtp")
    smtp = match_arg(str(smtp).lower(), ["band", "taucs", "pardiso", "stiles", "default"])
    _writeln(file, f"smtp = {smtp}\n")

    # openmp.strategy
    if openmp_strategy is None or not (isinstance(openmp_strategy, str) and len(openmp_strategy) > 0):
        openmp_strategy = "default"
    openmp_strategy = match_arg(str(openmp_strategy).lower(),
                                ["default", "small", "medium", "large", "huge",
                                 "pardiso.serial", "pardiso.parallel", "pardiso.nested", "pardiso"])
    if openmp_strategy in ("pardiso.serial", "pardiso.parallel", "pardiso.nested"):
        # Back-compat: all map to "pardiso"
        openmp_strategy = "pardiso"
    _writeln(file, f"openmp.strategy = {openmp_strategy}\n")

    if quantiles is not None:
        qv = " ".join(str(float(q)) for q in quantiles)
        _writeln(file, f"quantiles = {qv}\n")

    _writeln(file, "\n")


# ======================================================================================
# 10) Fixed effects helpers & sections
# ======================================================================================

def inla_parse_fixed_prior(name: str, prior: Optional[Union[float, Dict[str, float]]]) -> Optional[float]:
    if prior is None:
        return None
    if isinstance(prior, (int, float)):
        return float(prior)
    if isinstance(prior, dict):
        if name in prior:
            return float(prior[name])
        if "default" in prior:
            return float(prior["default"])
    return None

def inla_linear_section(file: str,
                        file_fixed: str,
                        label: str,
                        results_dir: str,
                        control_fixed: Dict[str, Any],
                        only_hyperparam: bool) -> Dict[str, Any]:
    _writeln(file, f"{inla_secsep(label)}\n")
    _writeln(file, f"dir = {results_dir}\n")
    _writeln(file, "type = linear\n")
    _writeln(file, f"covariates = {file_fixed}\n")
    if only_hyperparam or not bool(control_fixed.get("compute", True)):
        _writeln(file, "compute = 0\n")
    else:
        _writeln(file, "compute = 1\n")
    if control_fixed.get("cdf") is not None:
        _writeln(file, f"cdf = {control_fixed['cdf']}\n")
    if control_fixed.get("quantiles") is not None:
        _writeln(file, f"quantiles = {control_fixed['quantiles']}\n")

    # Intercept vs general
    if re.search(r"^\(\s*Intercept\s*\)$", inla_trim(label)) is not None:
        # Accept both dotted and underscored keys
        mean = control_fixed.get("mean.intercept", control_fixed.get("mean_intercept", inla_set_control_fixed_default()["mean.intercept"]))
        prec = control_fixed.get("prec.intercept", control_fixed.get("prec_intercept", inla_set_control_fixed_default()["prec.intercept"]))
    else:
        mean = inla_parse_fixed_prior(label, control_fixed.get("mean"))
        if mean is None:
            mean = inla_set_control_fixed_default()["mean"]
        prec = inla_parse_fixed_prior(label, control_fixed.get("prec"))
        if prec is None:
            prec = inla_set_control_fixed_default()["prec"]

    _writeln(file, f"mean = {float(mean)}\n")
    _writeln(file, f"precision = {float(prec)}\n")
    _writeln(file, "\n")
    return {"label": label, "prior.mean": float(mean), "prior.prec": float(prec)}


# ======================================================================================
# 11) Mode section (control.mode)
# ======================================================================================

def inla_mode_section(file: str, args: Dict[str, Any], data_dir: str) -> None:
    if args.get("result") is None and args.get("theta") is None and args.get("x") is None:
        return
    _writeln(file, f"{inla_secsep('INLA.Control.Mode')}\n")
    _writeln(file, "type = mode\n")

    # If result is a path: user should parse it themselves; here we accept dict result.
    result = args.get("result")
    if isinstance(result, str) and os.path.isfile(result):
        raise NotImplementedError("Reading state from a file is not implemented in Python version.")
    if args.get("theta") is None and isinstance(result, dict):
        args["theta"] = result.get("mode", {}).get("theta")
    if args.get("theta") is not None:
        theta = inla_text2vector(args["theta"])
        theta[~np.isfinite(theta)] = 0.0
        ftheta = inla_tempfile(tmpdir=data_dir)
        with open(ftheta, "wb") as fb:
            fb.write(struct.pack("<i", theta.size))
            fb.write(theta.astype(np.float64).tobytes(order="C"))
        _writeln(file, f"theta = {ftheta.replace(data_dir, '$inladatadir')}\n")

    if args.get("x") is None and isinstance(result, dict):
        args["x"] = result.get("mode", {}).get("x")
    if args.get("x") is not None:
        xval = inla_text2vector(args["x"])
        xval[~np.isfinite(xval)] = 0.0
        fx = inla_tempfile(tmpdir=data_dir)
        with open(fx, "wb") as fb:
            fb.write(struct.pack("<i", xval.size))
            fb.write(xval.astype(np.float64).tobytes(order="C"))
        _writeln(file, f"x = {fx.replace(data_dir, '$inladatadir')}\n")

    restart = bool(args.get("restart", True))
    fixed = bool(args.get("fixed", False))
    if fixed and restart:
        restart = False
        # R prints a warning; Python version silently corrects.
    if restart and args.get("theta") is None:
        restart = True  # Keep per R logic note
    inla_write_boolean_field("restart", restart, file)
    inla_write_boolean_field("fixed", fixed, file)
    _writeln(file, "\n")


# ======================================================================================
# 12) Expert section
# ======================================================================================

def inla_expert_section(file: str, args: Dict[str, Any], data_dir: str) -> None:
    _writeln(file, f"{inla_secsep('INLA.Expert')}\n")
    _writeln(file, "type = expert\n")

    if args.get("cpo.manual"):
        inla_write_boolean_field("cpo.manual", True, file)
        cpo_idx = np.asarray(args.get("cpo.idx"), dtype=int).ravel()
        _writeln(file, f"cpo.idx = {' '.join(str(int(x-1)) for x in cpo_idx)}\n")

    if args.get("jp") is not None:
        file_jp = inla_tempfile(tmpdir=data_dir)
        with open(file_jp, "wb") as fb:
            pickle.dump(args["jp"], fb)
        fnm = file_jp.replace(data_dir, "$inladatadir")
        model = ".inla.jp.model"
        _writeln(file, f"jp.file = {fnm}\n")
        _writeln(file, f"jp.model = {model}\n")

    inla_write_boolean_field("disable.gaussian.check", args.get("disable.gaussian.check", False), file)
    # R-INLA parity: opt.solve is always written; dot.product.gain only if explicitly set
    user_keys = args.get("__user_keys__", ())
    if "dot.product.gain" in user_keys or "dot_product_gain" in user_keys:
        inla_write_boolean_field("dot.product.gain", args.get("dot.product.gain", args.get("dot_product_gain", False)), file)
    inla_write_boolean_field("opt.solve", args.get("opt.solve", args.get("opt_solve", False)), file)
    opt_threads = args.get("opt.num.threads")
    if opt_threads is None:
        opt_threads = args.get("opt_num_threads")
    if opt_threads is None:
        opt_threads = args.get("opt.num_threads")
    inla_write_boolean_field("opt.num.threads", opt_threads, file)

    gconstr = args.get("globalconstr")
    if gconstr is not None and gconstr.get("A") is not None:
        A = _ensure_sparse_or_matrix(gconstr["A"])
        e = np.asarray(gconstr["e"], dtype=float).ravel()
        if A.shape[0] != e.size:
            raise ValueError("globalconstr: nrow(A) must equal length(e).")
        fA = inla_tempfile(tmpdir=data_dir)
        # Vectorize by columns - need dense for reshape
        if _is_sparse(A):
            A_dense = A.toarray()
        else:
            A_dense = A
        inla_write_fmesher_file(A_dense.T.reshape(-1, 1), filename=fA)
        _writeln(file, f"globalconstr.A.file = {fA.replace(data_dir, '$inladatadir')}\n")
        fe = inla_tempfile(tmpdir=data_dir)
        inla_write_fmesher_file(e.reshape(-1, 1), filename=fe)
        _writeln(file, f"globalconstr.e.file = {fe.replace(data_dir, '$inladatadir')}\n")

    _writeln(file, "\n")


# ======================================================================================
# 13) Update section
# ======================================================================================

def inla_update_section(file: str, data_dir: str, contr: Dict[str, Any]) -> None:
    res = contr.get("result")
    if res is None:
        return
    theta = np.asarray(res.get("mode", {}).get("theta", []), dtype=float).ravel()
    if theta.size == 0:
        return
    file_update = inla_tempfile(tmpdir=data_dir)
    _writeln(file, f"{inla_secsep('INLA.update')}\n")
    _writeln(file, "type = update\n")
    # Build vector x exactly as R: [len(theta), theta, stdev.corr.positive, stdev.corr.negative, 1/sqrt(eigvals), vec(eigvecs)]
    pos = np.asarray(res.get("misc", {}).get("stdev.corr.positive", []), dtype=float).ravel()
    neg = np.asarray(res.get("misc", {}).get("stdev.corr.negative", []), dtype=float).ravel()
    evals = np.asarray(res.get("misc", {}).get("cov.intern.eigenvalues", []), dtype=float).ravel()
    evecs = np.asarray(res.get("misc", {}).get("cov.intern.eigenvectors", []), dtype=float).ravel(order="F")
    core = np.concatenate([[theta.size], theta, pos, neg, 1.0/np.sqrt(evals+1e-32), evecs])
    inla_write_fmesher_file(core.reshape(-1, 1), filename=file_update)
    _writeln(file, f"filename = {file_update.replace(data_dir, '$inladatadir')}\n")
    _writeln(file, "\n")


# ======================================================================================
# 14) Pardiso / Stiles / Taucs / Numa sections
# ======================================================================================

def inla_pardiso_section(file: str, data_dir: str, contr: Dict[str, Any]) -> None:
    _writeln(file, f"\n{inla_secsep('INLA.pardiso')}\n")
    _writeln(file, "type = pardiso\n")
    verbose = bool(contr.get('verbose', False))
    debug = bool(contr.get('debug', False))
    parallel_reordering = contr.get('parallel.reordering', contr.get('parallel_reordering', True))
    nrhs = contr.get('nrhs', contr.get('num_rhs', -1))
    _writeln(file, f"verbose = {1 if verbose else 0}\n")
    _writeln(file, f"debug = {1 if debug else 0}\n")
    _writeln(file, f"parallel.reordering = {1 if bool(parallel_reordering) else 0}\n")
    _writeln(file, f"nrhs = {int(nrhs)}\n")
    _writeln(file, "\n")

def inla_stiles_section(file: str, data_dir: str, contr: Dict[str, Any]) -> None:
    _writeln(file, f"\n{inla_secsep('INLA.stiles')}\n")
    _writeln(file, "type = stiles\n")
    _writeln(file, f"verbose = {1 if contr.get('verbose') else 0}\n")
    _writeln(file, f"tile.size = {max(int(contr.get('tile.size', 0)), 0)}\n")
    _writeln(file, "\n")

def inla_taucs_section(file: str, data_dir: str, contr: Dict[str, Any]) -> None:
    _writeln(file, f"\n{inla_secsep('INLA.taucs')}\n")
    _writeln(file, "type = taucs\n")
    block_size = contr.get('block.size', contr.get('block_size', 12))
    _writeln(file, f"block.size = {max(int(block_size), 0)}\n")
    _writeln(file, "\n")

def inla_numa_section(file: str, data_dir: str, contr: Dict[str, Any]) -> None:
    enable = contr.get("enable")
    if enable is None:
        enable = inla_getOption("numa")
    _writeln(file, f"\n{inla_secsep('INLA.numa')}\n")
    _writeln(file, "type = numa\n")
    _writeln(file, f"enable = {1 if bool(enable) else 0}\n")
    _writeln(file, "\n")


# ======================================================================================
# 15) Linear combinations section (binary writer)
# ======================================================================================

def inla_lincomb_section(file: str, data_dir: str, contr: Dict[str, Any],
                         lincomb: Optional[Dict[str, Any]]) -> None:
    """
    Accepts a dict mapping names -> LC, where LC is a list like:
      [ {"a": {"idx": [..], "weight": [..]}}, {"b": {...}}, ... ]
    Names None/"" → auto 'lincomb.###'.
    Writes an index file and a single binary payload file for all lincombs.
    """
    if lincomb is None:
        return
    fnm = inla_tempfile(tmpdir=data_dir)
    open(fnm, "wb").close()
    fp = open(fnm, "ab")

    numlen = inla_numlen(len(lincomb))
    prev_names: List[str] = []

    for i, (nm, lc) in enumerate(lincomb.items(), start=1):
        if not nm:
            secname = f"lincomb.{inla_num(i, width=numlen)}"
        else:
            secname = f"lincomb.{nm}"
        if secname in prev_names:
            raise ValueError(f"Duplicated name [{secname}] in 'lincomb'; need unique names or ''.")
        prev_names.append(secname)

        _writeln(file, f"\n{inla_secsep(secname)}\n")
        _writeln(file, "type = lincomb\n")
        _writeln(file, f"lincomb.order = {i}\n")
        # verbose defaults to 0 (False) to match R-INLA behavior
        verbose_val = contr.get("verbose", False)
        _writeln(file, f"verbose = {1 if verbose_val else 0}\n")
        # current offset
        off = fp.tell()
        _writeln(file, f"file.offset = {int(off)}\n")

        # number of entries
        fp.write(struct.pack("<i", len(lc)))
        for entry in lc:
            # entry is dict with a single key: variable name
            if not isinstance(entry, dict) or len(entry) != 1:
                raise ValueError("Each lincomb entry must be a dict with single variable key.")
            vname = list(entry.keys())[0]
            spec = entry[vname]
            idx = np.asarray(spec.get("idx", []), dtype=int).ravel()
            weight = np.asarray(spec.get("weight", []), dtype=float).ravel()
            if idx.size == 0 and weight.size == 1:
                idx = np.array([1], dtype=int)  # default to 1 like R
            # Remove NAs (not applicable to Python; keep as-is)
            if idx.size == 0 or weight.size == 0:
                raise ValueError(f"lincomb {secname} has only zero entries; not allowed.")
            if idx.size != weight.size:
                raise ValueError("idx and weight length mismatch.")
            fp.write(struct.pack("<i", len(vname)))
            fp.write(vname.encode("utf-8"))
            fp.write(b'\x00')  # Null terminator (R's writeBin adds this for character)
            fp.write(struct.pack("<i", idx.size))
            fp.write(_np_to_int32(idx).tobytes(order="C"))
            fp.write(_np_to_float64(weight).tobytes(order="C"))

        fnm_rel = fnm.replace(data_dir, "$inladatadir")
        _writeln(file, f"filename = {fnm_rel}\n")
        _writeln(file, "\n")

    fp.close()


# ======================================================================================
# 16) Public small helpers you may want to import directly
# ======================================================================================

# Exported helper aliases (optional)
inla_text2vector = inla_text2vector  # already defined
inla_secsep = inla_secsep

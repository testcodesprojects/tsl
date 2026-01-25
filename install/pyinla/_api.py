# pyinla_api.py
# Top-level INLA runner adapted from R's inla()/inla.core()/inla.core.safe(),
# using dict-based model specification (no formula needed).
# It assumes your earlier-converted Python helpers from section.R are available
# (e.g., problem_section, data_section, predictor_section, ffield_section, etc.).

from __future__ import annotations
import os
import sys
import time
import math
import json
import shutil
import random
import string
import platform
import subprocess
import re
import copy
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Callable
from itertools import combinations

import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import scipy.sparse as sp  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    sp = None  # type: ignore
    _HAVE_SCIPY = False

# --- Import section writers (Python port of section.R) ---
# In this codebase the section writers are defined with an `inla_` prefix.
# Import them and alias to the names used throughout this file.
try:
    from .sections import (
        inla_secsep as inla_secsep,
        inla_problem_section as problem_section,
        inla_data_section as data_section,
        inla_predictor_section as predictor_section,
        inla_ffield_section as ffield_section,
        inla_linear_section as linear_section,
        inla_lp_scale_section as lp_scale_section,
        inla_inla_section as inla_section,
        inla_mode_section as mode_section,
        inla_expert_section as expert_section,
        inla_lincomb_section as lincomb_section,
        inla_update_section as update_section,
        inla_pardiso_section as pardiso_section,
        inla_stiles_section as stiles_section,
        inla_taucs_section as taucs_section,
        inla_numa_section as numa_section,
        inla_write_fmesher_file as write_fmesher_file,
        inla_copy_file_for_section as copy_file_for_section,
        inla_copy_dir_for_section_spde as copy_dir_for_section_spde,
        inla_version as inla_version,
        inla_os_type as inla_os_type,
        inla_model_properties as inla_model_properties,
    )
except Exception as _e:
    raise ImportError(
        "Could not import your previously converted section-writer functions. "
        "Please ensure `sections.py` (the Python port of section.R) is importable."
    ) from _e

# Data-file generator for likelihood input (Python port of inla.create.data.file)
try:
    from .create_data_file import create_data_file as _create_data_file
except Exception:
    _create_data_file = None

from .control_defaults import (
    control_compute as control_compute_default,
    control_predictor as control_predictor_default,
    control_family as control_family_default,
    control_inla as control_inla_default,
    control_fixed as control_fixed_default,
    control_mode as control_mode_default,
    control_expert as control_expert_default,
    control_hazard as control_hazard_default,
    control_lp_scale as control_lp_scale_default,
    control_pardiso as control_pardiso_default,
    control_stiles as control_stiles_default,
    control_taucs as control_taucs_default,
    control_numa as control_numa_default,
    control_vb as control_vb_default,
)
from .options import inla_call_builtin
from .qinv import qinv as _qinv
from .utils import inla_rw1 as _rw1, inla_rw2 as _rw2
from .scale_model import scale_model as _scale_model
from .marginal_utils import (
    inla_dmarginal as _dmarginal,
    inla_emarginal as _emarginal,
    inla_hpdmarginal as _hpdmarginal,
    inla_mmarginal as _mmarginal,
    inla_marginal_fix as _marginal_fix,
    inla_marginal_transform as _marginal_transform,
    inla_pmarginal as _pmarginal,
    inla_qmarginal as _qmarginal,
    inla_rmarginal as _rmarginal,
    inla_smarginal as _smarginal,
    inla_sfmarginal as _sfmarginal,
    inla_spline as _spline,
    inla_tmarginal as _tmarginal,
    inla_zmarginal as _zmarginal,
)
from .fmesher_io import read_fmesher_file as _read_fmesher_file
from .models import INLAModels as _INLAModels
from .surv import is_inla_surv as _is_inla_surv
from .read_graph import inla_read_graph
from ._safety import (
    SafetyError,
    enforce_allowed_family,
    enforce_gaussian_hyperstructure,
    enforce_gamma_hyperstructure,
    enforce_gamma_support,
    enforce_logistic_hyperstructure,
    enforce_loglogistic_hyperstructure,
    enforce_sn_hyperstructure,
    enforce_t_hyperstructure,
    enforce_beta_hyperstructure,
    enforce_scale_usage,
    enforce_compute_section,
    enforce_exposure_usage,
    enforce_poisson_exposure,
    enforce_nbinomial_exposure,
    enforce_binomial_trials,
    enforce_binomial_family_variant,
    enforce_beta_support,
    enforce_survival_response,
    enforce_control_structure,
    enforce_random_structure,
    enforce_untested_arguments,
    enforce_poisson_support,
    enforce_nbinomial_support,
    enforce_binomial_support,
    enforce_exponential_support,
    enforce_lognormal_support,
    enforce_weibull_support,
    enforce_loglogistic_support,
    enforce_gaussian_support,
    enforce_logistic_support,
    enforce_t_support,
    enforce_sn_support,
)

_ANNOUNCED = False
_MODELS_DB = _INLAModels()  # shared model registry for defaults

# -----------------------
# License & Expiration
# -----------------------
# This package requires activation and expires yearly to ensure users get updates.
# Activation is saved to ~/.pyinla/license so users only need to activate once per machine.
import datetime as _datetime
import hashlib as _hashlib
from pathlib import Path as _Path

# Valid years for this version (update annually when releasing new version)
_VALID_YEARS = {2025, 2026}

# License file location
_LICENSE_DIR = _Path.home() / ".pyinla"
_LICENSE_FILE = _LICENSE_DIR / "license"

# Activation state
_ACTIVATED = False
_ACTIVATION_CODE = None

# The activation codes are SHA-256 hashes - you keep the original codes secret
# To generate a new code: hashlib.sha256("your_secret_code".encode()).hexdigest()
_VALID_ACTIVATION_HASHES = {
    # Add hashed activation codes here (one per authorized user/institution)
    # Example: hashlib.sha256("PYINLA-2026-BETA-TESTER".encode()).hexdigest()
    _hashlib.sha256("PYINLA-2026-VALIDATION".encode()).hexdigest(),
    _hashlib.sha256("PYINLA-RESEARCH-ACCESS".encode()).hexdigest(),
}


def _check_expiration() -> None:
    """Check if the package has expired based on current year."""
    current_year = _datetime.datetime.now().year
    if current_year not in _VALID_YEARS:
        raise RuntimeError(
            f"pyINLA version expired. This version is valid for years {sorted(_VALID_YEARS)}. "
            f"Current year: {current_year}. Please update to the latest version: pip install --upgrade pyinla"
        )


def _save_license(code: str) -> None:
    """Save license code to config file for one-time activation."""
    try:
        _LICENSE_DIR.mkdir(parents=True, exist_ok=True)
        _LICENSE_FILE.write_text(code.strip())
    except (OSError, IOError):
        # Silently fail if we can't write - activation still works for this session
        pass


def _load_license() -> str:
    """Load license code from config file if it exists."""
    try:
        if _LICENSE_FILE.exists():
            return _LICENSE_FILE.read_text().strip()
    except (OSError, IOError):
        pass
    return ""


def activate(code: str) -> bool:
    """
    Activate pyINLA with your license code.

    The license is saved to ~/.pyinla/license, so you only need to activate
    once per machine. Future sessions will auto-activate.

    Parameters
    ----------
    code : str
        Your activation code provided by the pyINLA team.

    Returns
    -------
    bool
        True if activation successful.

    Raises
    ------
    RuntimeError
        If activation code is invalid.

    Example
    -------
    >>> from pyinla import pyinla
    >>> pyinla.activate("YOUR-ACTIVATION-CODE")  # Only needed once
    True
    """
    global _ACTIVATED, _ACTIVATION_CODE

    # Check expiration first
    _check_expiration()

    # Hash the provided code and check against valid hashes
    code_hash = _hashlib.sha256(code.strip().encode()).hexdigest()

    if code_hash in _VALID_ACTIVATION_HASHES:
        _ACTIVATED = True
        _ACTIVATION_CODE = code
        # Save to config file for future sessions
        _save_license(code)
        print("[pyINLA] Activation successful. License saved to ~/.pyinla/license")
        print("[pyINLA] You won't need to activate again on this machine.")
        return True
    else:
        raise RuntimeError(
            "pyINLA activation failed: Invalid activation code. "
            "Please contact the pyINLA team for a valid license code."
        )


def _try_auto_activate() -> bool:
    """Try to auto-activate from saved license file or environment variable."""
    global _ACTIVATED, _ACTIVATION_CODE

    if _ACTIVATED:
        return True

    # Priority 1: Check saved license file
    saved_code = _load_license()
    if saved_code:
        code_hash = _hashlib.sha256(saved_code.encode()).hexdigest()
        if code_hash in _VALID_ACTIVATION_HASHES:
            _ACTIVATED = True
            _ACTIVATION_CODE = saved_code
            return True

    # Priority 2: Check environment variable
    env_code = os.environ.get("PYINLA_LICENSE_KEY", "").strip()
    if env_code:
        code_hash = _hashlib.sha256(env_code.encode()).hexdigest()
        if code_hash in _VALID_ACTIVATION_HASHES:
            _ACTIVATED = True
            _ACTIVATION_CODE = env_code
            # Also save to file for future sessions
            _save_license(env_code)
            return True

    return False


def _require_activation() -> None:
    """Check that the package is activated before use."""
    # First check expiration
    _check_expiration()

    # Try auto-activation from saved file or environment variable
    _try_auto_activate()

    # Then check activation
    if not _ACTIVATED:
        raise RuntimeError(
            "pyINLA is not activated. This package requires activation for use.\n\n"
            "To activate (only needed once per machine):\n\n"
            "  from pyinla import pyinla\n"
            "  pyinla.activate('YOUR-ACTIVATION-CODE')\n\n"
            "Your license will be saved to ~/.pyinla/license for future sessions.\n\n"
            "If you don't have an activation code, please contact the pyINLA team.\n"
            "This requirement will be removed after the validation paper is published."
        )


# -----------------------
# Safety Gate Token
# -----------------------
# This token ensures _run_impl cannot be called directly without passing through
# the safety checks. The token is generated once at module load and is only known
# internally. Direct calls to _run_impl without the correct token will fail.
import secrets as _secrets
_SAFETY_GATE_TOKEN = _secrets.token_hex(32)  # 256-bit random token


def _validate_safety_token(token: str) -> None:
    """Validate that the caller has passed through the safety gate."""
    if token != _SAFETY_GATE_TOKEN:
        raise PyINLAError(
            "pyINLA safety gate violation: direct calls to internal implementation are not allowed. "
            "Use pyinla(...) which enforces validation checks."
        )


# -----------------------
# Errors
# -----------------------
class PyINLAError(RuntimeError):
    pass

class PyINLACrashError(PyINLAError):
    """Raised when the INLA binary crashes or returns non-zero."""

class PyINLACollectError(PyINLAError):
    """Raised when collecting INLA results fails."""


def _stack_extra_constraints(
    existing: Optional[Tuple[np.ndarray, np.ndarray]],
    new_A: Any,
    new_e: Any,
    expected_cols: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Validate and append extra constraint rows to the (A, e) pair."""
    if new_A is None:
        return existing if existing is not None else None

    if _HAVE_SCIPY and sp is not None and sp.issparse(new_A):  # type: ignore[attr-defined]
        A_arr = new_A.toarray().astype(float)
    else:
        A_arr = np.asarray(new_A, dtype=float)

    if A_arr.ndim == 1:
        A_arr = A_arr.reshape(1, -1)
    elif A_arr.ndim != 2:
        raise PyINLAError("extraconstr$A must be a 1D or 2D array-like object.")

    if A_arr.shape[1] != expected_cols:
        raise PyINLAError(
            f"extraconstr has {A_arr.shape[1]} columns; expected {expected_cols}."
        )

    e_arr = np.asarray(new_e, dtype=float).reshape(-1)
    if A_arr.shape[0] != e_arr.size:
        raise PyINLAError("extraconstr rows must match length of 'e'.")

    if existing is None:
        return (A_arr.copy(), e_arr.copy())

    old_A, old_e = existing
    if old_A.shape[1] != expected_cols:
        raise PyINLAError("Existing extraconstr has incompatible column dimension.")

    stacked_A = np.vstack([old_A, A_arr]) if old_A.size else A_arr.copy()
    stacked_e = np.concatenate([old_e, e_arr]) if old_e.size else e_arr.copy()
    return (stacked_A, stacked_e)


# -----------------------
# Utilities
# -----------------------
_ENV_EXPORT_PATTERN = re.compile(
    r"^(INLA_|(OPENBLAS|MKL|BLIS)_NUM_THREADS|OMP_|MIMALLOC_|MALLOC_CONF|TSAN_OPTIONS)"
)
_ENV_CONTROLLED_VARS = [
    "OMP_NUM_THREADS",
    "OMP_SCHEDULE",
    "OMP_MAX_ACTIVE_LEVELS",
    "MIMALLOC_ARENA_EAGER_COMMIT",
    "MIMALLOC_PURGE_DELAY",
    "MIMALLOC_PURGE_DECOMMITS",
    "MIMALLOC_SHOW_STATS",
    "MIMALLOC_VERBOSE",
    "MIMALLOC_SHOW_ERRORS",
    "MALLOC_CONF",
    "TSAN_OPTIONS",
]
_MIMALLOC_DEFAULTS = {
    "MIMALLOC_ARENA_EAGER_COMMIT": "1",
    "MIMALLOC_PURGE_DELAY": "-1",
    "MIMALLOC_PURGE_DECOMMITS": "0",
    "MIMALLOC_SHOW_STATS": "0",
    "MIMALLOC_VERBOSE": "0",
    "MIMALLOC_SHOW_ERRORS": "0",
    "MALLOC_CONF": "abort_conf:true,metadata_thp:always,dirty_decay_ms:-1,percpu_arena:percpu",
    "TSAN_OPTIONS": "ignore_noninstrumented_modules=1",
}

_PRIOR_ALIASES = {
    "pc.mgamma": "pcmgamma",
    "pc.dof": "pcdof",
    "pc.alphaw": "pcalphaw",
}


def _normalize_prior_name(name: Any) -> Any:
    if isinstance(name, str):
        key = name.strip().lower()
        alias = _PRIOR_ALIASES.get(key)
        if alias is not None:
            return alias
    return name


def _normalize_control_priors(control: Dict[str, Any]) -> None:
    family_block = control.get("family") if isinstance(control, dict) else None
    if not isinstance(family_block, dict):
        return
    hyper = family_block.get("hyper")
    if isinstance(hyper, list):
        for entry in hyper:
            if isinstance(entry, dict) and "prior" in entry:
                entry["prior"] = _normalize_prior_name(entry.get("prior"))
    elif isinstance(hyper, dict):
        # Handle dict-style hyper specification (e.g., {'prec': {...}, 'dof': {...}})
        for key, entry in hyper.items():
            if isinstance(entry, dict) and "prior" in entry:
                entry["prior"] = _normalize_prior_name(entry.get("prior"))


def _package_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _default_inla_path() -> str:
    hint = os.environ.get("INLA_PATH")
    if hint and os.path.isdir(hint):
        return hint
    candidate = os.path.join(_package_root(), "bin")
    if os.path.isdir(candidate):
        return candidate
    return _package_root()


def _inla_set_environment() -> None:
    env_updates = {
        "INLA_PATH": _default_inla_path(),
        "INLA_OS": inla_os_type(),
        "INLA_VERSION": inla_version("version"),
        "INLA_RVERSION": f"Python {platform.python_version()}",
        "INLA_RHOME": os.environ.get("INLA_RHOME", sys.prefix),
        "INLA_MALLOC_LIB": os.environ.get("INLA_MALLOC_LIB", "mi"),
    }
    for key, value in env_updates.items():
        if value is not None:
            os.environ[key] = str(value)


def _inla_run_environment_set() -> Dict[str, str]:
    saved = {var: os.environ.get(var, "") for var in _ENV_CONTROLLED_VARS}
    for var in _ENV_CONTROLLED_VARS:
        os.environ.pop(var, None)
    for var, value in _MIMALLOC_DEFAULTS.items():
        os.environ[var] = value
    return saved


def _inla_run_environment_unset(saved: Dict[str, str]) -> None:
    if not saved:
        return
    for var, value in saved.items():
        if value:
            os.environ[var] = value
        else:
            os.environ.pop(var, None)


def _write_environment_file(inla_dir: str) -> str:
    env_file = os.path.join(inla_dir, "environment")
    with open(env_file, "w", encoding="utf-8") as fh:
        for key in sorted(os.environ.keys()):
            if _ENV_EXPORT_PATTERN.match(key):
                value = os.environ.get(key, "")
                fh.write(f"export {key}='{value}'\n")
    return env_file


def _as_list(x: Union[None, str, List[str]]) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _is_true(value: Any) -> bool:
    return isinstance(value, bool) and value is True


def _link_has_assignments(link: Any) -> bool:
    if link is None:
        return False
    if isinstance(link, np.ndarray):
        iterable = link.ravel().tolist()
    elif isinstance(link, (list, tuple, set)):
        iterable = list(link)
    else:
        return True
    for item in iterable:
        if item is None:
            continue
        if isinstance(item, (float, np.floating)):
            if math.isnan(item):
                continue
        return True
    return False


def _ensure_predictor_compute(control: Dict[str, Any]) -> None:
    compute = control.get("compute") or {}
    predictor = control.get("predictor") or {}
    control_inla = control.get("inla") or {}

    need_compute = any(
        _is_true(compute.get(flag))
        for flag in ("cpo", "dic", "po", "waic")
    )

    if not need_compute:
        gcpo = compute.get("control_gcpo") or {}
        if isinstance(gcpo, dict) and _is_true(gcpo.get("enable")):
            need_compute = True

    if not need_compute and _link_has_assignments(predictor.get("link")):
        need_compute = True

    if not need_compute:
        control_vb = control_inla.get("control_vb") or {}
        if isinstance(control_vb, dict):
            vb_enable = control_vb.get("enable")
        else:
            vb_enable = control_vb
        if isinstance(vb_enable, str):
            need_compute = True
        elif isinstance(vb_enable, bool) and vb_enable:
            need_compute = True

    if need_compute:
        predictor["compute"] = True
        control["predictor"] = predictor

def _is_dataframe_like(obj: Any) -> bool:
    return (pd is not None) and isinstance(obj, pd.DataFrame)

def _normalize_data_and_model(model: Dict[str, Any],
                              data: Union[Dict[str, Any], "pd.DataFrame", None]
                              ) -> Tuple["pd.DataFrame", Dict[str, Any]]:
    """Normalize caller-provided `data` to a pandas DataFrame and, when
    operating in vector-only mode, normalize `model['response']` to the
    canonical column name 'y'.

    Rules
    - If `data` is a pandas DataFrame, return it unchanged.
    - Otherwise, pandas must be available.
    - If `data` is None, expect `model['response']` to be array-like; build a
      minimal DataFrame with a single column 'y' and set `model['response']='y'`.
      If `model['response']` is a string (column name) or missing, raise.
    - If `data` is a dict-like, convert it with `pd.DataFrame(data)`.
    """
    if _is_dataframe_like(data):
        return data, model  # type: ignore[return-value]

    if pd is None:
        raise PyINLAError("pandas is required if you pass `data` as a dict or omit it.")

    if data is None:
        resp = model.get("response", None)
        if isinstance(resp, str) or resp is None:
            raise PyINLAError(
                "Data is required when `model['response']` is a column name. "
                "Either pass a DataFrame/dict in `data`, or pass `response` as a vector."
            )
        yvec = np.asarray(resp).reshape(-1)
        df = pd.DataFrame({"y": yvec})
        model2 = dict(model)
        model2["response"] = "y"
        return df, model2

    # dict-like input
    df = pd.DataFrame(data)
    return df, model

def _resolve_work_parent(working_directory: Optional[str]) -> Path:
    """
    Resolve and create (if needed) the parent directory that will hold INLA
    working folders. Defaults to a `work/` directory next to the bundled INLA
    binaries inside the installed package.
    """
    if working_directory:
        parent = Path(working_directory).expanduser()
        if not parent.is_absolute():
            parent = Path(os.getcwd()) / parent
    else:
        parent = Path(_default_inla_path()) / "work"
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PyINLAError(
            f"Failed to prepare pyINLA working directory at '{parent}': {exc}"
        ) from exc
    return parent.resolve()


def _ensure_workdir(keep: bool, working_directory: Optional[str]) -> str:
    """
    Create the directory used for a single INLA execution. Persistent folders
    live below the resolved parent (typically the installed package location);
    ephemeral runs receive a random suffix and are later cleaned up.
    """
    parent = _resolve_work_parent(working_directory)
    if keep:
        base = parent / "inla.model"
        candidate = base
        suffix = 0
        while candidate.exists():
            suffix += 1
            candidate = parent / f"inla.model-{suffix}"
        candidate.mkdir(parents=True, exist_ok=False)
        return str(candidate)

    # Ephemeral workspace: try a bounded number of random names.
    alphabet = string.ascii_lowercase + string.digits
    for _ in range(32):
        stem = "inla.model-" + "".join(random.choices(alphabet, k=10))
        candidate = parent / stem
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return str(candidate)

    raise PyINLAError("Unable to allocate a unique pyINLA working directory.")

def _mk_subdirs(inla_dir: str) -> Tuple[str, str, str]:
    data_dir = os.path.join(inla_dir, "data.files")
    os.makedirs(data_dir, exist_ok=True)
    # Match R's formatting pattern width (10 digits)
    results_dir = os.path.join(inla_dir, "results.files-%10.10d")
    file_ini  = os.path.join(inla_dir, "Model.ini")
    return data_dir, results_dir, file_ini


def _parse_reference_model_ini(reference: Optional[str]) -> Dict[str, Any]:
    mapping: Dict[str, Any] = {
        "data": {},
        "predictor": {},
        "fixed": [],
        "random": [],
    }
    if not reference:
        return mapping
    ref_path = Path(reference)
    if ref_path.is_dir():
        ref_path = ref_path / "Model.ini"
    if not ref_path.exists():
        return mapping

    def _extract_name(line: str) -> str:
        part = line.split("=", 1)[1].strip()
        if "/" in part:
            return part.split("/")[-1]
        return part

    current = None
    component: Optional[Dict[str, Any]] = None
    with ref_path.open("r", encoding="utf-8", errors="ignore") as fp:
        for raw in fp:
            line = raw.strip()
            if not line:
                if component and component.get("type") and component["type"].lower() != "linear":
                    mapping["random"].append(component)
                component = None
                current = None
                continue
            if line.startswith("!") and line.endswith("!"):
                if component and component.get("type") and component["type"].lower() != "linear":
                    mapping["random"].append(component)
                component = None
                name = line[1:-1]
                if name.startswith("INLA.Data"):
                    current = "data"
                elif name == "Predictor":
                    current = "predictor"
                elif name.startswith("INLA."):
                    current = None
                else:
                    current = "component"
                    component = {"name": name}
                continue

            if current == "data":
                if line.startswith("filename ="):
                    mapping["data"]["file.data"] = _extract_name(line)
                elif line.startswith("weights ="):
                    mapping["data"]["file.weights"] = _extract_name(line)
                elif line.startswith("attributes ="):
                    mapping["data"]["file.attr"] = _extract_name(line)
                elif line.startswith("lpscale ="):
                    mapping["data"]["file.lp.scale"] = _extract_name(line)
            elif current == "predictor":
                if line.startswith("offset ="):
                    mapping["predictor"]["offset"] = _extract_name(line)
            elif current == "component" and component is not None:
                if line.startswith("type ="):
                    component["type"] = line.split("=", 1)[1].strip().lower()
                elif line.startswith("covariates ="):
                    name = _extract_name(line)
                    if component.get("type") == "linear":
                        mapping["fixed"].append(name)
                    else:
                        component["covariates"] = name
                elif line.startswith("locations ="):
                    component["locations"] = _extract_name(line)

        # finalize if file ended inside component
        if component and component.get("type") and component["type"].lower() != "linear":
            mapping["random"].append(component)

    return mapping


def _resolve_reference_path(reuse_hint: Optional[str], working_directory: Optional[str]) -> Optional[str]:
    if reuse_hint:
        return reuse_hint

    candidates: List[Path] = []

    def collect(root: Optional[str]) -> None:
        if not root:
            return
        path = Path(root)
        if not path.exists():
            return
        for p in path.iterdir():
            if p.is_dir() and p.name.startswith("inla.model"):
                candidates.append(p)

    collect(os.environ.get("PYINLA_REFERENCE_ROOT"))
    if working_directory:
        wd = Path(working_directory)
        collect(str(wd))
        collect(str(wd.parent))

    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest)


def _normalize_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if k == "__control__":
            continue
        out[k.replace('.', '_')] = v
    return out


def _recursive_merge(default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(default)
    for key, value in user.items():
        if key == "__control__":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _recursive_merge(merged[key], _normalize_keys(value))
        else:
            merged[key] = value
    return merged

_FAMILY_HYPER_KEYS = {"initial", "fixed", "prior", "param"}


def _normalize_family_hyper_override(hyper_spec: Any) -> Optional[Dict[str, Dict[str, Any]]]:
    if not hyper_spec:
        return None

    def _filter_values(entry: Dict[str, Any]) -> Dict[str, Any]:
        return {k: entry[k] for k in _FAMILY_HYPER_KEYS if k in entry}

    if isinstance(hyper_spec, dict):
        out: Dict[str, Dict[str, Any]] = {}
        for key, value in hyper_spec.items():
            if isinstance(value, dict):
                filtered = _filter_values(value)
                if filtered:
                    out[str(key)] = filtered
        return out or None

    if isinstance(hyper_spec, (list, tuple)):
        out: Dict[str, Dict[str, Any]] = {}
        for idx, entry in enumerate(hyper_spec, start=1):
            if not isinstance(entry, dict):
                continue
            key: Optional[str] = None
            theta_val = entry.get("theta")
            if theta_val is not None:
                try:
                    key = f"theta{int(theta_val)}"
                except Exception:
                    pass
            if key is None:
                for name_key in ("name", "short_name", "id", "key"):
                    val = entry.get(name_key)
                    if val:
                        key = str(val)
                        break
            if key is None:
                key = f"theta{idx}"
            filtered = _filter_values(entry)
            if filtered:
                out[key] = filtered
        return out or None

    return None


def _merge_family_hyper_defaults(family_ctrl: Dict[str, Any], family_name: Optional[str]) -> None:
    if not family_name:
        return
    try:
        props = _MODELS_DB.get_model_properties(family_name, "likelihood", stop_on_error=False)
    except Exception:
        return
    if not props:
        return
    try:
        merged = _MODELS_DB.set_hyper(
            model=family_name,
            section="likelihood",
            hyper=_normalize_family_hyper_override(family_ctrl.get("hyper")),
            initial=family_ctrl.get("initial"),
            fixed=family_ctrl.get("fixed"),
            prior=family_ctrl.get("prior"),
            param=family_ctrl.get("param"),
        )
    except Exception:
        return
    family_ctrl["hyper"] = list(merged.values())


def _merge_random_hyper_defaults(random_spec: Dict[str, Any]) -> None:
    """Merge default hyperparameters from models.py into a random effect spec."""
    model_name = random_spec.get("model")
    if not model_name:
        return
    model_name = str(model_name).lower()
    try:
        props = _MODELS_DB.get_model_properties(model_name, "latent", stop_on_error=False)
    except Exception:
        return
    if not props:
        return
    try:
        merged = _MODELS_DB.set_hyper(
            model=model_name,
            section="latent",
            hyper=_normalize_family_hyper_override(random_spec.get("hyper")),
            initial=random_spec.get("initial"),
            fixed=random_spec.get("fixed"),
            prior=random_spec.get("prior"),
            param=random_spec.get("param"),
        )
    except Exception:
        return
    random_spec["hyper"] = list(merged.values())


def _apply_control_defaults(control: Optional[Dict[str, Any]], primary_family: Optional[str] = None) -> Dict[str, Any]:
    ctrl = _normalize_keys(control or {})

    def merge(default_factory, key: str) -> Dict[str, Any]:
        raw_user = ctrl.get(key, {})
        if not isinstance(raw_user, dict):
            user_section: Dict[str, Any] = {}
        else:
            user_section = _normalize_keys(raw_user)
        merged = _recursive_merge(default_factory(), user_section)
        merged.setdefault("__control__", key)
        merged["__user_keys__"] = tuple(sorted(user_section.keys()))
        return merged

    ctrl["compute"] = merge(control_compute_default, "compute")
    ctrl["predictor"] = merge(control_predictor_default, "predictor")
    ctrl["family"] = merge(control_family_default, "family")
    ctrl["inla"] = merge(control_inla_default, "inla")
    ctrl["fixed"] = merge(control_fixed_default, "fixed")
    ctrl["mode"] = merge(control_mode_default, "mode")
    ctrl["expert"] = merge(control_expert_default, "expert")
    ctrl["hazard"] = merge(control_hazard_default, "hazard")
    ctrl["lp_scale"] = merge(control_lp_scale_default, "lp_scale")
    ctrl["pardiso"] = merge(control_pardiso_default, "pardiso")
    ctrl["stiles"] = merge(control_stiles_default, "stiles")
    ctrl["taucs"] = merge(control_taucs_default, "taucs")
    ctrl["numa"] = merge(control_numa_default, "numa")
    ctrl.setdefault("lincomb", {})
    ctrl.setdefault("update", {})
    ctrl.setdefault("debug", False)
    ctrl.setdefault("only_hyperparam", False)

    # R parity: if user requested fixed-effects correlation matrix via
    # control['fixed']['correlation_matrix'], map this to the INLA.Parameters
    # flag 'lincomb.derived.correlation.matrix' so the binary produces it.
    try:
        fixed = ctrl.get("fixed", {}) or {}
        corr_flag = fixed.get("correlation_matrix", False)
        if bool(corr_flag):
            inla_sec = ctrl.get("inla", {}) or {}
            inla_sec["lincomb_derived_correlation_matrix"] = True
            ctrl["inla"] = inla_sec
    except Exception:
        pass

    family_section = ctrl.get("family")
    if isinstance(family_section, dict):
        _merge_family_hyper_defaults(family_section, primary_family)

    return ctrl

def _detect_length(vec_like) -> int:
    if vec_like is None:
        return 0
    if hasattr(vec_like, "__len__"):
        return len(vec_like)
    try:
        return int(np.size(vec_like))
    except Exception:
        return 0

def _num_threads_to_arg(num_threads: Union[str, int]) -> str:
    """
    Port of inla.parse.num.threads for simple cases.
    """
    if num_threads is None:
        return "0:1"
    s = str(num_threads).replace("L", "").strip()
    if s == "":
        return "0:1"
    # integer => "N:1"
    try:
        n = int(s)
        return f"{max(0, n)}:1"
    except Exception:
        pass
    # ":B" => cores:B
    if s.startswith(":") and s[1:].isdigit():
        try:
            import multiprocessing as mp
            c = mp.cpu_count()
        except Exception:
            c = 0
        return f"{c}:{s[1:]}"
    # "A:" => "A:1"
    if s.endswith(":") and s[:-1].isdigit():
        return f"{s}1"
    # "A" => "A:1"
    if s.isdigit():
        return f"{s}:1"
    # else assume already "A:B"
    return s


def _expand_gcpo_from_df(gcpo: Dict[str, Any], df: "pd.DataFrame", ndata: int) -> Dict[str, Any]:
    """
    Expand selected control.compute.control_gcpo fields when they are given
    as DataFrame column names. Validates lengths and types.
    Handles:
      - selection: boolean/int mask or 1-based indices; column name -> indices
      - group.selection: as above
      - weights: numeric vector of length ndata; column name -> vector
      - friends: list of column names is not supported; if given as string of JSON,
                 we ignore; users should pass concrete lists. (Future work)
    Returns a possibly-updated gcpo dict.
    """
    if df is None or (pd is None):
        return gcpo

    out = dict(gcpo or {})

    def _expand_indices(val):
        if val is None:
            return None
        if isinstance(val, str) and val in df.columns:
            col = pd.to_numeric(df[val], errors="coerce").to_numpy()
            if col.size != ndata:
                raise PyINLAError(f"gcpo field '{val}' has length {col.size}, expected {ndata}.")
            # Treat non-zero as selected; produce 1-based indices
            idx = np.where(np.isfinite(col) & (col != 0))[0] + 1
            return idx.tolist()
        # Already a sequence -> pass through
        return val

    def _expand_weights(val):
        if val is None:
            return None
        if isinstance(val, str) and val in df.columns:
            w = pd.to_numeric(df[val], errors="coerce").to_numpy(dtype=float)
            if w.size != ndata:
                raise PyINLAError(f"gcpo weights column '{val}' has length {w.size}, expected {ndata}.")
            w[~np.isfinite(w)] = 0.0
            return w
        arr = np.asarray(val, dtype=float).reshape(-1)
        if arr.size not in (1, ndata):
            raise PyINLAError(f"gcpo weights length {arr.size} must be 1 or {ndata}.")
        if arr.size == 1:
            arr = np.repeat(arr.item(), ndata)
        arr[~np.isfinite(arr)] = 0.0
        return arr

    out['selection'] = _expand_indices(out.get('selection'))
    out['group.selection'] = _expand_indices(out.get('group.selection'))
    w = out.get('weights')
    if w is not None:
        out['weights'] = _expand_weights(w)
    return out


def _validate_families(fams: List[str]) -> None:
    """Validate that each family is known in the model registry (likelihood section)."""
    for fam in fams:
        try:
            props = _MODELS_DB.get_model_properties(fam, "likelihood", stop_on_error=False)
        except Exception:
            props = None
        if not props:
            raise PyINLAError(
                f"Unknown family '{fam}'. Use names(inla.models()$likelihood) in R-INLA or the Python registry to list available families."
            )


def _finalize_and_validate_families(fams: List[str],
                                    control_resolved: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any], int]:
    """
    Mirror R's post-control family handling in one place:
      - Validate that each family is a known likelihood (registry lookup).
      - Populate control.family hyper defaults using the primary family (first entry).
      - Return the families (unchanged), possibly-updated control dict, and n_family.
    """
    _validate_families(fams)
    try:
        fam_sec = control_resolved.get("family")
        if isinstance(fam_sec, dict) and len(fams) > 0:
            _merge_family_hyper_defaults(fam_sec, fams[0])
    except Exception:
        # Do not fail family processing due to optional hyper default population
        pass
    return fams, control_resolved, len(fams)


def _determine_predictor_dimensions(model: Dict[str, Any], df: "pd.DataFrame") -> Tuple[Any, int, int]:
    """
    Determine predictor sizes and normalize the A matrix:
    - Returns (A, m, n) where n is the latent predictor length and m the
      observation length for A @ eta (eta*). If no A is provided, m == 0 and
      n defaults to df.shape[0].
    - Accepts A as dense, scipy.sparse, or list-of-triplets; normalizes via
      _maybe_to_sparse for downstream writers.
    """
    A = None
    m = 0
    n = int(getattr(df, "shape", (0,))[0])
    pred = model.get("predictor", {}) or {}
    if pred.get("A") is not None:
        A = _maybe_to_sparse(pred["A"])
        try:
            import scipy.sparse as sp  # type: ignore
            m, nA = (A.shape if sp.issparse(A) else np.shape(A))
        except Exception:
            m, nA = np.shape(A)
        n = int(nA)
    else:
        m = 0
    return A, int(m), int(n)


def _normalize_per_family_spec(value: Any, n_family: int) -> List[Any]:
    """Normalize a possibly scalar or list value to a per-family list of length n_family.
    If `value` is a list/tuple with length == n_family, return it. Otherwise replicate
    the single value across families. This mirrors the common R usage where scalar or
    shared vectors are reused for each family unless a list is provided.
    """
    if isinstance(value, (list, tuple)) and len(value) == n_family:
        return list(value)
    return [value] * n_family


def _apply_intercept_rule(fixed_terms: List[Any], intercept_spec: Any, n: int) -> List[Any]:
    """Apply R-like intercept handling to the fixed terms list.

    Rules
    - Support tokens '-1'/'0' in fixed list to drop the intercept.
    - If intercept is False: remove any implicit '1'.
    - If intercept is None and no '1' present: inject an intercept '1'.
    - If intercept is a numeric vector of length n: prepend explicit (Intercept) with
      that vector and remove any implicit '1' to avoid duplicates.
    Returns the normalized fixed_terms list.
    """
    # Remove tokens requesting intercept removal
    remove_tokens = {"-1", "0"}
    if any(isinstance(t, str) and t.strip() in remove_tokens for t in fixed_terms):
        fixed_terms = [t for t in fixed_terms if not (isinstance(t, str) and t.strip() in remove_tokens)]
        if intercept_spec is None:
            intercept_spec = False

    if intercept_spec is False:
        # Remove any implicit '1'
        return [t for t in fixed_terms if not (isinstance(t, str) and t.strip() == "1")]

    if intercept_spec is None:
        # Inject intercept if not present
        # Check for "1" string or ('intercept', True) tuple
        def _has_intercept_term(terms):
            for t in terms:
                if isinstance(t, str) and t.strip() == "1":
                    return True
                if isinstance(t, (list, tuple)) and len(t) == 2 and isinstance(t[0], str):
                    name, val = t[0].strip().lower(), t[1]
                    if name == "intercept" and val is True:
                        return True
            return False
        if not _has_intercept_term(fixed_terms):
            return ["1"] + fixed_terms
        return fixed_terms

    # Numeric vector case
    try:
        v = np.asarray(intercept_spec, dtype=float).reshape(-1)
    except Exception as e:  # noqa: BLE001
        raise PyINLAError("intercept must be False, None/True, or a numeric vector") from e
    if v.size != n:
        raise PyINLAError(f"Intercept vector length mismatch: expected {n}, got {v.size}")
    v = np.where(np.isfinite(v), v, 0.0)
    # Prepend explicit intercept and remove any implicit '1'
    cleaned = [t for t in fixed_terms if not (isinstance(t, str) and t.strip() == "1")]
    return [("(Intercept)", v)] + cleaned

def _expand_fixed_matrix(data: Union[pd.DataFrame, Dict[str, Any]],
                         fixed_terms: List[Any],
                         n_predictor: int,
                         *,
                         expand_factor_strategy: str = "model.matrix",
                         has_intercept: bool = True) -> Tuple[np.ndarray, List[str]]:
    """
    Rough equivalent of model.matrix building for fixed effects:
    Accepts a mixture of:
    - "1" to add an intercept
    - string column names (looked up in `data`)
    - array-like values provided directly (np.ndarray, pandas.Series, list/tuple), optionally
      as a pair (name, values). Length must equal `n_predictor`.
    Numeric inputs are used as-is with non-finite set to 0; non-numeric columns are expanded
    with pandas.get_dummies (if pandas is available).
    """
    if len(fixed_terms) == 0:
        return np.zeros((n_predictor, 0), dtype=float), []

    if _is_dataframe_like(data):
        df = data.copy()
    else:
        if pd is None:
            raise PyINLAError("pandas is required for dict->DataFrame conversion of fixed effects.")
        df = pd.DataFrame(data)

    cols: List[str] = []
    blocks: List[np.ndarray] = []
    seen: set[str] = set()

    def _numeric_series(name: str) -> np.ndarray:
        if name not in df.columns:
            raise PyINLAError(f"Fixed term '{name}' not found in data columns.")
        series = df[name]
        try:
            values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        except Exception:
            try:
                values = series.to_numpy(dtype=float)
            except Exception as e:
                raise PyINLAError(f"Fixed term '{name}' is not numeric and cannot be coerced to numeric.") from e
        if values.size != n_predictor:
            raise PyINLAError(f"Fixed term '{name}' length mismatch: expected {n_predictor}, got {values.size}")
        values = np.where(np.isfinite(values), values, 0.0)
        return values

    def _emit_column(label: str, vec: np.ndarray) -> None:
        if label in seen:
            return
        blocks.append(vec.reshape(-1, 1))
        cols.append(label)
        seen.add(label)

    for term in fixed_terms:
        # Intercept (internal marker - injected by _apply_intercept_rule based on 'intercept' key)
        if isinstance(term, str) and term.strip() == "1":
            blocks.append(np.ones((n_predictor, 1), dtype=float))
            cols.append("(Intercept)")
            continue

        # Tuple or list of length 2: (name, values)
        if isinstance(term, (list, tuple)) and len(term) == 2 and isinstance(term[0], str):
            name, values = term[0].strip(), term[1]
            if name == "":
                raise PyINLAError("Fixed term name cannot be empty/blank.")
            # Handle ('intercept', True) or similar - boolean True means column of 1s
            if values is True:
                _emit_column(name if name.lower() != "intercept" else "(Intercept)", np.ones(n_predictor, dtype=float))
                continue
            # Handle ('name', False) - skip this term
            if values is False:
                continue
            arr = np.asarray(values, dtype=float)
            if arr.ndim == 1:
                v = arr.reshape(-1)
                if v.size != n_predictor:
                    raise PyINLAError(f"Fixed term '{name}' length mismatch: expected {n_predictor}, got {v.size}")
                v = np.where(np.isfinite(v), v, 0.0)
                _emit_column(name, v)
                continue
            elif arr.ndim == 2:
                # Accept (n x p) or (p x n); orient so rows == n_predictor
                r, c = arr.shape
                if r == n_predictor:
                    M = arr
                elif c == n_predictor:
                    M = arr.T
                    r, c = M.shape
                else:
                    raise PyINLAError(
                        f"Fixed matrix '{name}' has incompatible shape {arr.shape}; expected (n,p) with n={n_predictor}"
                    )
                M = np.where(np.isfinite(M), M, 0.0)
                for j in range(c):
                    _emit_column(f"{name}{j+1}", M[:, j].reshape(-1))
                continue
            else:
                raise PyINLAError(f"Fixed term '{name}' must be 1D or 2D array-like, got ndim={arr.ndim}")

        # pandas Series: use its name (or auto)
        try:
            import pandas as _pd  # local import for isinstance check
        except Exception:
            _pd = None
        if _pd is not None and isinstance(term, _pd.Series):
            name = term.name or f"fixed_{len(cols)+1}"
            if isinstance(name, str):
                name = name.strip() or f"fixed_{len(cols)+1}"
            v = term.to_numpy(dtype=float).reshape(-1)
            if v.size != n_predictor:
                raise PyINLAError(f"Fixed term '{name}' length mismatch: expected {n_predictor}, got {v.size}")
            v = np.where(np.isfinite(v), v, 0.0).reshape(-1, 1)
            _emit_column(name, v.reshape(-1))
            continue

        # Raw array-like values: no explicit name provided
        if isinstance(term, (np.ndarray,)) or (
            isinstance(term, (list, tuple)) and not (len(term) == 2 and isinstance(term[0], str))
        ):
            v = np.asarray(term, dtype=float).reshape(-1)
            if v.size != n_predictor:
                raise PyINLAError(f"Fixed term vector length mismatch: expected {n_predictor}, got {v.size}")
            v = np.where(np.isfinite(v), v, 0.0).reshape(-1, 1)
            _emit_column(f"fixed_{len(cols)+1}", v.reshape(-1))
            continue

        # Otherwise, expect a string term (name, interaction, or expansion)
        if not isinstance(term, str):
            raise PyINLAError("Fixed term must be '1', a column name, a Series, an array-like, or (name, values).")
        term_clean = term.strip()
        if term_clean == "":
            raise PyINLAError("Fixed term cannot be empty/blank string.")

        # Handle interaction or expansion syntax (':' for interaction, '*' for expansion)
        if (":" in term_clean) or ("*" in term_clean):
            # Variables are split on ':' or '*', both require the base variables to exist
            if "*" in term_clean:
                vars_star = [t.strip() for t in term_clean.split("*") if t.strip()]
                if not vars_star:
                    raise PyINLAError(f"Malformed fixed term '{term_clean}'.")
                # R's x1*x2*... expands to all main effects and all interactions up to full order
                for r in range(1, len(vars_star) + 1):
                    for combo in combinations(vars_star, r):
                        label = ":".join(combo)
                        # product of involved variables
                        vec = None
                        for vname in combo:
                            v = _numeric_series(vname)
                            vec = v if vec is None else (vec * v)
                        _emit_column(label, vec)  # vec is 1d
                continue
            else:
                parts = [t.strip() for t in term_clean.split(":") if t.strip()]
                if len(parts) < 2:
                    raise PyINLAError(f"Malformed interaction '{term_clean}'.")
                vec = None
                for vname in parts:
                    v = _numeric_series(vname)
                    vec = v if vec is None else (vec * v)
                _emit_column(":".join(parts), vec)
                continue

        # Simple column name in df
        if term_clean not in df.columns:
            raise PyINLAError(f"Fixed term '{term_clean}' not found in data columns.")
        series = df[term_clean]
        if pd.api.types.is_numeric_dtype(series):
            v = _numeric_series(term_clean)
            _emit_column(term_clean, v)
        else:
            # categorical expansion
            # R parity: use prefix_sep='' to match R's naming convention (e.g., 'month_factor2' not 'month_factor_2')
            dummies = pd.get_dummies(series, prefix=term_clean, prefix_sep='', dummy_na=False)
            dummies = dummies.fillna(0.0)

            # R parity: sort factor levels numerically if possible, otherwise alphabetically
            # This ensures 'factor2, factor3, ..., factor10, factor11' order instead of 'factor10, factor11, factor2, ...'
            def _sort_key(col_name: str) -> tuple:
                """Sort key that handles numeric suffixes properly."""
                suffix = col_name[len(term_clean):]  # extract the level part after prefix
                try:
                    # Try to parse as integer for numeric sorting
                    return (0, int(suffix), suffix)
                except ValueError:
                    try:
                        # Try to parse as float for numeric sorting
                        return (0, float(suffix), suffix)
                    except ValueError:
                        # Fall back to lexicographic sorting
                        return (1, 0, suffix)

            sorted_cols = sorted(dummies.columns, key=_sort_key)
            dummies = dummies[sorted_cols]

            # R parity: model.matrix-style contrasts drop one level when an intercept is present.
            # Honor control.fixed.expand_factor_strategy: 'model.matrix' -> drop first level;
            # 'inla' -> keep all levels.
            # Key: only drop first level if intercept is present (like R's behavior with -1)
            if str(expand_factor_strategy).strip().lower() == "model.matrix" and has_intercept:
                if dummies.shape[1] > 0:
                    dummies = dummies.iloc[:, 1:]
            if dummies.shape[0] != n_predictor:
                raise PyINLAError(
                    f"Categorical fixed term '{term_clean}' length mismatch: expected {n_predictor}, got {dummies.shape[0]}"
                )
            # ensure unique columns
            for cname in dummies.columns:
                _emit_column(cname, dummies[cname].to_numpy(dtype=float).reshape(-1))

    if len(blocks) == 0:
        return np.zeros((n_predictor, 0), dtype=float), []
    X = np.concatenate(blocks, axis=1)
    return X, cols

def _as_index_vector(x: Any, n: int) -> np.ndarray:
    """
    Convert a replicate/group vector to integer index (1..K), NA where missing.
    """
    arr = np.asarray(x)
    if arr.shape[0] != n:
        raise PyINLAError(f"Length mismatch: expected {n}, got {arr.shape[0]}")
    # Fill NA with 1 as R code does when idx is NA
    arr = np.where(np.isfinite(arr), arr, 1)
    arr = arr.astype(int)
    arr[arr < 1] = 1
    return arr

def _normalize_family(fam: Union[str, List[str]]) -> List[str]:
    """Normalize user-supplied family names.

    Accepts a single family (string) or a list of families and returns
    a list of canonical, lowercased family identifiers. Whitespace is
    trimmed and common aliases are mapped to the canonical INLA names:

    - "normal" -> "gaussian"
    - "stdnormal" -> "stdgaussian"
    - "bcnormal" -> "bcgaussian"
    - "gennormal"/"gengaussian" -> "exppower"

    If the input is None or empty, an empty list is returned; the caller
    is responsible for applying the default (typically ["gaussian"]).
    """
    fams = _as_list(fam)
    # alias mapping as in the R code:
    alias = {
        "normal": "gaussian",
        "stdnormal": "stdgaussian",
        "bcnormal": "bcgaussian",
        "gennormal": "exppower",
        "gengaussian": "exppower",
    }
    out = []
    for f in fams:
        f2 = f.strip().lower()
        out.append(alias.get(f2, f2))
    return out

def _write_offset_file(offset_formula: np.ndarray,
                       offset_obs: Optional[np.ndarray],
                       A: Optional[Any],
                       n: int, m: int,
                       data_dir: str,
                       preferred_name: Optional[str] = None) -> str:
    """
    Implement the R-logic:
      eta* = A eta + offset_obs;  eta = ... + offset_formula
    For INLA, we write an index-value table either for n or for (m+n).
    """
    indN = np.arange(n, dtype=np.int64)
    if A is not None:
        indM = np.arange(m, dtype=np.int64)
        if offset_obs is None:
            offset_obs = np.zeros(m, dtype=float)
        x_top = np.column_stack([indM, (A @ offset_formula).reshape(-1) + offset_obs.reshape(-1)])
        x_bot = np.column_stack([m + indN, offset_formula.reshape(-1)])
        off = np.vstack([x_top, x_bot])
    else:
        off = np.column_stack([indN, (offset_formula if offset_obs is None else offset_formula + 0.0)])
    name = preferred_name or f"offset-{random.randint(10**8, 10**9-1)}.dat"
    path = os.path.join(data_dir, name)
    write_fmesher_file(off, path)
    # Return with $inladatadir prefix so diffs are path-stable
    try:
        path_masked = path.replace(os.path.abspath(data_dir), "$inladatadir")
    except Exception:
        path_masked = path
    return path_masked

def _maybe_to_sparse(A: Any):
    """
    Normalize `A` to something write_fmesher_file can handle:
    - numpy ndarray dense
    - scipy.sparse matrix
    - list of (i, j, x) triplets (0- or 1-based tolerated)
    """
    try:
        import scipy.sparse as sp
        if sp.issparse(A):
            return A
    except Exception:
        pass
    if isinstance(A, np.ndarray):
        return A
    if isinstance(A, list):
        # convert triplets to a tiny COO
        try:
            import scipy.sparse as sp
            if len(A) == 0:
                return np.zeros((0, 0))
            I = np.array([t[0] for t in A], dtype=int)
            J = np.array([t[1] for t in A], dtype=int)
            X = np.array([t[2] for t in A], dtype=float)
            # Fix to 0-based
            if I.min() == 1 or J.min() == 1:
                I -= 1
                J -= 1
            shape = (int(I.max())+1, int(J.max())+1)
            return sp.coo_matrix((X, (I, J)), shape=shape)
        except Exception:
            # fallback dense
            raise
    return A


def _sanitize_A_matrix(A: Any):
    """
    R-INLA analogue of inla.as.sparse(A, na.rm=TRUE, zeros.rm=TRUE) for simple cases:
    - Converts list-of-triplets to COO, handling 1-based indices.
    - Replaces NaN/Inf with zeros.
    - Removes explicit zeros from sparse representations.
    Dense matrices are returned with NaN/Inf replaced by zeros.
    """
    try:
        import scipy.sparse as sp
        HAVE_SP = True
    except Exception:
        HAVE_SP = False

    if HAVE_SP and sp.issparse(A):
        M = A.tocoo(copy=True)
        data = np.asarray(M.data, dtype=float)
        data = np.where(np.isfinite(data), data, 0.0)
        nz = data != 0.0
        return sp.coo_matrix((data[nz], (M.row[nz], M.col[nz])), shape=M.shape)

    if isinstance(A, np.ndarray):
        B = np.asarray(A, dtype=float)
        B[~np.isfinite(B)] = 0.0
        return B

    if isinstance(A, list):
        if len(A) == 0:
            return np.zeros((0, 0))
        I = np.array([t[0] for t in A], dtype=int)
        J = np.array([t[1] for t in A], dtype=int)
        X = np.array([t[2] for t in A], dtype=float)
        X = np.where(np.isfinite(X), X, 0.0)
        keep = X != 0.0
        I, J, X = I[keep], J[keep], X[keep]
        if I.size and (I.min() == 1 or J.min() == 1):
            I -= 1
            J -= 1
        shape = (int(I.max())+1 if I.size else 0, int(J.max())+1 if J.size else 0)
        try:
            import scipy.sparse as sp
            return sp.coo_matrix((X, (I, J)), shape=shape)
        except Exception:
            dense = np.zeros(shape, dtype=float)
            if I.size:
                dense[I, J] = X
            return dense

    return A


def _process_control_sections(control: Dict[str, Any],
                              model: Dict[str, Any],
                              df: "pd.DataFrame") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Process control.* sections similar to R's ctrl_object + A sanitization.
    Currently focuses on:
      - control['predictor']['A']: sanitize like inla.as.sparse(na.rm=TRUE, zeros.rm=TRUE)
        and, if model['predictor']['A'] is missing, adopt it for dimensioning.
    Returns (control_updated, model_updated).
    """
    updated = dict(control or {})
    predictor = dict(updated.get("predictor", {}) or {})
    A_ctrl = predictor.get("A", None)
    if A_ctrl is not None:
        try:
            A_sane = _sanitize_A_matrix(A_ctrl)
        except Exception:
            A_sane = A_ctrl
        predictor["A"] = A_sane
        updated["predictor"] = predictor

        # If model has no A, adopt the sanitized A so we can determine (m, n)
        mp = dict(model.get("predictor", {}) or {})
        if mp.get("A") is None:
            mp["A"] = A_sane
            model = dict(model)
            model["predictor"] = mp

    return updated, model

def _expand_predictor_cross(cross_val: Any, n: int, m: int, df: "pd.DataFrame") -> Optional[np.ndarray]:
    """Expand control.predictor['cross'] allowing references to DataFrame columns.
    Accepts:
      - a single column name (str) -> df[col].
      - list/tuple of column names -> stacked into a single vector by priority of first non-null? Here we 
        conservatively take the first provided column; users can precompute externally for complex logic.
      - a numeric array-like of length n+m -> returned as-is.
    Returns a 1D numpy array or None if expansion is not applicable.
    """
    total = n + m
    if cross_val is None:
        return None
    # Single string column name
    if isinstance(cross_val, str) and (pd is not None) and (cross_val in df.columns):
        vec = pd.to_numeric(df[cross_val], errors="coerce").to_numpy()
        if vec.size == n:
            # Predictor-only cross; extend with zeros for observation-part if A present
            if m > 0:
                vec = np.concatenate([np.zeros(m, dtype=vec.dtype), vec])
        if vec.size != total:
            raise PyINLAError(f"control.predictor['cross'] column '{cross_val}' has length {vec.size}, expected {total} (n+m).")
        return vec
    # List of column names -> pick first existing
    if isinstance(cross_val, (list, tuple)) and all(isinstance(x, str) for x in cross_val):
        for name in cross_val:
            if (pd is not None) and (name in df.columns):
                return _expand_predictor_cross(name, n, m, df)
        # fall through if none matched
    # Numeric array-like provided directly
    try:
        arr = np.asarray(cross_val).reshape(-1)
        if arr.size == total:
            return arr
    except Exception:
        pass
    return None


def _prepare_coxph_flow(model: Dict[str, Any],
                        fams: List[str],
                        df: "pd.DataFrame",
                        control_resolved: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], Optional[np.ndarray]]:
    """
    Minimal Cox PH -> Poisson hazard flow, expecting the caller to supply
    a pre-expanded data frame with columns from the Cox-to-Poisson conversion.

    Required columns in `df`:
      - 'E..coxph': exposure vector for Poisson likelihood
      - 'baseline.hazard': integer index (1..K) for baseline hazard effect

    Optional columns:
      - 'baseline.hazard.strata.coding': replicate coding for stratified baseline hazards

    Effects on inputs:
      - family is switched to 'poisson'
      - E is populated from df['E..coxph']
      - A baseline hazard random component is added to model['random'] using
        control.hazard parameters (model, constr, diagonal, scale.model)
    """
    if not fams or fams[0] != "coxph":
        return model, fams, None
    # Validate expected columns; if missing, try a minimal auto-expansion using
    # 'time' and 'event' columns (dense rank for baseline.hazard, unit exposure).
    required_cols = ["E..coxph", "baseline.hazard"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        # Attempt a minimal expansion from common survival columns
        if ("time" in df.columns) and ("event" in df.columns):
            t = pd.to_numeric(df["time"], errors="coerce").to_numpy()
            if not np.all(np.isfinite(t)):
                raise PyINLAError("Non-finite values in 'time' are not allowed for Cox expansion.")
            # dense rank of time into 1..K
            uniq = np.unique(t)
            mapping = {v: i+1 for i, v in enumerate(uniq)}
            bh = np.array([mapping[v] for v in t], dtype=int)
            df = df.copy()
            df["baseline.hazard"] = bh
            df["E..coxph"] = 1.0
            # Optional strata -> replicate coding
            for strat_col in ("strata", "strata.name", "baseline.hazard.strata"):
                if strat_col in df.columns:
                    s = pd.Series(df[strat_col]).astype("category")
                    df["baseline.hazard.strata.coding"] = s.cat.codes.to_numpy()
                    break
        else:
            raise PyINLAError(
                "Cox PH hazard flow requires pre-expanded data with columns "
                "['E..coxph','baseline.hazard'] or raw survival columns ['time','event'] to auto-expand."
            )
    E_vec = pd.to_numeric(df["E..coxph"], errors="coerce").to_numpy()
    if not np.all(np.isfinite(E_vec)):
        raise PyINLAError("Non-finite values in E..coxph are not allowed.")

    # Compose a baseline hazard random component
    hazard_ctrl = (control_resolved.get("hazard") or {})
    hz_model = str(hazard_ctrl.get("model", "rw1")).lower()
    comp: Dict[str, Any] = {
        "id": "baseline.hazard",
        "model": hz_model,
    }
    if "constr" in hazard_ctrl:
        comp["constr"] = bool(hazard_ctrl.get("constr"))
    if "diagonal" in hazard_ctrl and hazard_ctrl.get("diagonal") is not None:
        comp["diagonal"] = hazard_ctrl.get("diagonal")
    # Both spellings accepted
    sc = hazard_ctrl.get("scale.model", hazard_ctrl.get("scale_model", None))
    if sc is not None:
        comp["scale.model"] = bool(sc)

    # If a replicate coding was generated, wire it as 'replicate'
    if "baseline.hazard.strata.coding" in df.columns:
        comp["replicate"] = "baseline.hazard.strata.coding"

    # Inject/append into model['random']
    rnd = _as_list(model.get("random", []))
    rnd = list(rnd) if isinstance(rnd, list) else [rnd]
    rnd.append(comp)
    model = dict(model)
    model["random"] = rnd

    # Switch family to Poisson
    fams2 = ["poisson"] + fams[1:]
    return model, fams2, E_vec


# -----------------------
# Result container
# -----------------------
@dataclass
class PyINLAResult:
    cpu_used: Dict[str, float]
    call: str
    model_matrix: Optional[np.ndarray]
    args: Dict[str, Any]
    inla_dir: str
    all_hyper: Dict[str, Any]
    logfile: Optional[List[str]] = None
    results: Optional[Dict[str, Any]] = None
    collector_error: Optional[str] = None
    # These fields mirror your R “inla” object when available via collection.
    summary_fixed: Optional[Any] = None
    marginals_fixed: Optional[Any] = None
    summary_random: Optional[Any] = None
    marginals_random: Optional[Any] = None
    summary_hyperpar: Optional[Any] = None
    marginals_hyperpar: Optional[Any] = None
    summary_linear_predictor: Optional[Any] = None
    marginals_linear_predictor: Optional[Any] = None
    summary_fitted_values: Optional[Any] = None
    marginals_fitted_values: Optional[Any] = None
    summary_lincomb: Optional[Any] = None
    marginals_lincomb: Optional[Any] = None
    selection: Optional[Any] = None
    dic: Optional[Any] = None
    cpo: Optional[Any] = None
    po: Optional[Any] = None
    residuals: Optional[Any] = None
    waic: Optional[Any] = None
    mlik: Optional[Any] = None
    neffp: Optional[Any] = None
    mode: Optional[Any] = None
    formula: Optional[Any] = None
    nhyper: Optional[int] = None

    # Convenience helpers so the result can be used like the old dict API
    def _ensure_results(self) -> Dict[str, Any]:
        if self.results is None:
            raise KeyError(
                "No collected INLA results are attached to this object. "
                "Run with collect=True (default) or call collect_inla_results manually."
            )
        return self.results

    def __getitem__(self, key: str) -> Any:
        return self._ensure_results()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._ensure_results().get(key, default)

    def keys(self) -> Iterable[str]:
        return self._ensure_results().keys()

    def items(self):
        return self._ensure_results().items()

    def values(self):
        return self._ensure_results().values()

    def __contains__(self, key: object) -> bool:
        try:
            return key in self._ensure_results()
        except KeyError:
            return False

    @property
    def directory(self) -> str:
        """Absolute path of the working directory used for this run."""
        return self.inla_dir


# -----------------------
# Core runner (like inla.core in R)
# -----------------------
def _core(*,
          model: Dict[str, Any],
          family: Union[str, List[str]],
          data: Union[Dict[str, Any], "pd.DataFrame", None] = None,
          quantiles: List[float] = [0.025, 0.5, 0.975],
          E: Optional[Any] = None,
          offset: Optional[Any] = None,
          scale: Optional[Any] = None,
          weights: Optional[Any] = None,
          Ntrials: Optional[Any] = None,
          strata: Optional[Any] = None,
          lp_scale: Optional[Any] = None,
          link_covariates: Optional[Any] = None,
          verbose: bool = False,
          lincomb: Optional[Any] = None,
          selection: Optional[Any] = None,
          control: Optional[Dict[str, Any]] = None,
          inla_call: Optional[str] = None,
          inla_arg: Optional[str] = None,
          num_threads: Union[str, int] = "0:1",
          keep: bool = False,
          working_directory: Optional[str] = None,
          silent: Union[int, bool] = 0,
          inla_mode: str = "compact",
          debug: bool = False,
          reuse_filenames_from: Optional[str] = None,
          dry_run: bool = False,
          collect_results: bool = True) -> PyINLAResult:

    # --- Start timing
    t0 = time.time()

    # --- Normalize inputs
    # Input normalization rules:
    # - Data:
    #   * If a pandas DataFrame: use as-is.
    #   * Else pandas must be available.
    #   * If data is None and model['response'] is array-like: build df with a single
    #     column 'y' and set model['response'] = 'y' for downstream consistency.
    #   * If data is None and model['response'] is a string (column name): error, since
    #     there is no DataFrame to resolve the name.
    #   * If data is a dict: convert to a DataFrame.
    # Normalize/construct the working DataFrame 'df' and possibly the model
    df, model = _normalize_data_and_model(model, data)

    # --- Reference + controls
    # Resolve the parent directory that will host this run. When the caller
    # does not provide one we default to the installed package location so
    # `pip install pyinla` users automatically leverage the bundled runtime.
    working_directory_resolved = str(_resolve_work_parent(working_directory))
    working_directory = working_directory_resolved

    # Reference reuse (optional):
    # - Resolve a reference model folder (explicit `reuse_filenames_from`, or discover recent `inla.model*`).
    # - Parse its Model.ini to reuse only stable file names (e.g., data.files entries, offset, fixed/random files)
    #   for deterministic diffs. Content is freshly written; no old results are read.
    reference_path = _resolve_reference_path(reuse_filenames_from, working_directory)
    reference_names = _parse_reference_model_ini(reference_path)

    # Control defaults (R parity with ctrl_object):
    # - Merge user `control.*` trees with defaults (inla, fixed, mode, expert, hazard, lp_scale,
    #   pardiso, stiles, taucs, numa, predictor, compute, family).
    # - Enable predictor.compute when diagnostics/links imply it.
    control_resolved = _apply_control_defaults(control, primary_family=None)
    _normalize_control_priors(control_resolved)
    _ensure_predictor_compute(control_resolved)

    # R-like control processing:
    # - Sanitize control.predictor['A'] (NaN/Inf->0, drop explicit zeros in sparse) and adopt into
    #   model.predictor['A'] if missing so sizing (m,n) is well-defined.
    # - Perform targeted expansion (e.g., predictor['cross'] from DataFrame columns) later before writing.
    control_resolved, model = _process_control_sections(control_resolved, model, df)

    # Absolute path to data.files (used for path masking when writing files); set after creating subdirs
    data_dir_abs = None  # will be set after _mk_subdirs

    # --- Family handling (R: after ctrl_object)
    # Normalize aliases and default to ['gaussian'] when empty. Multiple families supported.
    fams = _normalize_family(family)
    if len(fams) == 0:
        fams = ["gaussian"]

    # --- Cox PH hazard flow (optional)
    # If the user requests family='coxph' and provides a pre-expanded data frame,
    # switch to Poisson, add baseline hazard component, and populate E from data.
    E_override: Optional[np.ndarray] = None
    try:
        model, fams, E_override = _prepare_coxph_flow(model, fams, df, control_resolved)
    except PyINLAError:
        raise
    except Exception as _ex:
        raise PyINLAError(f"Failed to prepare Cox PH hazard flow: {_ex}")

    # --- Family finalize + validate (R parity)
    # - Validate families via the registry and populate control.family hyper defaults
    #   using the primary family. Returns (fams, control_resolved, n_family).
    fams, control_resolved, n_family = _finalize_and_validate_families(fams, control_resolved)

    # --- Determine predictor lengths and A
    A, m, n = _determine_predictor_dimensions(model, df)

    # --- Response
    response = model.get("response", None)
    if response is None:
        raise PyINLAError("model['response'] must be set to the response column name.")
    if isinstance(response, str):
        if response not in df.columns:
            raise PyINLAError(f"Response '{response}' not found in data.")
        y = df[response].to_numpy()
        if m > 0 and len(y) != m:
            raise PyINLAError(f"Length of response (len={len(y)}) must equal rows of A (m={m}).")
    # Lists/arrays are validated later when building per-family data sections

    NData = m if (m > 0) else n
    if NData <= 0:
        raise PyINLAError("No data rows to fit.")

    # --- Build fixed effects design
    # TODO(pyINLA parity roadmap):
    # - Formula functions: support a safe subset of R's formula helpers in fixed terms,
    #   e.g., I(expr) for arithmetic (x**2, log(x), exp(x)), poly(x, d, raw=TRUE), and
    #   optionally spline bases (bs/ns) by delegating to a design-matrix builder (e.g., patsy)
    #   or precomputing columns. Guard with an opt-in flag for safety.
    # - Contrasts: allow per-factor contrast specifications akin to R, in addition to the
    #   global control.fixed.expand_factor_strategy ('model.matrix' vs 'inla').
    # - Stacking: add a lightweight helper to emulate inla.stack for multi-likelihood inputs
    #   with different observation counts, constructing block-diagonal A and concatenated y.
    # - Control expansion: consider targeted column-name expansion for more control fields
    #   beyond predictor.cross and compute.control_gcpo (e.g., link-covariates per family),
    #   with strict length and type validation.
    # - Cox PH: complete the internal Cox→Poisson expansion functions in coxph.py to avoid
    #   requiring pre-expanded data (implement inla_expand_dataframe_1/2).
    # - Per-family control: support per-family control.family overrides when multiple families
    #   are present, instead of one shared family control block.
    fixed_terms = _as_list(model.get("fixed", []))
    # Intercept handling (R parity): tokens '-1'/'0' drop intercept; default include;
    # vector intercept allowed when length == n.
    fixed_terms = _apply_intercept_rule(fixed_terms, model.get("intercept", None), n)

    # Honor control.fixed.expand_factor_strategy ('model.matrix' | 'inla') for factor encoding
    try:
        expand_strategy = str(control_resolved.get("fixed", {}).get("expand_factor_strategy", "model.matrix")).strip().lower()
    except Exception:
        expand_strategy = "model.matrix"
    # Detect if intercept is present (for R-parity categorical encoding)
    # Intercept appears as '1' string or ('(Intercept)', vector) tuple after _apply_intercept_rule
    has_intercept = any(
        (isinstance(t, str) and t.strip() == "1") or
        (isinstance(t, tuple) and len(t) >= 1 and t[0] == "(Intercept)")
        for t in fixed_terms
    )
    X, fixed_cols = _expand_fixed_matrix(df, fixed_terms, n, expand_factor_strategy=expand_strategy, has_intercept=has_intercept)

    # --- Build random effects definitions (like f(...) in R)
    # Format: [{'id': 'idx', 'model': 'iid'}, ...]
    random_specs: List[Dict[str, Any]] = _as_list(model.get("random", []))

    # --- Working directories / ini file
    inla_dir = _ensure_workdir(keep, working_directory)
    data_dir, results_dir, file_ini = _mk_subdirs(inla_dir)
    data_dir_abs = os.path.abspath(data_dir)

    # --- Problem section (like inla.problem.section)
    compute = control_resolved.get("compute", {})
    compute_user_keys = compute.get("__user_keys__", ())
    if isinstance(compute_user_keys, (list, tuple, set)):
        compute_user_keys = set(compute_user_keys)
    else:
        compute_user_keys = {str(compute_user_keys)} if compute_user_keys else set()
    internal_opt_val = compute.get("internal_opt", None)
    if internal_opt_val is not None:
        internal_opt_val = bool(internal_opt_val)

    save_memory_val = compute.get("save_memory", None)
    if save_memory_val is not None:
        save_memory_val = bool(save_memory_val)

    # Expand gcpo fields from DataFrame columns if applicable
    try:
        gcpo_raw = compute.get("control_gcpo", {}) or {}
        NData = (m if (m > 0) else n)
        gcpo_exp = _expand_gcpo_from_df(gcpo_raw, df, NData)
        compute = dict(compute)
        compute["control_gcpo"] = gcpo_exp
    except Exception as _ex:
        raise PyINLAError(f"Failed to expand control.compute.control_gcpo: {_ex}")

    likelihood_info_user = any(
        key in compute_user_keys for key in ("likelihood_info", "likelihood.info")
    )
    likelihood_info_arg = bool(compute.get("likelihood_info", False)) if likelihood_info_user else None

    problem_section(
        file=file_ini,
        data_dir=data_dir,
        result_dir=results_dir,
        hyperpar=bool(compute.get("hyperpar", True)),
        return_marginals=bool(compute.get("return_marginals", True)),
        return_marginals_predictor=bool(compute.get("return_marginals_predictor", False)),
        dic=bool(compute.get("dic", False)),
        cpo=bool(compute.get("cpo", False)),
        gcpo=(compute.get("control_gcpo", {"enable": False})),
        po=bool(compute.get("po", False) or compute.get("waic", False)),
        mlik=bool(compute.get("mlik", False)),
        quantiles=list(quantiles),
        smtp=compute.get("smtp", "default"),
        q=bool(compute.get("q", False)),
        openmp_strategy=compute.get("openmp_strategy", "default"),
        graph=bool(compute.get("graph", False)),
        config=compute.get("config", False),  # Match R-INLA default
        likelihood_info=likelihood_info_arg,
        internal_opt=internal_opt_val,
        save_memory=save_memory_val
    )

    # --- Data sections per family
    # Build data/weights/attr/lp.scale files; support multiple families with
    # a list form of model['response'] (strings or vectors). If a single
    # response is provided, it is reused for all families.
    fam = fams[0]
    if _create_data_file is None:
        raise PyINLAError("create_data_file module is missing; cannot build likelihood input files.")

    # Prepare responses for each family (broadcast if a single response was given)
    responses_spec = model.get("response", None)
    resp_list = _normalize_per_family_spec(responses_spec, n_family)

    # Supported survival families
    _ALLOWED_SURVIVAL_FAMILIES = {
        "exponentialsurv",
        "gammasurv",
        "lognormalsurv",
        "loglogisticsurv",
        "weibullsurv",
    }

    def _get_y(spec, family_name: str):
        # Pass through inla_surv objects for survival families
        if _is_inla_surv(spec):
            if family_name not in _ALLOWED_SURVIVAL_FAMILIES:
                raise PyINLAError(
                    f"Survival response (inla_surv) requires a survival family. "
                    f"Got '{family_name}', expected one of: {_ALLOWED_SURVIVAL_FAMILIES}"
                )
            return spec  # Return as dict for create_data_file
        if isinstance(spec, str):
            if spec not in df.columns:
                raise PyINLAError(f"Response '{spec}' not found in data.")
            return df[spec].to_numpy()
        return np.asarray(spec).reshape(-1)
    # Normalize per-family likelihood inputs (replicate shared values)
    if E_override is not None:
        E_list = [E_override] * n_family
    else:
        E_list = _normalize_per_family_spec(E, n_family)
    Ntrials_list = _normalize_per_family_spec(Ntrials, n_family)
    scale_list = _normalize_per_family_spec(scale, n_family)
    weights_list = _normalize_per_family_spec(weights, n_family)
    strata_list = _normalize_per_family_spec(strata, n_family)
    lp_scale_list = _normalize_per_family_spec(lp_scale, n_family)

    for i in range(n_family):
        fam_i = fams[i]
        y_i = _get_y(resp_list[i], fam_i)
        # Family-specific likelihood additions
        created = _create_data_file(
            y_orig=y_i,
            E=E_list[i],
            Ntrials=Ntrials_list[i],
            scale=scale_list[i],
            weights=weights_list[i],
            strata=strata_list[i],
            lp_scale=lp_scale_list[i],
            family=fam_i,
            data_dir=data_dir,
            debug=bool(control_resolved.get("debug", False)),
            reuse_names=reference_names.get("data"),
        )

        data_section(
            file=file_ini,
            family=fam_i,
            file_data=created["file.data"],
            file_weights=created["file.weights"],
            file_attr=created["file.attr"],
            file_lp_scale=created["file.lp.scale"],
            control=control_resolved.get("family", {}),
            i_family=str(i+1),
            link_covariates=link_covariates,
            data_dir=data_dir,
        )

    # --- Predictor section
    # Offsets: support providing them in the model dict (preferred) or as a top-level arg for backward compat
    # - model['offset'] or model['offset_formula']: length-n vector added at predictor level
    # - model['offset_obs']: length-m vector added at observation level (only meaningful when A is provided)
    # - `offset` (function arg): treated as observation-level if A is present; if A is None and length==n, it is folded into offset_formula
    offset_formula = np.zeros(n, dtype=float)
    model_offset_formula = None
    if isinstance(model, dict):
        if 'offset' in model:
            model_offset_formula = model.get('offset')
        elif 'offset_formula' in model:
            model_offset_formula = model.get('offset_formula')
    if model_offset_formula is not None:
        arr = np.asarray(model_offset_formula, dtype=float).reshape(-1)
        if arr.size == 1:
            offset_formula = np.full(n, float(arr[0]), dtype=float)
        elif arr.size == n:
            offset_formula = np.where(np.isfinite(arr), arr, 0.0)
        else:
            raise PyINLAError(f"model['offset'] length mismatch: expected {n}, got {arr.size}")

    # Observation-level offset
    offset_obs = None
    model_offset_obs = None
    if isinstance(model, dict) and ('offset_obs' in model):
        model_offset_obs = model.get('offset_obs')
    if model_offset_obs is not None:
        offset_obs = np.asarray(model_offset_obs, dtype=float).reshape(-1)
    elif offset is not None:
        offset_obs = np.asarray(offset, dtype=float).reshape(-1)

    # Validate/merge offsets with respect to A
    if A is not None:
        if offset_obs is not None and len(offset_obs) != m:
            raise PyINLAError(f"offset length ({len(offset_obs)}) must equal rows of A (m={m}).")
    else:
        # No A: observation-level offsets (if provided) must align with n and are folded into the formula-level offset
        if offset_obs is not None:
            if len(offset_obs) != n:
                raise PyINLAError(f"offset length ({len(offset_obs)}) must equal number of observations (n={n}).")
            # Fold into predictor-level offset and clear obs offset
            offset_formula = offset_formula + np.where(np.isfinite(offset_obs), offset_obs, 0.0)
            offset_obs = None

    offset_pref = reference_names.get("predictor", {}).get("offset")
    offset_path = _write_offset_file(offset_formula, offset_obs, A, n=n, m=m, data_dir=data_dir, preferred_name=offset_pref)
    try:
        if isinstance(offset_path, str) and offset_path.startswith(data_dir_abs):
            offset_path = offset_path.replace(data_dir_abs, "$inladatadir")
    except Exception:
        pass

    # Push A into predictor_spec (the writer expects A inside the spec, not as arg)
    pred_spec = dict(control_resolved.get("predictor", {}))
    if A is not None:
        pred_spec["A"] = A
    # Ensure compute=1 like R when unspecified
    pred_spec.setdefault("compute", True)
    # Default predictor hyper to R-like baseline if missing
    if not pred_spec.get("hyper"):
        pred_spec["hyper"] = [{
            "initial": 13.81551,
            "fixed": True,
            "hyperid": 53001,
            "prior": "loggamma",
            "param": [1.0, 1e-05],
            "to.theta": "function (x) <<NEWLINE>>log(x)",
            "from.theta": "function (x) <<NEWLINE>>exp(x)",
        }]

    # Expand cross-constraint if referenced by DataFrame columns
    try:
        if "cross" in pred_spec and pd is not None:
            xc = _expand_predictor_cross(pred_spec.get("cross"), n=n, m=m, df=df)
            if xc is not None:
                pred_spec["cross"] = xc
    except Exception as _ex:
        raise PyINLAError(f"Failed to expand control.predictor['cross']: {_ex}")

    # Handle link parameter for fitted values (R-INLA: control.predictor$link)
    # link specifies which link function to use for NA observations when computing fitted values
    # Values: 0/NA = identity, 1 = likelihood default link (e.g., log for Poisson), etc.
    file_link_fitted_values = None
    link_param = pred_spec.get("link")
    if link_param is not None:
        # Setting link implies compute=TRUE
        pred_spec["compute"] = True
        n_predictor = n
        m_predictor = m if A is not None else 0
        total_predictor = n_predictor + m_predictor

        # Convert link to array
        link_arr = np.atleast_1d(link_param).astype(float)

        # Handle single value (replicate for all observations)
        if link_arr.size == 1:
            link_arr = np.repeat(link_arr[0], n_predictor)

        # Validate length
        if link_arr.size != n_predictor and link_arr.size != total_predictor:
            raise PyINLAError(
                f"control.predictor['link'] length ({link_arr.size}) must equal "
                f"n={n_predictor} or n+m={total_predictor}."
            )

        # Build (index, link_value - 1) pairs like R-INLA
        if A is not None:
            if link_arr.size == m_predictor:
                # Only m values provided, pad with NA for n
                tlink = np.column_stack([
                    np.arange(total_predictor, dtype=np.int64),
                    np.concatenate([link_arr - 1, np.full(n_predictor, np.nan)])
                ])
            else:
                tlink = np.column_stack([
                    np.arange(total_predictor, dtype=np.int64),
                    link_arr - 1
                ])
        else:
            ind_n = np.arange(n_predictor, dtype=np.int64)
            tlink = np.column_stack([ind_n, link_arr - 1])

        # Write binary file
        link_file_path = os.path.join(data_dir, "link-fitted-values.dat")
        write_fmesher_file(tlink, link_file_path)
        file_link_fitted_values = link_file_path.replace(data_dir_abs, "$inladatadir")

    predictor_section(
        file=file_ini,
        n=n,
        m=m,
        predictor_spec=pred_spec,
        file_offset=offset_path,
        data_dir=data_dir,
        file_link_fitted_values=file_link_fitted_values
    )

    # --- Fixed effects => linear sections
    all_hyper: Dict[str, Any] = {"fixed": []}
    fixed_refs = reference_names.get("fixed", [])
    if X.shape[1] > 0:
        for j, col_name in enumerate(fixed_cols, start=1):
            # write (index, value) pairs for this column
            ind = np.arange(n, dtype=np.int64)
            col = X[:, j-1].astype(float)
            table = np.column_stack([ind, col])
            ref_name = fixed_refs[j-1] if j-1 < len(fixed_refs) else f"fixed-{j}.dat"
            fixed_abs = os.path.join(data_dir, ref_name)
            write_fmesher_file(table, fixed_abs)
            fixed_path = fixed_abs.replace(data_dir_abs, "$inladatadir")

            hyper = linear_section(
                file=file_ini,
                file_fixed=fixed_path,
                label=col_name,
                results_dir=f"fixed.effect{j:010d}",
                control_fixed=control_resolved.get("fixed", {}),
                only_hyperparam=bool(control_resolved.get("only_hyperparam", False))
            )
            all_hyper["fixed"].append(hyper)

    # --- Random effects => ffield sections
    if len(random_specs) > 0:
        all_hyper["random"] = []
        random_refs = reference_names.get("random", [])
        linear_count = 0

        def _get_linear_param(spec: Dict[str, Any], key: str) -> Any:
            if key in spec and spec[key] is not None:
                return spec[key]
            alt = key.replace('.', '_')
            return spec.get(alt)

        for ridx, r in enumerate(random_specs, start=1):
            # Check for SPDE object (marked by _safety.py)
            if r.get("__spde_object__"):
                # SPDE2PcMatern object - extract and convert to spde2.prefix format
                spde_model = r.get("model")
                if spde_model is None:
                    raise PyINLAError(f"Random component {ridx}: SPDE model object is None.")

                # Create temp directory for SPDE files
                spde_dir = os.path.join(data_dir, f"spde-{ridx}")
                os.makedirs(spde_dir, exist_ok=True)
                spde_prefix = os.path.join(spde_dir, "spde")

                # Write SPDE files
                spde_model.write_spde_files(spde_prefix)

                # Convert the spec to spde2 format
                r = dict(r)  # Make a copy to avoid modifying original
                r["model"] = "spde2"
                r["spde2.prefix"] = spde_prefix
                r["spde2.transform"] = "identity"
                r["n"] = spde_model.n_spde

                # Set PC-Matern hyperparameters
                # PC-Matern prior parameters: (lambda_r * r0, lambda_s, alpha)
                # where lambda_r = -log(p_r) / r0 and lambda_s = -log(p_s) / s0
                import math as _math_local
                prior_range = getattr(spde_model, 'prior_range', (1.0, 0.5))
                prior_sigma = getattr(spde_model, 'prior_sigma', (1.0, 0.5))
                r0, p_r = prior_range
                s0, p_s = prior_sigma
                alpha = 2  # SPDE2 uses alpha=2 (Matern with nu=1)

                # pcmatern parameters: (lambda_r * r0, lambda_s, alpha)
                pcmatern_param1 = -_math_local.log(p_r) * r0
                pcmatern_param2 = -_math_local.log(p_s) / s0
                pcmatern_param3 = alpha

                # Initial values for theta (log-range and log-sigma)
                # R-INLA formula from spde2.R lines 1251-1262:
                #   initial.range = log(prior.range[1]) + 1
                #   initial.sigma = log(prior.sigma[1]) - 1
                initial_theta0 = _math_local.log(r0) + 1.0
                initial_theta1 = _math_local.log(s0) - 1.0

                # Override hyperparameters to use pcmatern prior
                # SPDE2 uses identity transformations (theta IS the internal parameter)
                # and specific hyperid values for proper tracking
                r["hyper"] = [
                    {
                        "name": "theta1",
                        "short_name": "Range",
                        "initial": initial_theta0,
                        "fixed": False,
                        "hyperid": 23001,  # R-INLA uses 23001 for first SPDE hyper
                        "prior": "pcmatern",
                        "param": [pcmatern_param1, pcmatern_param2, pcmatern_param3],
                        "to.theta": "function (x) <<NEWLINE>>x",  # Identity transform
                        "from.theta": "function (x) <<NEWLINE>>x",
                    },
                    {
                        "name": "theta2",
                        "short_name": "Stdev",
                        "initial": initial_theta1,
                        "fixed": False,
                        "hyperid": 23002,  # R-INLA uses 23002 for second SPDE hyper
                        "prior": "none",
                        "param": [],
                        "to.theta": "function (x) <<NEWLINE>>x",  # Identity transform
                        "from.theta": "function (x) <<NEWLINE>>x",
                    },
                ]

                # Remove __spde_object__ marker and mark to skip default hyper merge
                r.pop("__spde_object__", None)
                r["__skip_hyper_merge__"] = True

            model_name = str(r.get("model", "iid")).lower()

            # Merge default hyperparameters from models.py into the random effect spec
            # (skip for SPDE which sets its own hyperparameters)
            if not r.pop("__skip_hyper_merge__", False):
                _merge_random_hyper_defaults(r)

            # Get model properties early (needed for replicate handling and later)
            model_props = inla_model_properties(model_name, "latent", stop_on_error=False) or {}

            if model_name == "linear":
                r_local = dict(r)
                # Use 'id' for naming
                label = r_local.get("id")
                if not isinstance(label, str) or not label.strip():
                    label = f"linear{ridx}"
                label = label.strip()
                r_local["id"] = label

                if r_local.get("weights") is not None:
                    raise PyINLAError(f"Random component '{label}' with model='linear' does not accept weights.")
                if r_local.get("group") is not None:
                    raise PyINLAError(f"Random component '{label}' with model='linear' does not accept 'group'.")
                if r_local.get("replicate") is not None:
                    raise PyINLAError(f"Random component '{label}' with model='linear' does not accept 'replicate'.")

                cov_spec = r_local.get("covariate")
                if cov_spec is None:
                    cov_spec = r_local.get("values")
                if cov_spec is None:
                    cov_spec = r.get("id")

                if cov_spec is None:
                    raise PyINLAError(f"Random component '{label}' with model='linear' requires a covariate vector.")

                if isinstance(cov_spec, str):
                    if cov_spec not in df.columns:
                        raise PyINLAError(
                            f"Random component '{label}': covariate column '{cov_spec}' not found in data."
                        )
                    series = df[cov_spec]
                    try:
                        if pd is not None and hasattr(series, "to_numpy"):
                            cov_values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
                        else:
                            cov_values = np.asarray(series, dtype=float)
                    except Exception as e:
                        raise PyINLAError(
                            f"Random component '{label}': unable to coerce covariate '{cov_spec}' to numeric."
                        ) from e
                else:
                    try:
                        cov_values = np.asarray(cov_spec, dtype=float)
                    except Exception as e:
                        raise PyINLAError(
                            f"Random component '{label}': covariate must be array-like of length {n}."
                        ) from e

                if cov_values.ndim != 1:
                    cov_values = cov_values.reshape(-1)
                if cov_values.size != n:
                    raise PyINLAError(
                        f"Random component '{label}': covariate length mismatch (expected {n}, got {cov_values.size})."
                    )
                cov_values = np.where(np.isfinite(cov_values), cov_values, 0.0)

                table = np.column_stack([np.arange(n, dtype=float), cov_values.astype(float)])
                ref_index = len(fixed_cols) + linear_count
                fixed_ref_name = fixed_refs[ref_index] if ref_index < len(fixed_refs) else None
                file_linear_name = fixed_ref_name or f"linear-{ridx}.dat"
                file_linear_abs = os.path.join(data_dir, file_linear_name)
                write_fmesher_file(table, file_linear_abs)
                file_linear = file_linear_abs.replace(data_dir_abs, "$inladatadir")

                control_linear = copy.deepcopy(control_resolved.get("fixed", {}))
                mean_linear_val = _get_linear_param(r_local, "mean.linear")
                if mean_linear_val is not None:
                    try:
                        control_linear["mean"] = float(mean_linear_val)
                    except Exception as e:
                        raise PyINLAError(
                            f"Random component '{label}': mean.linear must be numeric."
                        ) from e
                prec_linear_val = _get_linear_param(r_local, "prec.linear")
                if prec_linear_val is not None:
                    try:
                        control_linear["prec"] = float(prec_linear_val)
                    except Exception as e:
                        raise PyINLAError(
                            f"Random component '{label}': prec.linear must be numeric."
                        ) from e
                if r_local.get("cdf") is not None:
                    control_linear["cdf"] = r_local["cdf"]
                if r_local.get("quantiles") is not None:
                    control_linear["quantiles"] = r_local["quantiles"]
                if r_local.get("compute") is not None:
                    control_linear["compute"] = r_local["compute"]

                linear_count += 1
                results_idx = len(fixed_cols) + linear_count
                hyper_linear = linear_section(
                    file=file_ini,
                    file_fixed=file_linear,
                    label=label,
                    results_dir=f"fixed.effect{results_idx:010d}",
                    control_fixed=control_linear,
                    only_hyperparam=bool(control_resolved.get("only_hyperparam", False))
                )
                all_hyper.setdefault("linear", []).append(hyper_linear)
                continue

            # Non-linear latent models fall back to the general ffield writer
            inferred_n = r.get("n", None)
            # Get 'id' - can be a column name string or an index vector
            id_obj = r.get("id")
            id_values: Optional[np.ndarray] = None

            if isinstance(id_obj, str):
                if id_obj in df.columns:
                    try:
                        col = df[id_obj]
                        if pd is not None and hasattr(col, "to_numpy"):
                            id_values = pd.to_numeric(col, errors="coerce").to_numpy()
                        else:
                            id_values = np.asarray(col, dtype=float)
                    except Exception:
                        id_values = None
            else:
                try:
                    tv = np.asarray(id_obj)
                    if tv.ndim > 1:
                        tv = tv.reshape(-1)
                    if tv.size == n:
                        id_values = tv.astype(float)
                        if not isinstance(r.get("id", None), str):
                            r = dict(r)
                            r["id"] = f"random{ridx}"
                    elif tv.size != 0:
                        raise PyINLAError(
                            f"Random id vector for component {ridx} must have length {n}, got {tv.size}"
                        )
                except Exception:
                    id_values = None

            if ("id" not in r) and isinstance(id_obj, str):
                r = dict(r)
                r["id"] = id_obj

            # For models like clinear/copy that use 'covariate' instead of 'id',
            # generate a label from covariate name or model+index
            if "id" not in r:
                r = dict(r)
                cov_name = r.get("covariate")
                if isinstance(cov_name, str) and cov_name.strip():
                    r["id"] = cov_name.strip()
                else:
                    r["id"] = f"{model_name}{ridx}"

            if inferred_n is None and id_values is not None:
                try:
                    valid = id_values[np.isfinite(id_values)]
                    if valid.size:
                        inferred_n = int(np.nanmax(valid))
                except Exception:
                    inferred_n = None

            # --- Special handling for clinear model ---
            # clinear uses actual covariate values (like linear fixed effects), not index mapping
            is_clinear = model_name.lower() == "clinear"
            clinear_cov_values = None

            if is_clinear:
                # Extract covariate values for clinear
                # Only allow 'covariate' or 'id' keys
                cov_spec = r.get("covariate")
                if cov_spec is None:
                    id_val = r.get("id")
                    if isinstance(id_val, str):
                        cov_spec = id_val
                if cov_spec is None:
                    raise PyINLAError(
                        f"Random component with model='clinear' requires 'covariate' or 'id' to specify the covariate column."
                    )

                if isinstance(cov_spec, str):
                    if cov_spec not in df.columns:
                        raise PyINLAError(
                            f"Random component '{r.get('id', ridx)}': covariate column '{cov_spec}' not found in data."
                        )
                    series = df[cov_spec]
                    try:
                        if pd is not None and hasattr(series, "to_numpy"):
                            clinear_cov_values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
                        else:
                            clinear_cov_values = np.asarray(series, dtype=float)
                    except Exception as e:
                        raise PyINLAError(
                            f"Random component '{r.get('id', ridx)}': unable to coerce covariate '{cov_spec}' to numeric."
                        ) from e
                else:
                    try:
                        clinear_cov_values = np.asarray(cov_spec, dtype=float)
                    except Exception as e:
                        raise PyINLAError(
                            f"Random component '{r.get('id', ridx)}': covariate must be array-like of length {n}."
                        ) from e

                if clinear_cov_values.ndim != 1:
                    clinear_cov_values = clinear_cov_values.reshape(-1)
                if clinear_cov_values.size != n:
                    raise PyINLAError(
                        f"Random component '{r.get('id', ridx)}': covariate length mismatch (expected {n}, got {clinear_cov_values.size})."
                    )
                # Replace NaN with 0 (like R-INLA does for clinear)
                clinear_cov_values = np.where(np.isfinite(clinear_cov_values), clinear_cov_values, 0.0)

            # --- Special handling for SPDE models with A.local ---
            # SPDE models use A.local projection matrix; n comes from SPDE model, not from id column
            is_spde_model = model_name.lower() in ("spde", "spde2", "spde3") or r.get("A.local") is not None

            # --- Special handling for z model and bym2 model ---
            # z model uses Z matrix directly; n should be z.n + z.m (augmented size)
            is_z_model = model_name.lower() == "z"
            is_bym2_model = model_name.lower() == "bym2"
            is_iidkd_model = model_name.lower() == "iidkd"
            if is_spde_model:
                # For SPDE models, use the n from the SPDE model (set earlier)
                # Don't use id column values to determine component size
                if inferred_n is not None:
                    component_n = int(inferred_n)
                else:
                    raise PyINLAError(
                        f"Random component '{r.get('id', ridx)}' with SPDE model requires 'n' to be set."
                    )
                # When A.local is provided, set eff_idx to -1 for all observations
                # This tells INLA to use the A.local matrix for mapping instead of direct indices
                # (matching R-INLA behavior: covariates column 2 = -1 when A.local is used)
                obs_idx = np.arange(n, dtype=np.int64)
                eff_idx = np.full(n, -1, dtype=np.int64)  # -1 means use A.local for mapping
                # Locations: mesh vertex indices (1-based for INLA)
                location_vals = np.arange(1, component_n + 1, dtype=np.int64)
            elif is_z_model:
                Z = r.get("Z")
                if Z is None:
                    raise PyINLAError(
                        f"Random component '{r.get('id', ridx)}' with model='z' requires 'Z' design matrix."
                    )
                if _HAVE_SCIPY and sp.issparse(Z):
                    z_n, z_m = Z.shape
                else:
                    Z_arr = np.asarray(Z)
                    if Z_arr.ndim != 2:
                        raise PyINLAError(
                            f"Random component '{r.get('id', ridx)}': Z must be a 2D matrix."
                        )
                    z_n, z_m = Z_arr.shape
                # For z model, component_n = z.n + z.m (total augmented latent field size)
                component_n = z_n + z_m
                # Simple sequential index mapping for z model
                obs_idx = np.arange(n, dtype=np.int64)
                eff_idx = np.arange(n, dtype=np.int64)  # Direct 1:1 mapping
                # Locations should have n+m elements (full augmented latent field)
                location_vals = np.arange(1, component_n + 1, dtype=np.int64)
            else:
                # --- Random effect index handling ---
                obs_idx = np.arange(n, dtype=np.int64)
                eff_idx = np.full(n, -1, dtype=np.int64)  # Default to -1 (NA/excluded)
                location_vals = None  # Will hold actual values

            if not is_z_model and not is_spde_model and is_clinear and clinear_cov_values is not None:
                # clinear: R-INLA sorts unique covariate values and creates a mapping
                # location = sort(unique(values)), cov = match(values, location) - 1
                finite_mask = np.isfinite(clinear_cov_values)
                if finite_mask.any():
                    valid_cov = clinear_cov_values[finite_mask]
                    # Sort unique covariate values (like R-INLA)
                    location_vals = np.sort(np.unique(valid_cov))
                    component_n = len(location_vals)

                    # Map each observation to its position in the sorted location array
                    for i in range(n):
                        if finite_mask[i]:
                            val = clinear_cov_values[i]
                            pos = np.searchsorted(location_vals, val)
                            if pos < len(location_vals) and np.isclose(location_vals[pos], val):
                                eff_idx[i] = pos
                            # else: stays -1 (value not found, treated as NA)
                        # else: stays -1 (NA in input)
                else:
                    # All values are NA - use identity mapping as fallback
                    eff_idx = obs_idx.copy()
                    location_vals = clinear_cov_values
                    component_n = n
            elif not is_z_model and not is_spde_model and r.get("values") is not None:
                # User-specified values for RW1/RW2 or other models
                # R-INLA: values = numeric/factor vector, n = length(values)
                # The locations are the specified values, and data indices map to positions
                user_values = np.asarray(r.get("values"), dtype=float).ravel()
                # Sort unique values (like R-INLA)
                location_vals = np.sort(np.unique(user_values[np.isfinite(user_values)]))
                component_n = len(location_vals)

                # Map id_values (from data) to positions in location_vals
                if id_values is not None:
                    finite_mask = np.isfinite(id_values)
                    for i in range(n):
                        if finite_mask[i]:
                            val = id_values[i]
                            # Find position in location_vals using nearest match for floats
                            pos = np.searchsorted(location_vals, val)
                            if pos < len(location_vals):
                                # Check for exact or very close match
                                if pos > 0 and abs(location_vals[pos-1] - val) < abs(location_vals[pos] - val):
                                    pos = pos - 1
                                if abs(location_vals[pos] - val) < 1e-10:
                                    eff_idx[i] = pos
                                # else: stays -1 (value not in location, treated as NA)
                        # else: stays -1 (NA in input)
                else:
                    # No id_values - use direct mapping (identity)
                    for i in range(min(n, component_n)):
                        eff_idx[i] = i
            elif not is_z_model and not is_spde_model and id_values is not None:
                # Standard random effect: index mapping
                # R-INLA: location = sort(unique(xx)), cov = match(xx, location) - 1, NA -> -1
                finite_mask = np.isfinite(id_values)
                if finite_mask.any():
                    # Round to integers for finite values
                    valid_vals = id_values[finite_mask]
                    rounded_valid = np.rint(valid_vals).astype(np.int64)

                    # Get unique sorted values (like R-INLA: location = sort(unique(xx)))
                    location_vals = np.sort(np.unique(rounded_valid))

                    # Map each value to its 0-based position in location array
                    # (like R-INLA: cov = match(xx, location) - 1)
                    for i in range(n):
                        if finite_mask[i]:
                            val = int(np.rint(id_values[i]))
                            pos = np.searchsorted(location_vals, val)
                            if pos < len(location_vals) and location_vals[pos] == val:
                                eff_idx[i] = pos
                            # else: stays -1 (value not in location, treated as NA)
                        # else: stays -1 (NA in input)

                # Determine component size for standard random effects
                if location_vals is not None and len(location_vals) > 0:
                    component_n = len(location_vals)
                else:
                    valid_effect = eff_idx[eff_idx >= 0]
                    candidate_n = int(valid_effect.max()) + 1 if valid_effect.size else int(n)
                    component_n = int(inferred_n) if inferred_n is not None else candidate_n
                    component_n = max(component_n, candidate_n)
                    # Create default location array
                    location_vals = np.arange(1, component_n + 1, dtype=np.int64)
            elif not is_z_model and not is_spde_model:
                # No id_values provided - use default identity
                valid_effect = eff_idx[eff_idx >= 0]
                candidate_n = int(valid_effect.max()) + 1 if valid_effect.size else int(n)
                component_n = int(inferred_n) if inferred_n is not None else candidate_n
                component_n = max(component_n, candidate_n)
                location_vals = np.arange(1, component_n + 1, dtype=np.int64)

            # --- Handle replicate parameter ---
            # R-INLA: nrep = max(replicate), cov = base_idx + (replicate - 1) * N * ngroup
            replicate_spec = r.get("replicate")
            nrep = 1
            replicate_vec = None
            if replicate_spec is not None:
                # Get replicate values (column name or array)
                if isinstance(replicate_spec, str):
                    if replicate_spec not in df.columns:
                        raise PyINLAError(
                            f"Random component '{r.get('id', ridx)}': replicate column '{replicate_spec}' not found in data."
                        )
                    replicate_vec = df[replicate_spec].to_numpy()
                else:
                    replicate_vec = np.asarray(replicate_spec).ravel()

                if replicate_vec.size != n:
                    raise PyINLAError(
                        f"Random component '{r.get('id', ridx)}': replicate length mismatch (expected {n}, got {replicate_vec.size})."
                    )

                # Convert to integer indices (1-based like R)
                replicate_vec = np.rint(replicate_vec).astype(np.int64)

                # Compute nrep from max(replicate) where not NA
                valid_rep = replicate_vec[~np.isnan(replicate_vec.astype(float))]
                if valid_rep.size > 0:
                    nrep = int(np.max(valid_rep))
                else:
                    nrep = 1

            # --- Handle group parameter ---
            # R-INLA: ngroup = max(group), cov includes (group - 1) * N offset
            group_spec = r.get("group")
            ngroup = 1
            group_vec = None
            if group_spec is not None:
                # Get group values (column name or array)
                if isinstance(group_spec, str):
                    if group_spec not in df.columns:
                        raise PyINLAError(
                            f"Random component '{r.get('id', ridx)}': group column '{group_spec}' not found in data."
                        )
                    group_vec = df[group_spec].to_numpy()
                else:
                    group_vec = np.asarray(group_spec).ravel()

                if group_vec.size != n:
                    raise PyINLAError(
                        f"Random component '{r.get('id', ridx)}': group length mismatch (expected {n}, got {group_vec.size})."
                    )

                # Convert to integer indices (1-based like R)
                group_vec = np.rint(group_vec).astype(np.int64)

                # Compute ngroup from max(group) where not NA, or use explicit ngroup if provided
                explicit_ngroup = r.get("ngroup")
                if explicit_ngroup is not None:
                    ngroup = int(explicit_ngroup)
                else:
                    valid_grp = group_vec[~np.isnan(group_vec.astype(float))]
                    if valid_grp.size > 0:
                        ngroup = int(np.max(valid_grp))
                    else:
                        ngroup = 1

            # Get augmentation factor for this model (bym2=2, others=1)
            aug_factor_rep = int(model_props.get("aug_factor", 1)) if model_props.get("augmented") else 1
            N_aug = component_n * aug_factor_rep  # Augmented dimension per replicate/group

            # Apply offsets to eff_idx
            # R-INLA formula: cov = base_idx + (replicate - 1) * N * ngroup + (group - 1) * N
            if nrep > 1 or ngroup > 1:
                for i in range(n):
                    if eff_idx[i] >= 0:
                        offset = 0
                        # Replicate offset: (replicate - 1) * N * ngroup
                        if replicate_vec is not None and replicate_vec[i] >= 1:
                            offset += (replicate_vec[i] - 1) * N_aug * ngroup
                        # Group offset: (group - 1) * N
                        if group_vec is not None and group_vec[i] >= 1:
                            offset += (group_vec[i] - 1) * N_aug
                        eff_idx[i] = eff_idx[i] + offset

            file_cov_path = None
            ref_cov = random_refs[ridx-1].get("covariates") if ridx-1 < len(random_refs) else None
            # Write covariates file for all random effects (not just IID)
            if A is None:
                try:
                    tbl = np.column_stack([obs_idx, eff_idx]).astype(np.int64)
                    cov_name = ref_cov or f"rand-{ridx}-cov.dat"
                    cov_abs = os.path.join(data_dir, cov_name)
                    write_fmesher_file(tbl, cov_abs)
                    file_cov_path = cov_abs.replace(data_dir_abs, "$inladatadir")
                except Exception:
                    file_cov_path = None

            file_loc_path = None
            ref_loc = random_refs[ridx-1].get("locations") if ridx-1 < len(random_refs) else None
            try:
                # Write actual location values (like R-INLA)
                loc = location_vals.astype(float).reshape(-1, 1)
                loc_name = ref_loc or f"rand-{ridx}-loc.dat"
                loc_abs = os.path.join(data_dir, loc_name)
                write_fmesher_file(loc, loc_abs)
                file_loc_path = loc_abs.replace(data_dir_abs, "$inladatadir")
            except Exception:
                file_loc_path = None

            model_lower = model_name.lower()
            graph_required_models = {"besag", "bym", "bym2", "besagproper", "besagproper2", "besag2"}
            adjust_models = {"besag", "bym", "bym2", "besag2"}
            graph_obj = None
            cc_n1 = 0
            cc_n2 = 0

            if model_lower in graph_required_models:
                graph_spec = r.get("graph")
                if graph_spec is None:
                    raise PyINLAError(
                        f"Random component '{r.get('id', ridx)}' with model='{model_name}' requires 'graph'."
                    )
                try:
                    graph_obj = inla_read_graph(graph_spec)
                    if hasattr(graph_obj, "with_cc"):
                        graph_obj = graph_obj.with_cc()
                except Exception as exc:
                    raise PyINLAError(
                        f"Random component '{r.get('id', ridx)}': unable to read graph ({exc})."
                    ) from exc
                graph_nodes = int(graph_obj.n)
                expected_n = graph_nodes if model_lower != "besag2" else 2 * graph_nodes
                if component_n is None:
                    component_n = expected_n
                elif int(component_n) != int(expected_n):
                    raise PyINLAError(
                        f"Random component '{r.get('id', ridx)}': n={component_n} does not match graph size {expected_n}."
                    )
                cc_nodes = (graph_obj.cc or {}).get("nodes", [])
                cc_sizes = [len(nodes) for nodes in cc_nodes]
                cc_n1 = sum(size == 1 for size in cc_sizes)
                cc_n2 = sum(size >= 2 for size in cc_sizes)
                if r.get("n") is None or int(r.get("n")) != int(component_n):
                    r = r.copy()
                    r["n"] = int(component_n)

            component_n = int(component_n)
            aug_factor = int(model_props.get("aug_factor", 1)) if model_props.get("augmented") else 1
            latent_dim = int(component_n * aug_factor)
            aug_constr = model_props.get("aug_constr", 1)
            if isinstance(aug_constr, (list, tuple)) and aug_constr:
                aug_constr_val = int(aug_constr[0])
            else:
                aug_constr_val = int(aug_constr or 1)
            constr_offset = int((aug_constr_val - 1) * component_n) if model_props.get("augmented") else 0

            extra_constr = None
            force_constr_off = False
            ref_extraconstr = random_refs[ridx-1].get("extraconstraint") if ridx-1 < len(random_refs) else None
            extraconstr = r.get("extraconstr")

            if extraconstr is not None:
                A_constr = extraconstr.get("A")
                if A_constr is None:
                    raise PyINLAError("extraconstr requires 'A'.")
                e_constr = extraconstr.get("e")
                if e_constr is None:
                    if _HAVE_SCIPY and sp is not None and sp.issparse(A_constr):
                        A_tmp = A_constr.toarray()
                    else:
                        A_tmp = np.asarray(A_constr)
                    rows = 1 if A_tmp.ndim == 1 else A_tmp.shape[0]
                    e_constr = np.zeros(rows)
                extra_constr = _stack_extra_constraints(extra_constr, A_constr, e_constr, latent_dim)

            if is_z_model and r.get("constr", False):
                try:
                    Z_mat = r.get("Z")
                    if _HAVE_SCIPY and sp is not None and sp.issparse(Z_mat):
                        z_n, z_m = Z_mat.shape
                    else:
                        Z_arr = np.asarray(Z_mat)
                        z_n, z_m = Z_arr.shape
                    A_constr = np.concatenate([np.zeros(z_n), np.ones(z_m)]).reshape(1, -1)
                    extra_constr = _stack_extra_constraints(extra_constr, A_constr, np.array([0.0]), latent_dim)
                    force_constr_off = True
                except Exception:
                    pass

            if is_bym2_model and r.get("constr", True):
                try:
                    bym2_n = component_n
                    # R-INLA writes only 1 constraint row for BYM2:
                    # sum of unstructured effect = 0 [0,0,...,0,1,1,...,1]
                    # The structured component sum-to-zero is handled internally by ICAR with scale.model
                    A_constr = np.concatenate([np.zeros(bym2_n), np.ones(bym2_n)]).reshape(1, -1)
                    extra_constr = _stack_extra_constraints(extra_constr, A_constr, np.array([0.0]), latent_dim)
                    force_constr_off = True
                except Exception:
                    pass

            # IIDKD: sum-to-zero constraint per dimension (k constraints for k dimensions)
            # For k dimensions and m groups (n = k * m):
            # - Row 0: sum of indices 1 to m = 0 (dimension 1)
            # - Row 1: sum of indices m+1 to 2m = 0 (dimension 2)
            # - etc.
            if is_iidkd_model and r.get("constr", False):
                try:
                    k = int(r.get("order", 2))  # Number of dimensions
                    iidkd_n = component_n  # Total n = k * m
                    m = iidkd_n // k  # Number of groups per dimension
                    # Build k constraint rows (one per dimension)
                    rows = []
                    for d in range(k):
                        row = np.zeros(iidkd_n, dtype=float)
                        row[d * m : (d + 1) * m] = 1.0
                        rows.append(row)
                    A_constr = np.vstack(rows)
                    extra_constr = _stack_extra_constraints(extra_constr, A_constr, np.zeros(k), latent_dim)
                    force_constr_off = True
                except Exception:
                    pass

            adjust_for_cc = bool(r.get("adjust.for.con.comp", True))
            if (
                graph_obj is not None
                and model_lower in adjust_models
                and adjust_for_cc
                and (graph_obj.cc or {}).get("n", 1) > 1
                and r.get("constr", True)
            ):
                rows = []
                for nodes in (graph_obj.cc or {}).get("nodes", []):
                    if len(nodes) >= 2:
                        row = np.zeros(latent_dim, dtype=float)
                        idx_nodes = np.asarray(nodes, dtype=int) - 1
                        placement = constr_offset + idx_nodes
                        if np.any(placement < 0) or np.any(placement >= latent_dim):
                            raise PyINLAError("extraconstr index out of bounds when adjusting for components.")
                        row[placement] = 1.0
                        rows.append(row)
                if rows:
                    extra_constr = _stack_extra_constraints(extra_constr, np.vstack(rows), np.zeros(len(rows)), latent_dim)
                    force_constr_off = True

            scale_model_flag = r.get("scale.model")
            if scale_model_flag is None:
                scale_model_flag = True if model_lower == "bym2" else False

            if graph_obj is not None and model_lower in adjust_models and r.get("rankdef") is None:
                extra_rows = 0
                if extra_constr is not None:
                    A_rows, _ = extra_constr
                    extra_rows = A_rows.shape[0]
                r = r.copy()
                if is_bym2_model:
                    # R-INLA creates 2 conceptual constraint rows for BYM2 but writes 1 to file.
                    # R-INLA uses: rankdef <- nrow(extraconstr$A) %/% 2L where nrow=2 in memory.
                    # We write 1 row, so use (extra_rows + 1) // 2 to account for the extra conceptual row.
                    r["rankdef"] = int((extra_rows + 1) // 2)
                else:
                    base_rankdef = cc_n2 if scale_model_flag else cc_n2 + cc_n1
                    r["rankdef"] = int(base_rankdef + extra_rows)

            file_extraconstr_path = None
            has_extraconstr = False
            if extra_constr is not None:
                A_mat, e_vec = extra_constr
                if A_mat is not None and A_mat.size and A_mat.shape[0] > 0:
                    data_vec = np.concatenate([A_mat.astype(float).ravel(order="C"), e_vec.astype(float)])
                    constr_matrix = data_vec.reshape(-1, 1)
                    extraconstr_name = ref_extraconstr or f"rand-{ridx}-extraconstr.dat"
                    extraconstr_abs = os.path.join(data_dir, extraconstr_name)
                    write_fmesher_file(constr_matrix, extraconstr_abs)
                    file_extraconstr_path = extraconstr_abs.replace(data_dir_abs, "$inladatadir")
                    has_extraconstr = True
            elif ref_extraconstr is not None:
                file_extraconstr_path = ref_extraconstr
                has_extraconstr = True

            random_spec_for_section = r
            if force_constr_off:
                random_spec_for_section = r.copy()
                random_spec_for_section["constr"] = False
                if random_spec_for_section.get("diagonal") is None:
                    random_spec_for_section["diagonal"] = 1e-04
            elif has_extraconstr:
                random_spec_for_section = r.copy()
                if random_spec_for_section.get("diagonal") is None:
                    random_spec_for_section["diagonal"] = 1e-04
            elif r.get("constr"):
                random_spec_for_section = r.copy()
                if random_spec_for_section.get("diagonal") is None:
                    random_spec_for_section["diagonal"] = 1e-04

            rs_updated = ffield_section(
                file=file_ini,
                file_loc=file_loc_path,
                file_cov=file_cov_path,
                file_id_names=None,
                n=component_n,
                nrep=nrep if nrep > 1 else None,
                ngroup=ngroup if ngroup > 1 else None,
                file_extraconstr=file_extraconstr_path,
                file_weights=None,
                random_spec=random_spec_for_section,
                results_dir=f"random.effect{ridx:010d}",
                only_hyperparam=bool(control_resolved.get("only_hyperparam", False)),
                data_dir=data_dir
            )
            all_hyper["random"].append({
                "hyperid": r.get("id", f"rand{ridx}"),
                "hyper": rs_updated.get("hyper", {}),
                "group_hyper": (rs_updated.get("control_group", {}) or {}).get("hyper", {})
            })

    # --- INLA internal sections
    inla_section(
        file=file_ini,
        inla_spec=control_resolved.get("inla", {}),
        data_dir=data_dir,
        inla_mode=inla_mode
    )

    mode_section(file=file_ini, args=control_resolved.get("mode", {}), data_dir=data_dir)
    expert_section(file=file_ini, args=control_resolved.get("expert", {}), data_dir=data_dir)
    lincomb_section(file=file_ini, data_dir=data_dir,
                    contr=control_resolved.get("lincomb", {}),
                    lincomb=lincomb)
    update_section(file=file_ini, data_dir=data_dir, contr=control_resolved.get("update", {}))

    # lp.scale section
    lp_scale_section(
        file=file_ini,
        contr=control_resolved.get("lp_scale", {}),
        data_dir=data_dir,
        write_hyper=(lp_scale is not None)
    )

    # Solver sections: emit the standard quartet to mirror R-INLA output
    pardiso_section(file=file_ini, data_dir=data_dir, contr=control_resolved.get("pardiso", {}))
    numa_section(file=file_ini, data_dir=data_dir, contr=control_resolved.get("numa", {}))
    stiles_section(file=file_ini, data_dir=data_dir, contr=control_resolved.get("stiles", {}))
    taucs_section(file=file_ini, data_dir=data_dir, contr=control_resolved.get("taucs", {}))

    _inla_set_environment()
    env_saved = _inla_run_environment_set()
    env_file = _write_environment_file(inla_dir)

    # --- Done writing; optionally call INLA binary
    t1 = time.time()
    call_str = ""
    logfile = None

    # Resolve default INLA binary path when the caller did not supply one.
    if not dry_run and inla_arg is None and inla_call is None:
        try:
            inla_call = inla_call_builtin()
        except (FileNotFoundError, RuntimeError) as e:
            raise PyINLAError(
                "INLA binary not found and auto-download failed.\n"
                "Try: pyinla.download_binary(os_name='Ubuntu-22.04')\n"
                "Or:  pyinla.list_available_os() to see options."
            ) from e

    # If caller sets dry_run=True skip executing the INLA binary.
    skip_binary = bool(dry_run)

    results_dict: Optional[Dict[str, Any]] = None
    collector_error: Optional[str] = None
    summary_fixed = marginals_fixed = None
    summary_random = marginals_random = None
    summary_hyperpar = marginals_hyperpar = None
    summary_linear_predictor = marginals_linear_predictor = None
    summary_fitted_values = marginals_fitted_values = None
    summary_lincomb = marginals_lincomb = None
    selection = dic = cpo = po = residuals = waic = mlik = neffp = mode_obj = formula = None
    nhyper_val: Optional[int] = None

    try:
        if skip_binary:
            cpu = {
                "Pre": t1 - t0,
                "Running": 0.0,
                "Post": 0.0,
                "Total": t1 - t0
            }
            return PyINLAResult(
                cpu_used=cpu,
                call="(dry run: built Model.ini and data files)",
                model_matrix=X,
                args={
                    "model": model,
                    "family": fams,
                    "control": control_resolved,
                    "data.orig": df,
                    "file_ini": file_ini,
                    "inla_dir": inla_dir,
                    "environment_file": env_file,
                },
                inla_dir=inla_dir,
                all_hyper=all_hyper,
                logfile=None,
                results=None,
                collector_error=None
            )

        if inla_arg is not None:
            arg_custom = inla_arg
            nt = ""
            vflag = ""
            sflag = ""
        else:
            nt = f"-t{_num_threads_to_arg(num_threads)}"
            vflag = "-v"
            sflag = "-s" if (silent == 1 or silent is True) else ""
            arg_custom = ""

        Pflag = {
            "classic": "-P classic",
            "twostage": "-P twostage",
            "compact": "-P compact"
        }.get(inla_mode, "-P compact")

        call = (inla_call or "inla").strip()
        all_args = " ".join(s for s in [arg_custom, sflag, vflag, nt, Pflag] if s)
        call_str = f"{call} {all_args} {file_ini}"

        if debug:
            print(f"[pyinla] Running: {call_str}", file=sys.stderr)

        file_log = os.path.join(inla_dir, "Logfile.txt") if not verbose else None
        t2 = time.time()
        try:
            if file_log is None:
                rc = subprocess.call(call_str, shell=True)
            else:
                with open(file_log, "w", encoding="utf-8") as lf:
                    rc = subprocess.call(call_str, shell=True, stdout=lf, stderr=lf)
        except Exception as e:
            raise PyINLACrashError("The INLA program call crashed.") from e
        t3 = time.time()

        if rc != 0:
            log_content = ""
            if file_log and os.path.exists(file_log):
                with open(file_log, "r", encoding="utf-8", errors="ignore") as lf:
                    logfile = lf.readlines()
                    # Extract last 100 lines for error message (or all if fewer)
                    error_lines = logfile[-100:] if len(logfile) > 100 else logfile
                    log_content = "".join(error_lines)
            error_msg = "The INLA program exited with non-zero status."
            if log_content:
                error_msg += f"\n\n=== INLA Output (last 100 lines) ===\n{log_content}"
            raise PyINLACrashError(error_msg)

        if file_log and os.path.exists(file_log):
            with open(file_log, "r", encoding="utf-8", errors="ignore") as lf:
                logfile = lf.readlines()

        cpu_used = {
            "Pre": t1 - t0,
            "Running": t3 - t2,
            "Post": 0.0,
            "Total": (t3 - t0)
        }

        if collect_results:
            # Lazy import to avoid hard dependency when only doing dry-runs
            from .collect import collect_inla_results
            try:
                results_dict = collect_inla_results(inla_dir, allow_parent=False, debug=debug)
            except Exception as exc:  # noqa: BLE001
                collector_error = f"collect_inla_results failed: {exc}"
            else:
                if isinstance(results_dict, dict):
                    summary_fixed = results_dict.get("summary.fixed")
                    marginals_fixed = results_dict.get("marginals.fixed")
                    summary_random = results_dict.get("summary.random")
                    marginals_random = results_dict.get("marginals.random")
                    summary_hyperpar = results_dict.get("summary.hyperpar")
                    marginals_hyperpar = results_dict.get("marginals.hyperpar")
                    summary_linear_predictor = results_dict.get("summary.linear.predictor")
                    marginals_linear_predictor = results_dict.get("marginals.linear.predictor")
                    summary_fitted_values = results_dict.get("summary.fitted.values")
                    marginals_fitted_values = results_dict.get("marginals.fitted.values")
                    summary_lincomb = results_dict.get("summary.lincomb")
                    marginals_lincomb = results_dict.get("marginals.lincomb")
                    selection = results_dict.get("selection")
                    dic = results_dict.get("dic")
                    cpo = results_dict.get("cpo")
                    po = results_dict.get("po")
                    residuals = results_dict.get("residuals")
                    waic = results_dict.get("waic")
                    mlik = results_dict.get("mlik")
                    neffp = results_dict.get("neffp")
                    mode_obj = results_dict.get("mode")
                    formula = results_dict.get("formula")
                    nhyper_val = results_dict.get("nhyper")
                    results_dict["directory"] = inla_dir

        return PyINLAResult(
            cpu_used=cpu_used,
            call=call_str,
            model_matrix=X,
            args={
                "model": model,
                "family": fams,
                "control": control_resolved,
                "data.orig": df,
                "file_ini": file_ini,
                "inla_dir": inla_dir,
                "environment_file": env_file,
            },
            inla_dir=inla_dir,
            all_hyper=all_hyper,
            logfile=logfile,
            results=results_dict,
            collector_error=collector_error,
            summary_fixed=summary_fixed,
            marginals_fixed=marginals_fixed,
            summary_random=summary_random,
            marginals_random=marginals_random,
            summary_hyperpar=summary_hyperpar,
            marginals_hyperpar=marginals_hyperpar,
            summary_linear_predictor=summary_linear_predictor,
            marginals_linear_predictor=marginals_linear_predictor,
            summary_fitted_values=summary_fitted_values,
            marginals_fitted_values=marginals_fitted_values,
            summary_lincomb=summary_lincomb,
            marginals_lincomb=marginals_lincomb,
            selection=selection,
            dic=dic,
            cpo=cpo,
            po=po,
            residuals=residuals,
            waic=waic,
            mlik=mlik,
            neffp=neffp,
            mode=mode_obj,
            formula=formula,
            nhyper=nhyper_val,
        )
    finally:
        _inla_run_environment_unset(env_saved)
        if not keep:
            shutil.rmtree(inla_dir, ignore_errors=True)


# -----------------------
# Safe wrapper (like inla.core.safe in R)
# -----------------------
def _core_safe(**kwargs) -> PyINLAResult:
    """
    Retry strategy similar to R's inla.core.safe:
    - First run with user options.
    - If it fails (crash), fall back to a simpler setting (gaussian strategy,
      fewer outputs) and then optionally rerun a final pass if you implement it.
    """
    try:
        return _core(**kwargs)
    except PyINLACrashError as e:
        # fallback: simplify controls to help initial values
        control = (kwargs.get("control") or {}).copy()
        control_inla = dict(control.get("inla", {}))
        control_compute = dict(control.get("compute", {}))
        control_predictor = dict(control.get("predictor", {}))

        # reduce workload
        control_inla.setdefault("strategy", "gaussian")
        control_inla.setdefault("int_strategy", "eb")
        control_inla["compute_initial_values"] = True
        control_inla["force_diagonal"] = True
        control_inla.setdefault("optimise_strategy", "plain")
        control_inla.setdefault("tolerance", 0.01)

        for k in ("return_marginals", "return_marginals_predictor", "dic", "cpo", "po", "waic", "residuals", "config", "q", "graph"):
            control_compute[k] = False

        control_predictor["compute"] = False

        control["inla"] = control_inla
        control["compute"] = control_compute
        control["predictor"] = control_predictor

        kwargs2 = dict(kwargs)
        kwargs2["control"] = control
        # one retry
        return _core(**kwargs2)


# -----------------------
# Public entry points
# -----------------------
def _run_impl(*,
              _safety_token: Optional[str] = None,
              model: Optional[Dict[str, Any]] = None,
              formula: Optional[Dict[str, Any]] = None,
              family: Union[str, List[str]] = "gaussian",
              data: Union[Dict[str, Any], "pd.DataFrame", None] = None,
              quantiles: List[float] = [0.025, 0.5, 0.975],
              E: Optional[Any] = None,
              offset: Optional[Any] = None,
              scale: Optional[Any] = None,
              weights: Optional[Any] = None,
              Ntrials: Optional[Any] = None,
              strata: Optional[Any] = None,
              lp_scale: Optional[Any] = None,
              link_covariates: Optional[Any] = None,
              verbose: bool = False,
              lincomb: Optional[Any] = None,
              selection: Optional[Any] = None,
              control: Optional[Dict[str, Any]] = None,
              inla_call: Optional[str] = None,
              inla_arg: Optional[str] = None,
              num_threads: Union[str, int] = "0:1",
              keep: bool = False,
              working_directory: Optional[str] = None,
              silent: Union[int, bool] = 0,
              inla_mode: str = "compact",
              safe: bool = True,
              debug: bool = False,
              reuse_filenames_from: Optional[str] = None,
              dry_run: bool = False,
              collect: bool = True) -> PyINLAResult:
    """Internal implementation backing ``pyinla.run``.

    Parameters
    ----------
    model : dict, optional
        {
          "response": <str>,               # column name in `data`
          "fixed":    [<str or array-like>, ...],
                        # e.g. ["1", "z1", "z2"]; "1" adds intercept; you may also pass
                        # numeric vectors (or (name, values)) of length n_predictor directly
          "random":   [ { "id": <str or array-like>, "model": <str>, ... }, ... ],
                        # "id" accepts a column name or an index vector of length n
          "predictor": { "A": <sparse/dense/ijx> }  # optional
        }
    formula : dict, optional
        Alias for ``model``. If provided, ``model`` must be omitted. Exactly one of
        ``model`` or ``formula`` must be specified.
    family : str or list[str]
        Likelihood family (e.g., "gaussian").
    data : pandas.DataFrame or dict of arrays, optional
        Observed data. If omitted, you may pass the response vector directly in
        ``model['response']`` (array-like). In that case, a minimal DataFrame is
        constructed internally with a single response column ``'y'`` and the
        model's response is normalized to that label. Any fixed terms supplied
        as direct vectors remain valid; interaction terms still require named
        columns in ``data``.
    control : dict
        Merged control.* tree: keys "compute", "predictor", "family", "inla",
        "fixed", "mode", "expert", "hazard", "lincomb", "update", "lp_scale",
        "pardiso", "stiles", "taucs", "numa", "only_hyperparam".
    inla_call : str or None
        Path/name of INLA binary. If None AND inla_arg is None => DRY RUN (files only).
    inla_arg : str or None
        Expert full-arg string (skip defaults). If set, `num_threads`, `silent`, etc., are ignored.
    reuse_filenames_from : str or None
        Optional path to a reference INLA Model.ini (or its parent directory). When provided,
        generated files under data.files reuse the filenames from the reference so R/Python
        outputs line up byte-for-byte.
    dry_run : bool
        If True, skip running the INLA binary and only write Model.ini and data files.
    collect : bool
        If True (default), immediately collect the INLA results so the returned object exposes
        the same dictionary-style API as R's `inla()` output.
    """
    # -------------------------------------------------------------------------
    # SAFETY GATE: Validate token to ensure caller passed through safety checks
    # -------------------------------------------------------------------------
    _validate_safety_token(_safety_token)

    # Accept alias: exactly one of model or formula must be provided because model = and formula = (both works as formula)
    if model is not None and formula is not None:
        raise PyINLAError("Provide exactly one of 'model' or 'formula', not both.")
    if model is None:
        model = formula
    if model is None:
        raise PyINLAError("Missing model specification: provide 'model={...}' or 'formula={...}'.")

    #this is added just to compare with R input/output
    if reuse_filenames_from is None:
        reuse_filenames_from = os.environ.get("PYINLA_REFERENCE_MODEL")
    # Choose execution runner based on the 'safe' flag: //i think the default of safe is true.
    # - _core runs the model once and raises on INLA binary crash.
    # - _core_safe first calls _core; if that crashes, it retries with
    #   conservative INLA settings (e.g., gaussian strategy, EB integration,
    #   initial values, forced diagonal, relaxed tolerances) and disables
    #   heavy outputs (dic/waic/cpo/po/marginals) to improve robustness.
    # This mirrors R-INLA's default safe behaviour. When the first attempt
    # succeeds there is effectively no extra overhead.
    runner = _core_safe if safe else _core
    return runner(
        model=model, family=family, data=data, quantiles=list(quantiles),
        E=E, offset=offset, scale=scale, weights=weights, Ntrials=Ntrials,
        strata=strata, lp_scale=lp_scale, link_covariates=link_covariates,
        verbose=verbose, lincomb=lincomb, selection=selection, control=control,
        inla_call=inla_call, inla_arg=inla_arg, num_threads=num_threads,
        keep=keep, working_directory=working_directory, silent=silent,
        inla_mode=inla_mode, debug=debug, reuse_filenames_from=reuse_filenames_from,
        dry_run=dry_run, collect_results=collect
    )


class _PyINLAInterface:
    """Restricted public-facing API surface for pyINLA."""

    @staticmethod
    def activate(code: str) -> bool:
        """
        Activate pyINLA with your license code.

        Parameters
        ----------
        code : str
            Your activation code provided by the pyINLA team.

        Returns
        -------
        bool
            True if activation successful.

        Example
        -------
        >>> from pyinla import pyinla
        >>> pyinla.activate("YOUR-ACTIVATION-CODE")
        True
        """
        return activate(code)

    def __call__(self, **kwargs) -> PyINLAResult:
        """Alias for run(...): allows calling `pyinla(...)` directly."""
        global _ANNOUNCED

        # Check activation and expiration before running
        _require_activation()

        if not _ANNOUNCED:
            print("[pyINLA] Running via Python interface (safety gate enabled).")
            _ANNOUNCED = True

        def _merge_control_alias(
            alias: str,
            key: str,
            example: str,
            *,
            allowed_keys: Optional[Iterable[str]] = None,
        ) -> None:
            block = kwargs.pop(alias, None)
            if block is None:
                return
            if not isinstance(block, dict):
                raise PyINLAError(f"{alias} must be a dict (e.g., {example}).")
            if allowed_keys is not None:
                allowed_keys_set = set(allowed_keys)
                extra = set(block.keys()) - allowed_keys_set
                if extra:
                    raise PyINLAError(
                        f"{alias} currently supports only {sorted(allowed_keys_set)} fields; "
                        f"found unsupported key(s): {', '.join(sorted(extra))}."
                    )
            control_obj = kwargs.get("control")
            if control_obj is None:
                control_obj = {}
            elif not isinstance(control_obj, dict):
                raise PyINLAError(f"'control' must be a dict when used with {alias}.")
            else:
                control_obj = dict(control_obj)
            merged_block = dict(control_obj.get(key) or {})
            merged_block.update(block)
            control_obj[key] = merged_block
            kwargs["control"] = control_obj

        # Accept shortcuts like `control_family={...}` / `control_fixed={...}`.
        _merge_control_alias("control_family", "family", "{'variant': 1}")
        _merge_control_alias(
            "control_fixed",
            "fixed",
            "{'prec.intercept': 1.0}",
            allowed_keys={
                "prec.intercept",
                "prec_intercept",
                "prec",
                "mean.intercept",
                "mean_intercept",
                "mean",
            },
        )
        try:
            # First check for untested arguments before other validations
            enforce_untested_arguments(kwargs)

            families = enforce_allowed_family(kwargs)
            enforce_gaussian_hyperstructure(kwargs, families=families)
            enforce_gamma_hyperstructure(kwargs, families=families)
            enforce_gamma_support(kwargs, families=families)
            enforce_logistic_hyperstructure(kwargs, families=families)
            enforce_loglogistic_hyperstructure(kwargs, families=families)
            enforce_sn_hyperstructure(kwargs, families=families)
            enforce_t_hyperstructure(kwargs, families=families)
            enforce_beta_hyperstructure(kwargs, families=families)
            enforce_scale_usage(kwargs, families=families)
            enforce_compute_section(kwargs)
            enforce_exposure_usage(kwargs, families=families)
            enforce_poisson_exposure(kwargs, families=families)
            enforce_nbinomial_exposure(kwargs, families=families)
            enforce_binomial_trials(kwargs, families=families)
            enforce_binomial_family_variant(kwargs, families=families)
            enforce_beta_support(kwargs, families=families)
            enforce_survival_response(kwargs, families=families)
            enforce_control_structure(kwargs, families=families)
            enforce_random_structure(kwargs)
            # Response value validation for various likelihoods
            enforce_poisson_support(kwargs, families=families)
            enforce_nbinomial_support(kwargs, families=families)
            enforce_binomial_support(kwargs, families=families)
            enforce_exponential_support(kwargs, families=families)
            enforce_lognormal_support(kwargs, families=families)
            enforce_weibull_support(kwargs, families=families)
            enforce_loglogistic_support(kwargs, families=families)
            enforce_gaussian_support(kwargs, families=families)
            enforce_logistic_support(kwargs, families=families)
            enforce_t_support(kwargs, families=families)
            enforce_sn_support(kwargs, families=families)
        except SafetyError as err:
            raise PyINLAError(str(err)) from err
        # Pass the safety token to prove we went through the safety gate
        return _run_impl(_safety_token=_SAFETY_GATE_TOKEN, **kwargs)

    def run(self, **kwargs) -> PyINLAResult:
        return self(**kwargs)

    def qinv(self, *args, **kwargs):
        return _qinv(*args, **kwargs)

    def rw1(self, *args, **kwargs):
        return _rw1(*args, **kwargs)

    def rw2(self, *args, **kwargs):
        return _rw2(*args, **kwargs)

    def scale_model(self, *args, **kwargs):
        return _scale_model(*args, **kwargs)

    def read_fmesher_file(self, *args, **kwargs):
        return _read_fmesher_file(*args, **kwargs)

    # Marginal utilities -------------------------------------------------
    def dmarginal(self, *args, **kwargs):
        return _dmarginal(*args, **kwargs)

    def emarginal(self, *args, **kwargs):
        return _emarginal(*args, **kwargs)

    def hpdmarginal(self, *args, **kwargs):
        return _hpdmarginal(*args, **kwargs)

    def mmarginal(self, *args, **kwargs):
        return _mmarginal(*args, **kwargs)

    def marginal_fix(self, *args, **kwargs):
        return _marginal_fix(*args, **kwargs)

    def marginal_transform(self, *args, **kwargs):
        return _marginal_transform(*args, **kwargs)

    def pmarginal(self, *args, **kwargs):
        return _pmarginal(*args, **kwargs)

    def qmarginal(self, *args, **kwargs):
        return _qmarginal(*args, **kwargs)

    def rmarginal(self, *args, **kwargs):
        return _rmarginal(*args, **kwargs)

    def smarginal(self, *args, **kwargs):
        return _smarginal(*args, **kwargs)

    def sfmarginal(self, *args, **kwargs):
        return _sfmarginal(*args, **kwargs)

    def spline(self, *args, **kwargs):
        return _spline(*args, **kwargs)

    def tmarginal(self, *args, **kwargs):
        return _tmarginal(*args, **kwargs)

    def zmarginal(self, *args, **kwargs):
        return _zmarginal(*args, **kwargs)


pyinla = _PyINLAInterface()

__all__ = ["pyinla"]

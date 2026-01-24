# options.py
"""
Global options for the Python port (R-INLA options analogue).

Key functions:
- inla_get_option_default()
- inla_get_option(option=None)
- inla_set_option(**kwargs)   # also supports inla_set_option("key", value)
- inla_enabled_INLAjoint_features()
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, Optional, Union

from .inla_call import inla_call_builtin, fmesher_call_builtin, inla_remote_script
from .utils import inla_anyMultibyteUTF8Characters as _any_mb


# Internal storage
_INLA_OPTIONS: Dict[str, Any] = {}


def _cpu_default_string() -> str:
    # Detect physical-ish cores is non-trivial; emulate R default loosely
    n = os.cpu_count() or 1
    n = max(1, min(16, n))
    return f"{n}:1"


def inla_get_option_default() -> Dict[str, Any]:
    """
    Default options (no recursion to get current options).
    """
    return {
        "inla.arg": None,
        "fmesher.arg": "",
        "num.threads": _cpu_default_string(),
        "smtp": "default",
        "safe": True,
        "keep": False,
        "verbose": False,
        "save.memory": False,
        "internal.opt": True,
        "working.directory": None,
        "silent": True,
        "debug": False,
        "show.warning.graph.file": True,
        "scale.model.default": False,
        "short.summary": False,
        "inla.timeout": 0,
        "fmesher.timeout": 0,
        "inla.mode": "compact",
        "malloc.lib": "mi",  # default like R code
        "fmesher.evolution": 2,
        "fmesher.evolution.warn": True,
        "fmesher.evolution.verbosity": "default",
        "INLAjoint.features": False,
        "numa": False,
        # Filled dynamically on get:
        # "inla.call": inla_call_builtin()
        # "fmesher.call": fmesher_call_builtin()
    }


def inla_get_option(option: Optional[Union[str, list]] = None):
    """
    Get current option(s). If option is None, return a dict of current values.
    Special handling for 'inla.call' and 'fmesher.call' to avoid recursion.
    """
    defaults = inla_get_option_default()

    # Make sure call paths exist (computed on demand)
    if "inla.call" not in _INLA_OPTIONS:
        defaults["inla.call"] = inla_call_builtin()
    if "fmesher.call" not in _INLA_OPTIONS:
        defaults["fmesher.call"] = fmesher_call_builtin()

    if option is None:
        # Full copy with current overrides
        out = {**defaults, **_INLA_OPTIONS}
        # Special case: if user explicitly set 'inla.call' to "remote"/"inla.remote"
        val = _INLA_OPTIONS.get("inla.call", None)
        if isinstance(val, str) and val.lower() in ("remote", "inla.remote"):
            rem = inla_remote_script()
            if rem:
                out["inla.call"] = rem
        return out

    # Normalize to list of keys
    if isinstance(option, str):
        keys = [option]
    else:
        keys = list(option)

    out_list = []
    for key in keys:
        if key in _INLA_OPTIONS:
            val = _INLA_OPTIONS[key]
        elif key in defaults:
            val = defaults[key]
            if key == "inla.call":
                val = _INLA_OPTIONS.get("inla.call", defaults["inla.call"])
                if isinstance(val, str) and val.lower() in ("remote", "inla.remote"):
                    rem = inla_remote_script()
                    if rem:
                        val = rem
            elif key == "fmesher.call":
                val = _INLA_OPTIONS.get("fmesher.call", defaults["fmesher.call"])
        else:
            raise KeyError(f"Unknown option '{key}'")
        out_list.append(val)

    return out_list[0] if isinstance(option, str) else out_list


def inla_set_option(*args, **kwargs) -> None:
    """
    Set global options.

    Supports:
      inla_set_option("keep", True)
      inla_set_option(keep=True, num_threads="4:1")
      inla_set_option(keep=True, num_threads="4:1", verbose=True)
    """
    # Positional signature ("key", value)
    if len(args) == 2 and not kwargs:
        k, v = args
        if not isinstance(k, str):
            raise TypeError("First positional argument must be a string key")
        _set_one(k, v)
    elif len(args) == 0 and kwargs:
        for k, v in kwargs.items():
            _set_one(k, v)
    else:
        raise TypeError("Use either inla_set_option('key', value) or inla_set_option(key=value, ...).")

    # Post-set sanity checks
    _post_set_checks()


def _set_one(option: str, value: Any) -> None:
    valid = set(inla_get_option_default().keys()) | {"inla.call", "fmesher.call"}
    if option not in valid:
        raise KeyError(f"Unknown option '{option}'")

    # Assign directly
    _INLA_OPTIONS[option] = value


def _post_set_checks() -> None:
    # Ensure inla.mode is valid; coerce 'experimental' back to 'compact' (as R code)
    mode = inla_get_option("inla.mode")
    if mode not in ("compact", "classic", "twostage", "experimental"):
        raise ValueError("Invalid 'inla.mode'. Must be 'compact', 'classic', 'twostage', or 'experimental'.")
    if mode == "experimental":
        _INLA_OPTIONS["inla.mode"] = "compact"

    # malloc.lib handling
    arg = inla_get_option("malloc.lib")
    if arg is None or arg == "default":
        _INLA_OPTIONS["malloc.lib"] = inla_get_option_default()["malloc.lib"]
    else:
        if isinstance(arg, str) and arg.startswith("/") and not os.path.exists(arg):
            warnings.warn(
                f"User-defined library for option 'malloc.lib', {arg}, does not exist. Using default."
            )
            _INLA_OPTIONS["malloc.lib"] = inla_get_option_default()["malloc.lib"]
        else:
            if arg not in ("default", "compiler", "je", "tc", "mi") and not os.path.exists(arg):
                # Keep but warn softly; availability checks depend on local packaging
                warnings.warn(
                    f"Value for option 'malloc.lib' = {arg!r} may not be available on this system."
                )

    # working.directory multibyte warning
    wdir = inla_get_option("working.directory")
    if isinstance(wdir, str) and _any_mb(wdir):
        warnings.warn(
            f"*** working.directory=[{wdir}] contains multibyte characters. This may fail on some systems."
        )


def inla_enabled_INLAjoint_features() -> bool:
    return bool(inla_get_option("INLAjoint.features"))

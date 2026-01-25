# inla_call.py
"""
Python equivalents of R-INLA's inla.call helpers.

We try to locate binaries relative to this package directory, with the same
layout used in R-INLA:

- mac (x86_64):   bin/mac/<32|64>bit/inla.run
- mac.arm64:      bin/mac.arm64/inla.run
- linux:          bin/linux/<32|64>bit/inla.mkl.run
- windows:        bin/windows/<32|64>bit/inla.exe

Same for fmesher: .../fmesher.run (or .exe on Windows)

Environment overrides:
- INLA_BIN        (path to inla.* binary)
- FMESHER_BIN     (path to fmesher.* binary)
- INLA_REMOTE_BIN (path to bin/remote/inla.remote)
"""

from __future__ import annotations

import os
import platform
from typing import Optional


def _bits() -> str:
    # 32 or 64 as a string. sys.maxsize is 2**63-1 on 64-bit CPython.
    return "64" if os.sys.maxsize > 2**32 else "32"


def _system() -> str:
    return platform.system().lower()  # 'darwin', 'linux', 'windows'


def _machine() -> str:
    return platform.machine().lower()  # 'x86_64', 'arm64', 'aarch64', ...


def _package_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _resolve_path(*parts: str) -> Optional[str]:
    path = os.path.join(_package_root(), *parts)
    return path if os.path.exists(path) else None


def inla_call_builtin() -> str:
    """Return the built-in path to the inla executable for this OS/arch.

    If the binary is not found, automatically downloads it from R-INLA servers.
    """
    env = os.environ.get("INLA_BIN")
    if env and os.path.exists(env):
        return env

    sysname = _system()
    bits = _bits()
    mach = _machine()

    if sysname == "darwin":
        # mac vs mac.arm64
        if mach in ("arm64", "aarch64"):
            path = _resolve_path("bin", "mac.arm64", "inla.run")
        else:
            path = _resolve_path("bin", "mac", f"{bits}bit", "inla.run")
    elif sysname == "linux":
        path = _resolve_path("bin", "linux", f"{bits}bit", "inla.mkl.run")
    elif sysname.startswith("win"):
        path = _resolve_path("bin", "windows", f"{bits}bit", "inla.exe")
    else:
        raise RuntimeError("Unknown OS")

    if path:
        return path

    # Binary not found - auto-download from R-INLA servers
    from .binary_manager import ensure_binary
    binary_path = ensure_binary()
    return str(binary_path)


def inla_call_no_remote() -> str:
    """
    Return the current inla.call (if set), except if it is 'remote'/'inla.remote',
    then return the built-in path instead.
    (This is wired into options.get to avoid recursion.)
    """
    from .options import inla_get_option  # local import to avoid cycles
    call = inla_get_option("inla.call")
    if call is None:
        return inla_call_builtin()
    if str(call).lower() in ("remote", "inla.remote"):
        return inla_call_builtin()
    return call


def fmesher_call_builtin() -> str:
    """Return the built-in path to the fmesher executable for this OS/arch."""
    env = os.environ.get("FMESHER_BIN")
    if env and os.path.exists(env):
        return env

    sysname = _system()
    bits = _bits()
    mach = _machine()

    if sysname == "darwin":
        if mach in ("arm64", "aarch64"):
            path = _resolve_path("bin", "mac.arm64", "fmesher.run")
        else:
            path = _resolve_path("bin", "mac", f"{bits}bit", "fmesher.run")
    elif sysname == "linux":
        path = _resolve_path("bin", "linux", f"{bits}bit", "fmesher.run")
    elif sysname.startswith("win"):
        path = _resolve_path("bin", "windows", f"{bits}bit", "fmesher.exe")
    else:
        raise RuntimeError("Unknown OS")

    if path:
        return path
    raise FileNotFoundError("INLA installation error; fmesher binary not found for this OS/arch.")


def inla_remote_script() -> Optional[str]:
    """
    Return the path to the remote launcher script if present, else None.
    """
    env = os.environ.get("INLA_REMOTE_BIN")
    if env and os.path.exists(env):
        return env
    path = _resolve_path("bin", "remote", "inla.remote")
    return path

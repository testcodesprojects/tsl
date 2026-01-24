# os.py
"""
OS and architecture helpers mirroring the behavior of os.R:
- inla.os(type = c('linux','mac','mac.arm64','windows','else'))
- inla.os.type()
- inla.os.32or64bit(), inla.os.is.32bit(), inla.os.is.64bit()
"""

from __future__ import annotations

import os
import platform
import struct
import subprocess
from typing import Literal

__all__ = [
    "inla_os",
    "inla_os_type",
    "inla_os_32or64bit",
    "inla_os_is_32bit",
    "inla_os_is_64bit",
]


def _is_macos_dirs_present() -> bool:
    ok = os.path.isdir("/Library") and os.path.isdir("/Applications")
    return bool(ok)


def _macos_product_version_major_minor() -> float | None:
    """
    Returns major + minor/10 as in the R code (e.g., 10.15 -> 10.1 + 0.5 = 10.15 effectively).
    If sw_vers is unavailable, returns None.
    """
    try:
        out = subprocess.check_output(
            ["sw_vers", "-productVersion"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None
    parts = out.split(".")
    if not parts:
        return None
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except Exception:
        return None
    return major + minor / 10.0


def inla_os(
    type: Literal["linux", "mac", "mac.arm64", "windows", "else"]
) -> bool:
    """
    Identify operating system type following the R logic.

    Notes:
    - 'mac.arm64': requires macOS presence + machine == 'arm64'
    - 'mac': requires macOS presence + machine != 'arm64' (Intel or Rosetta)
    - 'linux': Unix but not mac/mac.arm64
    """
    if type == "windows":
        return platform.system().lower() == "windows"

    if type == "mac.arm64":
        result = _is_macos_dirs_present()
        if result is None:
            result = False
        mach = platform.machine().lower()
        result = bool(result and (mach == "arm64" or mach == "aarch64"))
        return result

    if type == "mac":
        result = _is_macos_dirs_present()
        if not result:
            return False
        # Optional warning logic for older versions (mirrors R’s message)
        ver = _macos_product_version_major_minor()
        s_req = 10.15
        if ver is not None and ver < s_req:
            # match R's behavior of warning; here we just log to stderr via print
            print(
                f"Warning: Your macOS version ({ver}) might be too old for some R-INLA binaries (built on {s_req})."
            )
        mach = platform.machine().lower()
        return mach not in {"arm64", "aarch64"}

    if type == "linux":
        is_unix = os.name == "posix" and platform.system().lower() == "linux"
        return bool(is_unix and not inla_os("mac") and not inla_os("mac.arm64"))

    if type == "else":
        return True

    raise ValueError("This shouldn't happen.")


def inla_os_type() -> str:
    for os_name in ("windows", "mac", "mac.arm64", "linux", "else"):
        if inla_os(os_name):  # type: ignore[arg-type]
            return os_name
    raise RuntimeError("This shouldn't happen.")


def inla_os_32or64bit() -> str:
    """Return '32' or '64' based on pointer size, as in R's .Machine$sizeof.pointer."""
    bits = struct.calcsize("P") * 8
    return "64" if bits == 64 else "32"


def inla_os_is_32bit() -> bool:
    return inla_os_32or64bit() == "32"


def inla_os_is_64bit() -> bool:
    return inla_os_32or64bit() == "64"

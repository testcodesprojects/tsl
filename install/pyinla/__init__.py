"""
pyinla package entry point.
"""

from ._api import pyinla
from ._api import PyINLAError, PyINLACrashError, PyINLACollectError
from ._api import PyINLAResult  # retained for backward compatibility
from .surv import inla_surv
from .marginal_utils import (
    inla_smarginal as smarginal,
    inla_dmarginal as dmarginal,
    inla_pmarginal as pmarginal,
    inla_qmarginal as qmarginal,
    inla_emarginal as emarginal,
    inla_rmarginal as rmarginal,
    inla_tmarginal as tmarginal,
    inla_mmarginal as mmarginal,
    inla_zmarginal as zmarginal,
    inla_hpdmarginal as hpdmarginal,
)
from .binary_manager import (
    download_binary,
    list_available_binaries,
    is_binary_installed,
)

__all__ = [
    "pyinla",
    "inla_surv",
    "PyINLAError",
    "PyINLACrashError",
    "PyINLACollectError",
    "PyINLAResult",
    # Marginal utility functions
    "smarginal",
    "dmarginal",
    "pmarginal",
    "qmarginal",
    "emarginal",
    "rmarginal",
    "tmarginal",
    "mmarginal",
    "zmarginal",
    "hpdmarginal",
    # Binary management
    "download_binary",
    "list_available_binaries",
    "is_binary_installed",
]

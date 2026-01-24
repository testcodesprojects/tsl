"""Exception classes for the fmesher package."""


class FmesherError(Exception):
    """Base exception for fmesher errors."""
    pass


class Rpy2NotAvailableError(FmesherError):
    """Raised when rpy2 is not installed."""
    pass


class FmesherNotAvailableError(FmesherError):
    """Raised when the fmesher R package is not installed."""
    pass

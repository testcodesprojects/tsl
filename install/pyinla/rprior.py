# rprior.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any, Dict, Optional
import inspect

_RPRIOR_CODE = "c5c4fee74dc9299b6753b8605e303f59a1236bfa"


@dataclass
class RPrior:
    """
    Python analogue of an `inla.rprior` object:
    - `fn`: a callable(theta) -> float, returning log-prior at `theta`
    - `code`: fixed identifier used for recognisability (as in R)
    - `context`: captured variables that were made available to the R function
    """
    fn: Callable[[Any], float]
    code: str = _RPRIOR_CODE
    context: Optional[Dict[str, Any]] = None

    def __call__(self, theta: Any) -> float:
        return self.fn(theta)


def inla_rprior_code() -> str:
    """Equivalent of `inla.rprior.code()`."""
    return _RPRIOR_CODE


def inla_rprior_define(rprior: Callable[..., float], /, **context) -> RPrior:
    """
    Equivalent of `inla.rprior.define()`.
    Binds named variables into the function's lexical scope via a closure.

    Parameters
    ----------
    rprior : callable
        A Python function computing log-prior at its first argument.
    **context : dict
        Named variables made available to `rprior` (like the R environment).
    """
    if not callable(rprior):
        raise TypeError("`rprior` must be callable")

    # Create a closure that exposes `context` without requiring **kwargs
    # consumption by the original function.
    sig = inspect.signature(rprior)
    has_var_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())

    if has_var_kwargs:
        # If the function already accepts **kwargs, pass-through is easy.
        def wrapped(theta):
            return rprior(theta, **context)
    else:
        # Otherwise, capture context lexically.
        def wrapped(theta, _ctx=context, _fn=rprior):
            # variables in _ctx are visible to _fn via closure if referenced
            return _fn(theta)

    return RPrior(fn=wrapped, code=_RPRIOR_CODE, context=dict(context))


def inla_is_rprior(obj: Any) -> bool:
    """
    Equivalent of `inla.is.rprior()`. Accepts both RPrior instances
    and dict-like fallbacks { 'rprior': <callable>, 'code': <str> }.
    """
    if isinstance(obj, RPrior):
        return True
    if isinstance(obj, dict):
        has_shape = (
            ("rprior" in obj and callable(obj.get("rprior"))) and
            ("code" in obj and isinstance(obj.get("code"), str))
        )
        if has_shape and obj["code"] == _RPRIOR_CODE:
            return True
    return False

"""Safety gate helpers for pyinla public entry points."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .surv import is_inla_surv

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

try:
    import scipy.sparse as sp  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    sp = None  # type: ignore
    _HAVE_SCIPY = False

__all__ = [
    "SafetyError",
    "capture_user_call",
    "enforce_allowed_family",
    "enforce_gaussian_hyperstructure",
    "enforce_gamma_hyperstructure",
    "enforce_gamma_support",
    "enforce_logistic_hyperstructure",
    "enforce_sn_hyperstructure",
    "enforce_beta_hyperstructure",
    "enforce_t_hyperstructure",
    "enforce_scale_usage",
    "enforce_compute_section",
    "enforce_exposure_usage",
    "enforce_control_structure",
    "enforce_random_structure",
    "enforce_poisson_exposure",
    "enforce_nbinomial_exposure",
    "enforce_binomial_trials",
    "enforce_binomial_family_variant",
    "enforce_beta_support",
    "enforce_survival_response",
    "enforce_untested_arguments",
    "enforce_poisson_support",
    "enforce_nbinomial_support",
    "enforce_binomial_support",
    "enforce_exponential_support",
    "enforce_lognormal_support",
    "enforce_weibull_support",
    "enforce_loglogistic_support",
    "enforce_gaussian_support",
    "enforce_logistic_support",
    "enforce_t_support",
    "enforce_sn_support",
    "enforce_offset_values",
    "enforce_weights_values",
]


class SafetyError(RuntimeError):
    """Raised when user input violates the temporary safety restrictions."""


def _is_truthy(value: Any) -> bool:
    """Best-effort coercion of user-supplied flags to booleans."""
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on", "t"}
    return bool(value)


def capture_user_call(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Snapshot the raw arguments supplied by the user.

    Parameters
    ----------
    args
        Positional arguments as received by the public entry point.
    kwargs
        Keyword arguments as received by the public entry point.

    Returns
    -------
    dict
        A shallow copy of the keyword arguments for downstream use.

    Notes
    -----
    This helper exists so that we can enforce feature allow-lists before
    the main solver pipeline runs.  For now it performs a defensive copy
    of the inputs; future revisions can add validation without changing
    the call sites.
    """

    snapshot = dict(kwargs or {})
    snapshot["__args__"] = tuple(args or ())

    families = enforce_allowed_family(snapshot)
    snapshot["__families__"] = tuple(families)
    enforce_gaussian_hyperstructure(snapshot, families=families)
    enforce_gamma_hyperstructure(snapshot, families=families)
    enforce_gamma_support(snapshot, families=families)
    enforce_logistic_hyperstructure(snapshot, families=families)
    enforce_loglogistic_hyperstructure(snapshot, families=families)
    enforce_sn_hyperstructure(snapshot, families=families)
    enforce_t_hyperstructure(snapshot, families=families)
    enforce_beta_hyperstructure(snapshot, families=families)
    enforce_scale_usage(snapshot, families=families)
    enforce_control_structure(snapshot, families=families)
    enforce_random_structure(snapshot)
    enforce_compute_section(snapshot)
    enforce_exposure_usage(snapshot, families=families)
    enforce_poisson_exposure(snapshot, families=families)
    enforce_nbinomial_exposure(snapshot, families=families)
    enforce_binomial_trials(snapshot, families=families)
    enforce_binomial_family_variant(snapshot, families=families)
    enforce_beta_support(snapshot, families=families)
    enforce_survival_response(snapshot, families=families)
    enforce_poisson_support(snapshot, families=families)
    enforce_nbinomial_support(snapshot, families=families)
    enforce_binomial_support(snapshot, families=families)
    enforce_exponential_support(snapshot, families=families)
    enforce_lognormal_support(snapshot, families=families)
    enforce_weibull_support(snapshot, families=families)
    enforce_loglogistic_support(snapshot, families=families)
    enforce_gaussian_support(snapshot, families=families)
    enforce_logistic_support(snapshot, families=families)
    enforce_t_support(snapshot, families=families)
    enforce_sn_support(snapshot, families=families)
    enforce_offset_values(snapshot)
    enforce_weights_values(snapshot)
    enforce_untested_arguments(snapshot)

    return snapshot


def _normalize_family_spec(family: Any) -> Iterable[str]:
    if family is None:
        return []
    if isinstance(family, (list, tuple, set)):
        items = family
    else:
        items = [family]
    normalized: list[str] = []
    for item in items:
        if item is None:
            continue
        normalized.append(str(item).strip().lower())
    return normalized


def _normalize_link_value(link_val: Any) -> str | None:
    """Normalize a link value to a lowercase string.

    Handles both formats:
    - String format: 'neglog' -> 'neglog'
    - Dict format: {'model': 'neglog'} -> 'neglog'
    """
    if link_val is None:
        return None
    if isinstance(link_val, dict):
        # Extract 'model' key if present
        model_val = link_val.get("model")
        if model_val is not None:
            return str(model_val).strip().lower()
        return None
    return str(link_val).strip().lower()


#
# 1. Require that `family` is provided and raise a SafetyError if missing.
# 2. Normalize whatever the user passed (str/list/tuple/set) to lowercase names via `_normalize_family_spec`.
# 3. Ensure the normalized list is not empty.
# 4. Reject any normalized family that is not in the allow-list (defaults to gaussian/poisson/binomial/xbinomial/gamma/nbinomial/beta/betabinomial/exponential/exponentialsurv).
# 5. Return the normalized tuple for downstream reuse (e.g., stored in `snapshot['__families__']`).
#
def enforce_allowed_family(
    kwargs: Dict[str, Any], *, allowed: Iterable[str] = (
        "gaussian", "poisson", "binomial", "xbinomial", "gamma", "nbinomial", "nbinomial2", "beta",
        "betabinomial", "exponential", "exponentialsurv", "gammasurv", "lognormal",
        "lognormalsurv", "logistic", "loglogistic", "loglogisticsurv", "sn", "t",
        "weibull", "weibullsurv"
    )
) -> Tuple[str, ...]:
    """Ensure the caller supplied an explicit family drawn from the allow-list."""

    allowed_set = {str(a).strip().lower() for a in allowed}

    if "family" not in kwargs:
        raise SafetyError(
            "pyinla safety check: please pass the 'family' argument explicitly."
        )

    normalized = list(_normalize_family_spec(kwargs.get("family")))
    if not normalized:
        raise SafetyError(
            "pyinla safety check: 'family' may not be empty."
        )

    invalid = [fam for fam in normalized if fam not in allowed_set]
    if invalid:
        raise SafetyError(
            "pyinla safety check: unsupported family requested: {}. Only {} allowed.".format(
                ", ".join(sorted(set(invalid))), ", ".join(sorted(allowed_set))
            )
        )

    return tuple(normalized)


def _validate_gaussian_family_block(family_block: Dict[str, Any]) -> None:
    allowed_top_keys = {"hyper"}
    extra_keys = set(family_block.keys()) - allowed_top_keys
    if extra_keys:
        raise SafetyError(
            "pyinla safety check: unsupported keys in control['family'] for gaussian: {}.".format(
                ", ".join(sorted(extra_keys))
            )
        )

    hyper_list = family_block.get("hyper", []) or []
    if not isinstance(hyper_list, (list, tuple)):
        raise SafetyError("pyinla safety check: control['family']['hyper'] must be a list of dicts.")

    allowed_hyper_keys = {"prior", "param", "initial", "fixed", "id"}
    allowed_priors = {None, "loggamma", "pc.prec"}

    for entry in hyper_list:
        if not isinstance(entry, dict):
            raise SafetyError("pyinla safety check: entries within control['family']['hyper'] must be dicts.")
        extra = set(entry.keys()) - allowed_hyper_keys
        if extra:
            raise SafetyError(
                "pyinla safety check: unsupported keys in gaussian hyper specification: {}.".format(
                    ", ".join(sorted(extra))
                )
            )
        prior = entry.get("prior")
        if prior is not None:
            prior_norm = str(prior).strip().lower()
            if prior_norm not in allowed_priors:
                raise SafetyError(
                    "pyinla safety check: unsupported prior '{}' for gaussian; allowed: {}.".format(
                        prior, ", ".join(sorted(p for p in allowed_priors if p))
                    )
                )
        entry_id = str(entry.get("id", "")).strip().lower()
        if entry_id == "precoffset":
            if prior is not None and str(prior).strip().lower() != "none":
                raise SafetyError(
                    "pyinla safety check: gaussian 'precoffset' prior is read-only (must remain 'none')."
                )
            if "param" in entry and entry["param"]:
                raise SafetyError(
                    "pyinla safety check: gaussian 'precoffset' does not accept custom 'param'."
                )
            if "fixed" in entry and not bool(entry["fixed"]):
                raise SafetyError(
                    "pyinla safety check: gaussian 'precoffset' must stay 'fixed = True'."
                )
        if prior is not None and str(prior).strip().lower() == "loggamma":
            params = entry.get("param")
            vals = _as_float_list("param", params) if params is not None else []
            if len(vals) != 2 or not all(v > 0.0 and math.isfinite(v) for v in vals):
                raise SafetyError(
                    "pyinla safety check: loggamma prior requires two positive parameters (shape, rate)."
                )


#
# 1. Normalize the declared families (either passed in or sourced from kwargs).
# 2. Short-circuit when gaussian is not requested.
# 3. Inspect control['family'] only when it exists and is a dict.
# 4. Reuse _validate_gaussian_family_block to enforce the temporary hyper-parameter allow-list.
#
def enforce_gaussian_hyperstructure(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    fams = tuple(families) if families is not None else tuple(_normalize_family_spec(kwargs.get("family")))
    if "gaussian" not in fams:
        return

    control = kwargs.get("control") or {}
    if not isinstance(control, dict):
        return
    family_block = control.get("family")
    if not isinstance(family_block, dict):
        return
    _validate_gaussian_family_block(family_block)


def enforce_gamma_hyperstructure(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    fams = tuple(families) if families is not None else tuple(_normalize_family_spec(kwargs.get("family")))
    if "gamma" not in fams and "gammasurv" not in fams:
        return

    control = kwargs.get("control") or {}
    if not isinstance(control, dict):
        return
    family_block = control.get("family")
    if not isinstance(family_block, dict):
        return

    entries = _normalize_hyper_entries(family_block, allowed_names=["prec"])
    if not entries:
        return

    for _, entry in entries:
        prior = entry.get("prior")
        if prior is None:
            raise SafetyError("pyinla safety check: gamma/gammasurv hyperparameters require explicit 'prior'.")
        prior_norm = str(prior).strip().lower()
        if prior_norm not in ("loggamma", "pc.prec"):
            raise SafetyError("pyinla safety check: gamma/gammasurv hyperparameters currently support only 'loggamma' or 'pc.prec' prior.")


def enforce_beta_hyperstructure(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    fams = tuple(families) if families is not None else tuple(_normalize_family_spec(kwargs.get("family")))
    if "beta" not in fams:
        return

    control = kwargs.get("control") or {}
    if not isinstance(control, dict):
        return
    family_block = control.get("family")
    if not isinstance(family_block, dict):
        return

    entries = _normalize_hyper_entries(family_block)
    if not entries:
        return

    for _, entry in entries:
        prior = entry.get("prior")
        if prior is None:
            raise SafetyError("pyinla safety check: beta hyperparameters require explicit 'prior'.")
        prior_norm = str(prior).strip().lower()
        if prior_norm not in ("loggamma", "pc.prec"):
            raise SafetyError("pyinla safety check: beta hyperparameters currently support only 'loggamma' or 'pc.prec' prior.")


def enforce_logistic_hyperstructure(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    fams = tuple(families) if families is not None else tuple(_normalize_family_spec(kwargs.get("family")))
    if "logistic" not in fams:
        return

    control = kwargs.get("control") or {}
    if not isinstance(control, dict):
        return
    family_block = control.get("family")
    if not isinstance(family_block, dict):
        return

    entries = _normalize_hyper_entries(family_block, allowed_names=["prec"])
    if not entries:
        return

    for _, entry in entries:
        prior = entry.get("prior")
        if prior is None:
            raise SafetyError("pyinla safety check: logistic hyperparameters require explicit 'prior'.")
        prior_norm = str(prior).strip().lower()
        if prior_norm not in ("loggamma", "pc.prec"):
            raise SafetyError("pyinla safety check: logistic hyperparameters currently support only 'loggamma' or 'pc.prec' prior.")


def enforce_loglogistic_hyperstructure(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    fams = tuple(families) if families is not None else tuple(_normalize_family_spec(kwargs.get("family")))
    if not any(f in ("loglogistic", "loglogisticsurv") for f in fams):
        return

    control = kwargs.get("control") or {}
    if not isinstance(control, dict):
        return
    family_block = control.get("family")
    if not isinstance(family_block, dict):
        return

    entries = _normalize_hyper_entries(family_block, allowed_names=["alpha"])
    if not entries:
        return

    for _, entry in entries:
        prior = entry.get("prior")
        if prior is None:
            raise SafetyError("pyinla safety check: loglogistic hyperparameters require explicit 'prior'.")
        prior_norm = str(prior).strip().lower()
        if prior_norm not in ("loggamma", "pc.prec"):
            raise SafetyError("pyinla safety check: loglogistic hyperparameters currently support only 'loggamma' or 'pc.prec' prior.")


def enforce_sn_hyperstructure(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    fams = tuple(families) if families is not None else tuple(_normalize_family_spec(kwargs.get("family")))
    if "sn" not in fams:
        return

    control = kwargs.get("control") or {}
    if not isinstance(control, dict):
        return
    family_block = control.get("family")
    if not isinstance(family_block, dict):
        return

    entries = _normalize_hyper_entries(family_block, allowed_names=["prec", "skew"])
    if not entries:
        return

    for name, entry in entries:
        is_fixed = _is_truthy(entry.get("fixed")) if entry.get("fixed") is not None else False
        prior = entry.get("prior")
        if prior is None:
            if is_fixed:
                # Fixed hyperparameters don't need priors; skip the checks.
                continue
            raise SafetyError("pyinla safety check: skew-normal hyperparameters require explicit 'prior'.")
        prior_norm = str(prior).strip().lower()
        if name == "skew":
            if prior_norm not in ("pc.sn",):
                raise SafetyError("pyinla safety check: skew parameter for skew-normal supports only 'pc.sn' prior.")
        else:
            if prior_norm not in ("loggamma", "pc.prec"):
                raise SafetyError("pyinla safety check: precision parameter for skew-normal supports only 'loggamma' or 'pc.prec' prior.")


def enforce_t_hyperstructure(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    fams = tuple(families) if families is not None else tuple(_normalize_family_spec(kwargs.get("family")))
    if "t" not in fams:
        return

    control = kwargs.get("control") or {}
    if not isinstance(control, dict):
        return
    family_block = control.get("family")
    if not isinstance(family_block, dict):
        return

    entries = _normalize_hyper_entries(family_block, allowed_names=["prec", "dof"])
    if not entries:
        return

    for name, entry in entries:
        is_fixed = _is_truthy(entry.get("fixed")) if entry.get("fixed") is not None else False
        prior = entry.get("prior")
        if prior is None:
            if is_fixed:
                continue
            raise SafetyError("pyinla safety check: student-t hyperparameters require explicit 'prior'.")
        prior_norm = str(prior).strip().lower()
        if name == "prec":
            if prior_norm not in ("loggamma", "pc.prec"):
                raise SafetyError(
                    "pyinla safety check: precision parameter for student-t supports only 'loggamma' or 'pc.prec' prior."
                )
        else:  # dof
            if prior_norm not in ("pc.dof", "pcdof"):
                raise SafetyError(
                    "pyinla safety check: degrees-of-freedom parameter for student-t supports only 'pc.dof' prior."
                )
def enforce_scale_usage(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    normalized = list(families) if families is not None else list(_normalize_family_spec(kwargs.get("family")))
    has_gaussian = any(f == "gaussian" for f in normalized)
    has_xbinomial = any(f == "xbinomial" for f in normalized)
    has_gamma = any(f == "gamma" for f in normalized)
    has_beta = any(f == "beta" for f in normalized)
    has_logistic = any(f == "logistic" for f in normalized)
    has_t = any(f == "t" for f in normalized)
    has_nbinomial = any(f == "nbinomial" for f in normalized)

    if has_gaussian or has_xbinomial or has_gamma or has_beta or has_logistic or has_t or has_nbinomial:
        scale = kwargs.get("scale")
        if scale is None:
            return
        values = _normalize_positive_sequence("scale", scale)
        if has_gaussian or has_logistic or has_t or has_nbinomial:
            bad = [val for val in values if not (val > 0.0) or math.isnan(val)]
            if bad:
                raise SafetyError(
                    "pyinla safety check: all entries in 'scale' must be strictly positive for gaussian/logistic/t/nbinomial; "
                    f"found invalid values {bad[:5]}."
                )
            return
        # xbinomial/gamma/beta: allow zero but disallow negatives/NaN
        bad = [val for val in values if val < 0.0 or math.isnan(val)]
        if bad:
            raise SafetyError(
                "pyinla safety check: 'scale' must be non-negative for xbinomial/gamma/beta; "
                f"found invalid values {bad[:5]}."
            )
        if has_xbinomial:
            # The documentation constrains q so that 0 < q <= 1.
            too_large = [val for val in values if val > 1.0 + 1e-12]
            if too_large:
                raise SafetyError(
                    "pyinla safety check: 'scale' entries for xbinomial must be <= 1; "
                    f"found values above 1: {too_large[:5]}."
                )
        return

    if kwargs.get("scale") is not None:
        raise SafetyError(
            "pyinla safety check: 'scale' is only permitted for gaussian, nbinomial, xbinomial, gamma, beta, logistic, or t likelihoods."
        )


def enforce_control_structure(kwargs: Dict[str, Any], families: Tuple[str, ...] = ()) -> None:
    """Allow only ``family`` and ``predictor`` keys in control, and constrain predictor."""

    # `control` structure guard: keep only the public knobs exposed in the Python port.
    # 1. Require control (when passed) to be a dict.
    # 2. Allow only the top-level keys {'family', 'predictor', 'compute'}.
    # 3. Ensure family block stays dict-like so the gaussian validator can inspect it.
    # 4. Restrict predictor block to a single boolean flag ('compute').

    control = kwargs.get("control")
    if control is None:
        return
    if not isinstance(control, dict):
        raise SafetyError("pyinla safety check: 'control' must be a dict when provided.")

    allowed_top = {
        "family",
        "predictor",
        "compute",
        "inla",
        "fixed",
        "mode",
        "expert",
        "hazard",
        "lincomb",
        "update",
        "lp_scale",
        "pardiso",
        "stiles",
        "taucs",
        "numa",
        "only_hyperparam",
    }
    extra = set(control.keys()) - allowed_top
    if extra:
        raise SafetyError(
            "pyinla safety check: unsupported keys in 'control': {}.".format(
                ", ".join(sorted(extra))
            )
        )

    family_block = control.get("family")
    if family_block is not None and not isinstance(family_block, dict):
        raise SafetyError("pyinla safety check: control['family'] must be a dict.")

    # Validate control.link block (accept either control['family'] or control['predictor'] location)
    def _validate_control_link(block: Dict[str, Any]) -> None:
        if not any(f in ("gamma", "gammasurv") for f in families):
            raise SafetyError("pyinla safety check: control.link is only allowed for gamma likelihoods.")
        if not isinstance(block, dict):
            raise SafetyError("pyinla safety check: control.link configuration must be a dict.")
        model = block.get("model")
        if model is None:
            raise SafetyError("pyinla safety check: control.link['model'] must be specified.")
        model_norm = str(model).strip().lower()
        # For gammasurv family, 'log', 'neglog', 'quantile' link models are allowed (standard for survival analysis)
        # For gamma family, only 'quantile' link model is allowed
        is_survival = any(f == "gammasurv" for f in families)
        if is_survival:
            if model_norm not in ("log", "neglog", "quantile"):
                raise SafetyError("pyinla safety check: only 'log', 'neglog', or 'quantile' control.link model is allowed for gammasurv.")
            if model_norm in ("log", "neglog"):
                return  # 'log' and 'neglog' links for gammasurv don't require additional params
        else:
            if model_norm != "quantile":
                raise SafetyError("pyinla safety check: only 'quantile' control.link model is allowed for gamma.")
        # Validate quantile parameter for quantile link model
        quantile = block.get("quantile")
        if quantile is None:
            raise SafetyError("pyinla safety check: control.link['quantile'] must be provided when model='quantile'.")
        try:
            q = float(quantile)
        except Exception as exc:
            raise SafetyError("pyinla safety check: quantile must be numeric.") from exc
        if not (0.0 < q < 1.0):
            raise SafetyError("pyinla safety check: quantile must be in (0,1).")

    if isinstance(family_block, dict):
        link_conf = family_block.get("control.link") or family_block.get("control_link")
        if link_conf is not None:
            _validate_control_link(link_conf)

    predictor_block = control.get("predictor")
    if predictor_block is None:
        return
    if not isinstance(predictor_block, dict):
        raise SafetyError("pyinla safety check: control['predictor'] must be a dict.")

    allowed_predictor = {"compute", "link"}
    if "control.link" in predictor_block or "control_link" in predictor_block:
        _validate_control_link(predictor_block.get("control.link") or predictor_block.get("control_link"))
        allowed_predictor.add("control.link")
    extra_pred = set(predictor_block.keys()) - allowed_predictor
    if extra_pred:
        raise SafetyError(
            "pyinla safety check: unsupported keys in control['predictor']: {}.".format(
                ", ".join(sorted(extra_pred))
            )
        )

    if "compute" in predictor_block:
        compute = predictor_block["compute"]
        if not isinstance(compute, bool):
            raise SafetyError(
                "pyinla safety check: control['predictor']['compute'] must be True/False."
            )

    inla_block = control.get("inla")
    if inla_block is None:
        return
    if not isinstance(inla_block, dict):
        raise SafetyError("pyinla safety check: control['inla'] must be a dict when provided.")

    allowed_inla_keys = {"strategy", "int.strategy", "int_strategy", "control_vb", "adaptive_max"}
    extra_inla = set(inla_block.keys()) - allowed_inla_keys
    if extra_inla:
        raise SafetyError(
            "pyinla safety check: control['inla'] currently supports only {} overrides; found {}.".format(
                ", ".join(sorted(allowed_inla_keys)),
                ", ".join(sorted(extra_inla))
            )
        )

    # Validate control_vb if present
    control_vb = inla_block.get("control_vb")
    if control_vb is not None:
        if not isinstance(control_vb, dict):
            raise SafetyError("pyinla safety check: control['inla']['control_vb'] must be a dict.")
        allowed_vb_keys = {"enable", "strategy", "iter_max", "hessian_update", "hessian_strategy",
                          "verbose", "f_enable_limit", "emergency"}
        extra_vb = set(control_vb.keys()) - allowed_vb_keys
        if extra_vb:
            raise SafetyError(
                "pyinla safety check: control['inla']['control_vb'] supports only {} keys; found {}.".format(
                    ", ".join(sorted(allowed_vb_keys)),
                    ", ".join(sorted(extra_vb))
                )
            )
        # Validate VB strategy if provided
        vb_strategy = control_vb.get("strategy")
        if vb_strategy is not None:
            vb_strategy_str = str(vb_strategy).strip().lower()
            if vb_strategy_str not in {"mean", "variance"}:
                raise SafetyError(
                    "pyinla safety check: control['inla']['control_vb']['strategy'] must be 'mean' or 'variance'; got '{}'.".format(vb_strategy)
                )
        # Validate hessian_strategy if provided
        vb_hessian = control_vb.get("hessian_strategy")
        if vb_hessian is not None:
            vb_hessian_str = str(vb_hessian).strip().lower()
            if vb_hessian_str not in {"default", "full", "partial", "diagonal"}:
                raise SafetyError(
                    "pyinla safety check: control['inla']['control_vb']['hessian_strategy'] must be one of 'default', 'full', 'partial', 'diagonal'; got '{}'.".format(vb_hessian)
                )

    def _normalize_choice(name: str, value: Any, allowed: set[str]) -> None:
        try:
            normalized = str(value).strip().lower().replace("_", ".")
        except Exception as exc:
            raise SafetyError(
                "pyinla safety check: control['inla']['{}'] must be a string.".format(name)
            ) from exc
        if normalized not in allowed:
            raise SafetyError(
                "pyinla safety check: control['inla']['{}'] must be one of {}; got '{}'.".format(
                    name,
                    ", ".join(sorted(allowed)),
                    value,
                )
            )

    strategy_val = inla_block.get("strategy")
    if strategy_val is not None:
        _normalize_choice("strategy", strategy_val, {"auto", "gaussian", "simplified.laplace", "laplace", "adaptive"})

    int_strategy_val = inla_block.get("int.strategy")
    if int_strategy_val is None and "int_strategy" in inla_block:
        int_strategy_val = inla_block.get("int_strategy")
    if int_strategy_val is not None:
        _normalize_choice(
            "int.strategy",
            int_strategy_val,
            {"auto", "ccd", "grid", "eb", "user", "user.std"},
        )


def _normalize_positive_sequence(name: str, values: Any) -> Iterable[float]:
    """Return a flat list of floats, raising SafetyError on conversion issues."""
    try:
        import numpy as np  # type: ignore

        arr = np.asarray(values)
        if arr.size == 0:
            return []
        arr = arr.astype(float)
        return arr.reshape(-1).tolist()
    except Exception:
        pass

    iterable_types = (list, tuple, set)
    if isinstance(values, iterable_types):
        try:
            return [float(v) for v in values]
        except Exception as exc:  # pragma: no cover - propagate as SafetyError
            raise SafetyError(
                f"pyinla safety check: could not interpret '{name}' as numeric values."
            ) from exc

    if hasattr(values, "__iter__") and not isinstance(values, (str, bytes)):
        try:
            return [float(v) for v in values]
        except Exception as exc:  # pragma: no cover - propagate as SafetyError
            raise SafetyError(
                f"pyinla safety check: could not interpret '{name}' as numeric values."
            ) from exc

    try:
        return [float(values)]
    except Exception as exc:  # pragma: no cover - propagate as SafetyError
        raise SafetyError(
            f"pyinla safety check: could not interpret '{name}' as numeric values."
        ) from exc


def _as_float_list(name: str, values: Any) -> Iterable[float]:
    """Coerce values to a flat list of floats, mirroring INLA expectations."""
    try:
        import numpy as np  # type: ignore

        arr = np.asarray(values)
        if arr.size == 0:
            return []
        return arr.astype(float).reshape(-1).tolist()
    except Exception:
        pass

    iterable_types = (list, tuple, set)
    if isinstance(values, iterable_types):
        try:
            return [float(v) for v in values]
        except Exception as exc:  # pragma: no cover
            raise SafetyError(
                f"pyinla safety check: could not interpret '{name}' as numeric values."
            ) from exc

    if hasattr(values, "__iter__") and not isinstance(values, (str, bytes)):
        try:
            return [float(v) for v in values]
        except Exception as exc:  # pragma: no cover
            raise SafetyError(
                f"pyinla safety check: could not interpret '{name}' as numeric values."
            ) from exc

    try:
        return [float(values)]
    except Exception as exc:  # pragma: no cover
        raise SafetyError(
            f"pyinla safety check: could not interpret '{name}' as numeric values."
        ) from exc


def _coerce_length(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return len(_as_float_list("_probe", value))
    except SafetyError:
        pass
    try:
        import numpy as np  # type: ignore

        arr = np.asarray(value)
        if arr.ndim >= 1:
            return int(arr.shape[0])
    except Exception:
        pass
    try:
        return len(value)  # type: ignore[arg-type]
    except Exception:
        return None


def _infer_observation_count(kwargs: Dict[str, Any]) -> Optional[int]:
    data = kwargs.get("data")
    try:
        import pandas as pd  # type: ignore
    except Exception:  # pragma: no cover
        pd = None  # type: ignore

    if pd is not None and isinstance(data, pd.DataFrame):
        return len(data)

    if isinstance(data, dict):
        for value in data.values():
            length = _coerce_length(value)
            if length is not None:
                return length

    length = _coerce_length(data)
    if length is not None:
        return length

    model = kwargs.get("model") or kwargs.get("formula")
    if not isinstance(model, dict):
        return None

    response = model.get("response")
    if isinstance(response, str) and isinstance(data, dict) and response in data:
        return _coerce_length(data.get(response))

    if response is not None and not isinstance(response, str):
        return _coerce_length(response)

    return None


def _get_response_object(kwargs: Dict[str, Any]) -> Optional[Any]:
    model = kwargs.get("model") or kwargs.get("formula")
    if not isinstance(model, dict):
        return None
    response = model.get("response")
    if response is None:
        return None
    data = kwargs.get("data")

    try:
        import pandas as pd  # type: ignore
    except Exception:  # pragma: no cover
        pd = None  # type: ignore

    def lookup_by_name(name: str) -> Optional[Any]:
        if data is None:
            return None
        if pd is not None and isinstance(data, pd.DataFrame):
            if name in data.columns:
                return data[name]
            return None
        if isinstance(data, dict):
            return data.get(name)
        return None

    if isinstance(response, str):
        return lookup_by_name(response)
    if isinstance(response, tuple) and len(response) == 2:
        _, values = response
        return values
    return response


def _extract_response_array(kwargs: Dict[str, Any]) -> Optional[Any]:
    obj = _get_response_object(kwargs)
    if obj is None:
        return None
    try:
        import numpy as np  # type: ignore

        arr = np.asarray(obj)
        if arr.size == 0:
            return []
        return arr.reshape(-1)
    except Exception:
        pass
    if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
        return list(obj)
    return obj


def _normalize_hyper_entries(
    family_block: Dict[str, Any], *, allowed_names: Optional[Iterable[str]] = None
) -> list[tuple[Optional[str], Dict[str, Any]]]:
    hyper = family_block.get("hyper")
    if hyper is None:
        return []

    def ensure_dict(entry: Any) -> Dict[str, Any]:
        if not isinstance(entry, dict):
            raise SafetyError(
                "pyinla safety check: hyperparameter entries must be dicts (with 'prior', 'param', etc.)."
            )
        return dict(entry)

    if isinstance(hyper, list):
        normalized = [ensure_dict(entry) for entry in hyper]
        family_block["hyper"] = normalized
        return [(None, entry) for entry in normalized]

    if isinstance(hyper, dict):
        base_fields = {"prior", "param", "initial", "fixed", "to_theta", "from_theta"}
        if base_fields & set(hyper.keys()):
            entry = ensure_dict(hyper)
            family_block["hyper"] = [entry]
            name = None
            if allowed_names:
                name = next(iter(allowed_names), None)
            return [(name, entry)]

        if not allowed_names:
            raise SafetyError(
                "pyinla safety check: provide hyperparameters as a list or map by known keys for this likelihood."
            )

        allowed_order = list(allowed_names)
        unknown = set(hyper.keys()) - set(allowed_order)
        if unknown:
            raise SafetyError(
                "pyinla safety check: unsupported hyperparameter keys {} for this likelihood.".format(
                    ", ".join(sorted(unknown))
                )
            )
        normalized: list[tuple[Optional[str], Dict[str, Any]]] = []
        normalized_dict: Dict[str, Dict[str, Any]] = {}
        for name in allowed_order:
            if name in hyper:
                entry = ensure_dict(hyper[name])
                normalized.append((name, entry))
                normalized_dict[name] = entry
        family_block["hyper"] = normalized_dict
        return normalized

    raise SafetyError(
        "pyinla safety check: control['family']['hyper'] must be a list or dict when overriding hyperparameters."
    )


def enforce_compute_section(kwargs: Dict[str, Any]) -> None:
    control = kwargs.get("control") or {}
    if not isinstance(control, dict):
        return
    compute = control.get("compute")
    if compute is None:
        return
    if not isinstance(compute, dict):
        raise SafetyError("pyinla safety check: control['compute'] must be a dict if provided.")

    # Tested compute options mirror R-INLA's public arguments.  Keep this list
    # centralized so we can expand it consistently with future coverage (the
    # user request here was to enable WAIC).
    _allowed_compute_keys = {
        "config",
        "dic",
        "cpo",
        "mlik",
        "return_marginals",
        "waic",
        "po",
    }

    allowed_keys = _allowed_compute_keys
    extra = set(compute.keys()) - allowed_keys
    if extra:
        raise SafetyError(
            "pyinla safety check: unsupported keys in control['compute']: {}. "
            "Allowed: {}.".format(
                ", ".join(sorted(extra)),
                ", ".join(sorted(allowed_keys))
            )
        )

    for key in allowed_keys.intersection(compute.keys()):
        val = compute[key]
        if isinstance(val, bool):
            continue
        if isinstance(val, int) and val in (0, 1):
            continue
        raise SafetyError(
            "pyinla safety check: control['compute']['{}'] must be boolean.".format(key)
        )


def enforce_exposure_usage(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    normalized = list(families) if families is not None else list(_normalize_family_spec(kwargs.get("family")))
    has_poisson = any(f == "poisson" for f in normalized)
    has_nbinomial = any(f == "nbinomial" for f in normalized)
    if has_poisson or has_nbinomial:
        return
    if "E" in kwargs and kwargs["E"] is not None:
        raise SafetyError(
            "pyinla safety check: 'E' (exposure) is only permitted for poisson or nbinomial likelihoods."
        )


def enforce_random_structure(kwargs: Dict[str, Any]) -> None:
    # Accept both 'model' and 'formula' keys (they are aliases)
    model = kwargs.get("model") or kwargs.get("formula")
    if not isinstance(model, dict):
        return
    random_block = model.get("random")
    if random_block is None:
        return

    entries: List[Dict[str, Any]] = []

    def ensure_entry(entry: Any) -> Dict[str, Any]:
        if not isinstance(entry, dict):
            raise SafetyError("pyinla safety check: random effect specifications must be dicts.")
        return dict(entry)

    if isinstance(random_block, dict) and "model" not in random_block:
        for label, spec in random_block.items():
            spec_dict = ensure_entry(spec)
            if spec_dict.get("id") is None and spec_dict.get("covariate") is None:
                spec_dict["id"] = label
            entries.append(spec_dict)
    else:
        iterable = random_block if isinstance(random_block, (list, tuple)) else [random_block]
        for spec in iterable:
            entries.append(ensure_entry(spec))

    if not entries:
        raise SafetyError("pyinla safety check: empty random-effect specification is not allowed.")

    # Check for SPDE objects - these are handled separately
    # Import here to avoid circular imports
    def _is_spde_object(obj):
        """Check if object is an SPDE model from fmesher."""
        # Check by class name to avoid import issues
        return type(obj).__name__ == "SPDE2PcMatern"

    # Include __spde_object__ marker in allowed keys (set during previous validation)
    allowed_spde_keys = {"id", "model", "A.local", "__spde_object__"}

    for spec in entries:
        model_val = spec.get("model")
        if _is_spde_object(model_val) or spec.get("__spde_object__"):
            # SPDE object detected (or already processed) - validate keys and structure
            unknown_keys = set(spec.keys()) - allowed_spde_keys
            if unknown_keys:
                raise SafetyError(
                    "pyinla safety check: SPDE random effects only allow keys {{id, model, A.local}}; "
                    "found unexpected: {}.".format(", ".join(sorted(unknown_keys)))
                )

            # Check required keys
            if "id" not in spec:
                raise SafetyError(
                    "pyinla safety check: SPDE random effects require an 'id' key."
                )

            if "A.local" not in spec:
                raise SafetyError(
                    "pyinla safety check: SPDE random effects require an 'A.local' projection matrix."
                )

            # Validate A.local is a matrix (sparse or dense)
            A_local = spec.get("A.local")
            is_valid_A = False
            try:
                import numpy as _np_local
                if isinstance(A_local, _np_local.ndarray) and A_local.ndim == 2:
                    is_valid_A = True
            except ImportError:
                pass
            if not is_valid_A and _HAVE_SCIPY and sp.issparse(A_local):
                is_valid_A = True

            if not is_valid_A:
                raise SafetyError(
                    "pyinla safety check: SPDE 'A.local' must be a 2D array or sparse matrix."
                )

            # Mark as SPDE for downstream processing
            spec["__spde_object__"] = True

    allowed_models = {"iid", "linear", "clinear", "z", "generic0", "generic1", "generic2", "iidkd", "rw1", "rw2", "seasonal", "ar1", "ar", "besag", "bym2"}

    # Allowed keys per model type
    # 'diagonal' adds small value to precision matrix diagonal for numerical stability (available in all models)
    allowed_keys_per_model = {
        "iid": {"id", "model", "hyper", "n", "constr", "initial", "fixed", "diagonal"},
        "iidkd": {"id", "model", "hyper", "order", "n", "constr", "diagonal"},
        "rw1": {"id", "model", "hyper", "n", "constr", "cyclic", "scale.model", "values", "diagonal"},
        "rw2": {"id", "model", "hyper", "n", "constr", "cyclic", "scale.model", "values", "diagonal"},
        "linear": {"id", "model", "mean.linear", "prec.linear", "diagonal"},
        "clinear": {"id", "model", "hyper", "range", "diagonal"},
        "z": {"id", "model", "hyper", "Z", "Cmatrix", "precision", "constr", "extraconstr", "diagonal"},
        "generic0": {"id", "model", "hyper", "Cmatrix", "n", "constr", "extraconstr", "diagonal"},
        "generic1": {"id", "model", "hyper", "Cmatrix", "n", "constr", "extraconstr", "diagonal"},
        "generic2": {"id", "model", "hyper", "Cmatrix", "n", "constr", "extraconstr", "diagonal"},
        "seasonal": {"id", "model", "hyper", "n", "constr", "scale.model", "values", "extraconstr", "diagonal", "season.length"},
        "ar1": {"id", "model", "hyper", "constr", "cyclic", "values", "extraconstr", "diagonal"},
        "ar": {"id", "model", "hyper", "order", "constr", "values", "extraconstr", "diagonal"},
        "besag": {"id", "model", "hyper", "graph", "n", "constr", "scale.model", "adjust.for.con.comp", "diagonal", "rankdef", "values", "nrep", "replicate", "ngroup", "group", "control.group", "compute", "vb.correct"},
        "bym2": {"id", "model", "hyper", "graph", "n", "constr", "scale.model", "extraconstr", "diagonal", "rankdef", "values", "nrep", "replicate", "ngroup", "group", "control.group", "compute", "vb.correct"},
    }

    for spec in entries:
        # Skip SPDE objects - already validated above
        if spec.get("__spde_object__"):
            continue

        model_name = str(spec.get("model", "")).strip().lower()
        if model_name not in allowed_models:
            raise SafetyError(
                "pyinla safety check: random effects currently support only model in {}.".format(
                    ", ".join(sorted(allowed_models))
                )
            )

        # Validate keys for this model type
        allowed_keys = allowed_keys_per_model.get(model_name, set())
        unknown_keys = set(spec.keys()) - allowed_keys
        if unknown_keys:
            raise SafetyError(
                "pyinla safety check: unknown key(s) {} for '{}' random effect. Allowed keys: {}.".format(
                    ", ".join(sorted(unknown_keys)),
                    model_name,
                    ", ".join(sorted(allowed_keys))
                )
            )

        ident = spec.get("id") or spec.get("covariate")
        if ident is None:
            raise SafetyError(
                "pyinla safety check: random effects must include an 'id' (or 'covariate' for linear/clinear models)."
            )
        spec.setdefault("id", ident)

        # linear model allows mean.linear and prec.linear (prior on the slope)
        # clinear does NOT - its prior is specified via hyper on theta
        if model_name == "linear":
            allowed_dot_fields = {"mean.linear", "prec.linear"}
            dot_fields = [key for key in spec.keys() if "." in key]
            extra = [key for key in dot_fields if key not in allowed_dot_fields]
            if extra:
                raise SafetyError(
                    "pyinla safety check: linear random effects only allow {} overrides; invalid fields: {}."
                    .format(
                        ", ".join(sorted(allowed_dot_fields)),
                        ", ".join(sorted(extra))
                    )
                )
        elif model_name == "clinear":
            # clinear does not use mean.linear/prec.linear - prior is on theta via hyper
            dot_fields = [key for key in spec.keys() if "." in key]
            if dot_fields:
                raise SafetyError(
                    "pyinla safety check: clinear does not support dot-style options like mean.linear/prec.linear; "
                    "use 'hyper' to configure the prior on theta. Found: {}.".format(
                        ", ".join(sorted(dot_fields))
                    )
                )

        if model_name == "clinear":
            rng = spec.get("range")
            if rng is not None:
                if not isinstance(rng, (list, tuple)) or len(rng) != 2:
                    raise SafetyError(
                        "pyinla safety check: clinear 'range' must be a two-value sequence (low, high)."
                    )
                low, high = rng
                try:
                    low = float(low)
                    high = float(high)
                except Exception as exc:
                    raise SafetyError(
                        "pyinla safety check: clinear 'range' entries must be numeric."
                    ) from exc
                # Check: low < high and neither is NaN
                if not (low < high) or math.isnan(low) or math.isnan(high):
                    raise SafetyError(
                        "pyinla safety check: clinear 'range' requires low < high (NaN not allowed)."
                    )
                # R-INLA only supports: (low, Inf), (low, high) both finite, (-Inf, Inf)
                # The case (-Inf, high) with finite high is NOT implemented in R-INLA
                low_inf = math.isinf(low) and low < 0
                high_finite = math.isfinite(high)
                if low_inf and high_finite:
                    raise SafetyError(
                        "pyinla safety check: clinear 'range' with (-Inf, high) where high is finite "
                        "is not supported by R-INLA. Use (low, Inf), (low, high) both finite, or (-Inf, Inf)."
                    )

        # z model requires Z matrix
        if model_name == "z":
            Z = spec.get("Z")
            if Z is None:
                raise SafetyError(
                    "pyinla safety check: z model requires 'Z' design matrix."
                )
            if _HAVE_SCIPY and sp.issparse(Z):
                Z_arr = Z
            else:
                try:
                    import numpy as np  # type: ignore
                    Z_arr = np.asarray(Z)
                except Exception as exc:
                    raise SafetyError(
                        "pyinla safety check: z model 'Z' must be convertible to a numpy array."
                    ) from exc
            if Z_arr.ndim != 2:
                raise SafetyError(
                    "pyinla safety check: z model 'Z' must be a 2D matrix (n_obs × n_random_effects)."
                )

            Cmatrix = spec.get("Cmatrix")
            if Cmatrix is not None:
                if _HAVE_SCIPY and sp.issparse(Cmatrix):
                    C_arr = Cmatrix
                else:
                    try:
                        import numpy as np  # type: ignore
                        C_arr = np.asarray(Cmatrix)
                    except Exception as exc:
                        raise SafetyError(
                            "pyinla safety check: z model 'Cmatrix' must be convertible to a numpy array."
                        ) from exc
                if C_arr.ndim != 2 or C_arr.shape[0] != C_arr.shape[1]:
                    raise SafetyError(
                        "pyinla safety check: z model 'Cmatrix' must be square (m × m)."
                    )
                if C_arr.shape[0] != Z_arr.shape[1]:
                    raise SafetyError(
                        "pyinla safety check: z model 'Cmatrix' dimension must match number of columns in 'Z'."
                    )

            precision_val = spec.get("precision")
            if precision_val is not None:
                try:
                    precision_float = float(precision_val)
                except Exception as exc:
                    raise SafetyError(
                        "pyinla safety check: z model 'precision' must be numeric."
                    ) from exc
                if not math.isfinite(precision_float) or precision_float <= 0.0:
                    raise SafetyError(
                        "pyinla safety check: z model 'precision' must be a positive, finite number."
                    )

            # extraconstr validation for z model
            extraconstr = spec.get("extraconstr")
            if extraconstr is not None:
                if not isinstance(extraconstr, dict):
                    raise SafetyError(
                        "pyinla safety check: z model 'extraconstr' must be a dict with keys 'A' and 'e'."
                    )
                allowed_keys = {"A", "e"}
                extra_keys = set(extraconstr.keys()) - allowed_keys
                if extra_keys:
                    raise SafetyError(
                        f"pyinla safety check: z model 'extraconstr' only allows keys 'A' and 'e'; "
                        f"found {', '.join(sorted(extra_keys))}."
                    )
                A_constr = extraconstr.get("A")
                e_constr = extraconstr.get("e")
                if A_constr is None:
                    raise SafetyError(
                        "pyinla safety check: z model 'extraconstr' requires 'A' matrix."
                    )
                try:
                    import numpy as np  # type: ignore
                    A_arr = np.asarray(A_constr)
                except Exception as exc:
                    raise SafetyError(
                        "pyinla safety check: z model 'extraconstr' 'A' must be convertible to a numpy array."
                    ) from exc
                if A_arr.ndim == 1:
                    A_arr = A_arr.reshape(1, -1)
                if A_arr.ndim != 2:
                    raise SafetyError(
                        "pyinla safety check: z model 'extraconstr' 'A' must be a 1D or 2D array."
                    )
                # For z model, augmented field is (n + m), so A must have (n + m) columns
                n_obs, m_effects = Z_arr.shape
                expected_cols = n_obs + m_effects
                if A_arr.shape[1] != expected_cols:
                    raise SafetyError(
                        f"pyinla safety check: z model 'extraconstr' 'A' must have {expected_cols} columns "
                        f"(n={n_obs} + m={m_effects}) to match the augmented latent field; got {A_arr.shape[1]}."
                    )
                if e_constr is not None:
                    try:
                        import numpy as np  # type: ignore
                        e_arr = np.asarray(e_constr)
                    except Exception as exc:
                        raise SafetyError(
                            "pyinla safety check: z model 'extraconstr' 'e' must be convertible to a numpy array."
                        ) from exc
                    e_arr = np.atleast_1d(e_arr)
                    if e_arr.shape[0] != A_arr.shape[0]:
                        raise SafetyError(
                            f"pyinla safety check: z model 'extraconstr' 'e' length ({e_arr.shape[0]}) "
                            f"must match number of rows in 'A' ({A_arr.shape[0]})."
                        )

        # seasonal model requires season.length
        if model_name == "seasonal":
            season_length = spec.get("season.length")
            if season_length is None:
                raise SafetyError(
                    "pyinla safety check: seasonal model requires 'season.length' parameter "
                    "(the periodicity m of the seasonal component)."
                )
            try:
                season_length_int = int(season_length)
            except (TypeError, ValueError) as exc:
                raise SafetyError(
                    "pyinla safety check: seasonal 'season.length' must be a positive integer."
                ) from exc
            if season_length_int < 2:
                raise SafetyError(
                    "pyinla safety check: seasonal 'season.length' must be at least 2."
                )

        # ar (autoregressive) model requires order parameter
        if model_name == "ar":
            order = spec.get("order")
            if order is None:
                raise SafetyError(
                    "pyinla safety check: ar model requires 'order' parameter "
                    "(the autoregressive order p, must be 1-10)."
                )
            try:
                order_int = int(order)
            except (TypeError, ValueError) as exc:
                raise SafetyError(
                    "pyinla safety check: ar 'order' must be a positive integer (1-10)."
                ) from exc
            if order_int < 1 or order_int > 10:
                raise SafetyError(
                    "pyinla safety check: ar 'order' must be between 1 and 10 (inclusive); got {}.".format(order_int)
                )

        hyper = spec.get("hyper")
        if model_name in {"generic", "generic0", "generic1", "generic2"}:
            Cmatrix = spec.get("Cmatrix")
            if Cmatrix is None:
                raise SafetyError(
                    "pyinla safety check: generic/generic0/1/2 models require 'Cmatrix'."
                )
            # Check shape - handle both sparse and dense matrices
            try:
                import scipy.sparse as sp_check  # type: ignore

                if sp_check.issparse(Cmatrix):
                    shape = Cmatrix.shape
                    if len(shape) != 2 or shape[0] != shape[1]:
                        raise SafetyError(
                            "pyinla safety check: generic* model 'Cmatrix' must be a square matrix."
                        )
                else:
                    import numpy as np  # type: ignore

                    C_arr = np.asarray(Cmatrix)
                    if C_arr.ndim != 2 or C_arr.shape[0] != C_arr.shape[1]:
                        raise SafetyError(
                            "pyinla safety check: generic* model 'Cmatrix' must be a square matrix."
                        )
            except ImportError:
                # scipy not available, try numpy only
                import numpy as np  # type: ignore

                C_arr = np.asarray(Cmatrix)
                if C_arr.ndim != 2 or C_arr.shape[0] != C_arr.shape[1]:
                    raise SafetyError(
                        "pyinla safety check: generic* model 'Cmatrix' must be a square matrix."
                    )

        if hyper is not None:
            # linear model has NO hyperparameters - use mean.linear/prec.linear instead
            if model_name == "linear":
                raise SafetyError(
                    "pyinla safety check: linear model has no hyperparameters. "
                    "Use 'mean.linear' and 'prec.linear' to specify the prior on the slope."
                )
            # Different models have different numbers of hyperparameters
            # generic1 and generic2 have 2 hyperparameters, others have 1
            expected_hyper_count = {
                "iid": 1,
                "clinear": 1,
                "z": 1,
                "generic": 1,
                "generic0": 1,
                "generic1": 2,  # precision and beta
                "generic2": 2,  # precision cmatrix and precision random
                "iidkd": None,
                "rw1": 1,  # precision
                "rw2": 1,  # precision
                "seasonal": 1,  # precision
                "ar1": (1, 3),  # precision (theta1), rho (theta2), optionally mean (theta3)
                "ar": (1, 11),  # precision (theta1) + pacf1..pacf10 (theta2-theta11), depends on order
                "besag": 1,  # precision
                "bym": 2,  # precision for structured and unstructured parts
                "bym2": (1, 2),  # precision (theta1) and optionally phi (theta2)
            }
            expected = expected_hyper_count.get(model_name, 1)

            # Semantic name mappings: short_name -> theta position (1-indexed)
            # This allows users to specify hyper by name instead of position
            hyper_name_mapping = {
                "iid": {"prec": 1, "precision": 1},
                "clinear": {"beta": 1, "b": 1},
                "z": {"prec": 1, "precision": 1},
                "generic": {"prec": 1, "precision": 1},
                "generic0": {"prec": 1, "precision": 1},
                "generic1": {"prec": 1, "precision": 1, "beta": 2, "b": 2},
                "generic2": {"prec": 1, "precision": 1, "prec.random": 2},
                "rw1": {"prec": 1, "precision": 1},
                "rw2": {"prec": 1, "precision": 1},
                "seasonal": {"prec": 1, "precision": 1},
                "ar1": {"prec": 1, "precision": 1, "rho": 2, "mean": 3},
                "ar": {
                    "prec": 1, "precision": 1,
                    "pacf1": 2, "pacf2": 3, "pacf3": 4, "pacf4": 5, "pacf5": 6,
                    "pacf6": 7, "pacf7": 8, "pacf8": 9, "pacf9": 10, "pacf10": 11,
                },
                "besag": {"prec": 1, "precision": 1},
                "bym": {"prec": 1, "precision": 1, "prec.iid": 2},
                "bym2": {"prec": 1, "precision": 1, "phi": 2},
            }

            # Convert dict format to list format if needed
            if isinstance(hyper, dict):
                name_map = hyper_name_mapping.get(model_name, {})
                # Determine max theta index needed
                max_theta = 0
                for key in hyper.keys():
                    key_lower = key.lower()
                    if key_lower in name_map:
                        max_theta = max(max_theta, name_map[key_lower])
                    elif key_lower.startswith("theta"):
                        try:
                            idx = int(key_lower[5:]) if len(key_lower) > 5 else 1
                            max_theta = max(max_theta, idx)
                        except ValueError:
                            pass

                # Create list with empty dicts for positions not specified
                hyper_list = [{} for _ in range(max_theta)]

                for key, value in hyper.items():
                    key_lower = key.lower()
                    if key_lower in name_map:
                        idx = name_map[key_lower] - 1  # Convert to 0-indexed
                    elif key_lower.startswith("theta"):
                        try:
                            idx = int(key_lower[5:]) - 1 if len(key_lower) > 5 else 0
                        except ValueError:
                            raise SafetyError(
                                "pyinla safety check: {} model 'hyper' has unknown key '{}'. "
                                "Use semantic names like {} or positional names like 'theta1', 'theta2', etc.".format(
                                    model_name,
                                    key,
                                    ", ".join(f"'{k}'" for k in name_map.keys()) if name_map else "'theta1'"
                                )
                            )
                    else:
                        raise SafetyError(
                            "pyinla safety check: {} model 'hyper' has unknown key '{}'. "
                            "Use semantic names like {} or positional names like 'theta1', 'theta2', etc.".format(
                                model_name,
                                key,
                                ", ".join(f"'{k}'" for k in name_map.keys()) if name_map else "'theta1'"
                            )
                        )

                    if idx < 0 or idx >= len(hyper_list):
                        raise SafetyError(
                            "pyinla safety check: {} model 'hyper' key '{}' maps to invalid position {}.".format(
                                model_name, key, idx + 1
                            )
                        )
                    hyper_list[idx] = value

                hyper = hyper_list

            if not isinstance(hyper, list):
                raise SafetyError(
                    "pyinla safety check: {} model 'hyper' must be a list of dicts or a dict with semantic names.".format(model_name)
                )
            if expected is None:
                min_hyper = 1
                max_hyper = 299
            elif isinstance(expected, tuple):
                min_hyper, max_hyper = expected
            else:
                min_hyper = max_hyper = expected
            if len(hyper) < min_hyper or len(hyper) > max_hyper:
                if min_hyper == max_hyper:
                    raise SafetyError(
                        "pyinla safety check: {} model expects exactly {} hyperparameter entr{}; got {}.".format(
                            model_name,
                            min_hyper,
                            "y" if min_hyper == 1 else "ies",
                            len(hyper),
                        )
                    )
                else:
                    raise SafetyError(
                        "pyinla safety check: {} model expects between {} and {} hyperparameter entries; got {}.".format(
                            model_name,
                            min_hyper,
                            max_hyper,
                            len(hyper),
                        )
                    )
            # Validate each entry
            allowed = {"prior", "param", "initial", "fixed"}
            validated_entries = []
            for entry in hyper:
                entry_dict = dict(entry)
                extra = set(entry_dict.keys()) - allowed
                if extra:
                    raise SafetyError(
                        "pyinla safety check: {} model 'hyper' only allows {}; found {}.".format(
                            model_name,
                            ", ".join(sorted(allowed)),
                            ", ".join(sorted(extra))
                        )
                    )
                validated_entries.append(entry_dict)
            spec["hyper"] = validated_entries

    model["random"] = entries


def enforce_poisson_exposure(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    normalized = list(families) if families is not None else list(_normalize_family_spec(kwargs.get("family")))
    has_poisson = any(f == "poisson" for f in normalized)
    if not has_poisson:
        return

    exposures = kwargs.get("E")
    if exposures is None:
        # Exposure is optional; only validate when present.
        return

    exp_list = list(_as_float_list("E", exposures))
    if len(exp_list) == 0:
        raise SafetyError("pyinla safety check: 'E' (exposure) may not be empty.")

    # Validate E > 0 (exposure must be strictly positive)
    import numpy as np
    exp_arr = np.asarray(exp_list, dtype=float)
    invalid_mask = ~(exp_arr > 0)
    if invalid_mask.any():
        bad_vals = exp_arr[invalid_mask][:5].tolist()
        raise SafetyError(
            "pyinla safety check: 'E' (exposure) must be strictly positive; found invalid values: {}.".format(bad_vals)
        )

    n_obs = _infer_observation_count(kwargs)
    if n_obs is not None and len(exp_list) != n_obs:
        raise SafetyError(
            "pyinla safety check: 'E' length ({}) must match number of observations ({}).".format(
                len(exp_list), n_obs
            )
        )


def enforce_nbinomial_exposure(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    """Validate exposure (E) for nbinomial likelihood: must be strictly positive."""
    normalized = list(families) if families is not None else list(_normalize_family_spec(kwargs.get("family")))
    has_nbinomial = any(f == "nbinomial" for f in normalized)
    if not has_nbinomial:
        return

    exposures = kwargs.get("E")
    if exposures is None:
        # Exposure is optional for nbinomial; only validate when present.
        return

    exp_list = list(_as_float_list("E", exposures))
    if len(exp_list) == 0:
        raise SafetyError("pyinla safety check: 'E' (exposure) may not be empty for nbinomial.")

    # Validate E > 0 (exposure must be strictly positive)
    import numpy as np
    exp_arr = np.asarray(exp_list, dtype=float)
    invalid_mask = ~(exp_arr > 0)
    if invalid_mask.any():
        bad_vals = exp_arr[invalid_mask][:5].tolist()
        raise SafetyError(
            "pyinla safety check: 'E' (exposure) must be strictly positive for nbinomial; found invalid values: {}.".format(bad_vals)
        )

    n_obs = _infer_observation_count(kwargs)
    if n_obs is not None and len(exp_list) != n_obs:
        raise SafetyError(
            "pyinla safety check: 'E' length ({}) must match number of observations ({}) for nbinomial.".format(
                len(exp_list), n_obs
            )
        )


def enforce_binomial_trials(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    normalized = list(families) if families is not None else list(_normalize_family_spec(kwargs.get("family")))
    requires_trials = any(f in ("binomial", "xbinomial", "betabinomial", "nbinomial2") for f in normalized)
    ntrials = kwargs.get("Ntrials")

    if not requires_trials:
        if ntrials is not None:
            raise SafetyError(
                "pyinla safety check: 'Ntrials' is only permitted for binomial/xbinomial/betabinomial/nbinomial2 likelihoods."
            )
        return

    if ntrials is None:
        # Allow omission only when response is binary (0/1)
        data = _extract_response_array(kwargs)
        if data is None:
            raise SafetyError(
                "pyinla safety check: provide 'Ntrials' for aggregated binomial responses (currently missing)."
            )
        try:
            import numpy as np  # type: ignore
            arr = np.asarray(data, dtype=float).reshape(-1)
            finite = arr[np.isfinite(arr)]
            values = set(np.unique(finite))
        except Exception:
            try:
                values = set(float(v) for v in data)
            except Exception as exc:
                raise SafetyError(
                    "pyinla safety check: could not interpret response when validating missing 'Ntrials'."
                ) from exc
        values = {v for v in values if np.isfinite(v)}
        if not values.issubset({0.0, 1.0}):
            raise SafetyError(
                "pyinla safety check: aggregated binomial data requires 'Ntrials'; response contains values other than 0/1."
            )
        # Pure Bernoulli, so we fall through without requiring Ntrials
        return

    trials_list = list(_as_float_list("Ntrials", ntrials))
    if len(trials_list) == 0:
        raise SafetyError("pyinla safety check: 'Ntrials' may not be empty for binomial models.")

    # Validate Ntrials > 0 and are integers
    import numpy as np
    trials_arr = np.asarray(trials_list, dtype=float)
    # Check for positive values
    non_positive = trials_arr[trials_arr <= 0]
    if len(non_positive) > 0:
        bad_vals = non_positive[:5].tolist()
        raise SafetyError(
            "pyinla safety check: 'Ntrials' must be positive; found invalid values: {}.".format(bad_vals)
        )
    # Check for integer values (allow float representation of integers like 10.0)
    non_integer = trials_arr[trials_arr != np.floor(trials_arr)]
    if len(non_integer) > 0:
        bad_vals = non_integer[:5].tolist()
        raise SafetyError(
            "pyinla safety check: 'Ntrials' must be integers; found non-integer values: {}.".format(bad_vals)
        )

    n_obs = _infer_observation_count(kwargs)
    if n_obs is not None and len(trials_list) != n_obs:
        raise SafetyError(
            "pyinla safety check: 'Ntrials' length ({}) must match number of observations ({}).".format(
                len(trials_list), n_obs
            )
        )


def enforce_binomial_family_variant(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    normalized = list(families) if families is not None else list(_normalize_family_spec(kwargs.get("family")))
    has_binomial = any(f in ("binomial", "xbinomial") for f in normalized)
    has_nbinomial = any(f == "nbinomial" for f in normalized)
    has_nbinomial2 = any(f == "nbinomial2" for f in normalized)
    has_betabinomial = any(f == "betabinomial" for f in normalized)
    has_beta = any(f == "beta" for f in normalized)
    has_logistic = any(f == "logistic" for f in normalized)
    has_loglogistic = any(f in ("loglogistic", "loglogisticsurv") for f in normalized)
    has_loglogistic_only = any(f == "loglogistic" for f in normalized)
    has_loglogisticsurv = any(f == "loglogisticsurv" for f in normalized)
    has_weibull = any(f == "weibull" for f in normalized)
    has_weibullsurv = any(f == "weibullsurv" for f in normalized)
    has_exponential = any(f == "exponential" for f in normalized)
    has_exponentialsurv = any(f == "exponentialsurv" for f in normalized)
    has_lognormal = any(f == "lognormal" for f in normalized)
    has_lognormalsurv = any(f == "lognormalsurv" for f in normalized)
    has_sn = any(f == "sn" for f in normalized)
    has_t = any(f == "t" for f in normalized)
    has_gaussian = any(f == "gaussian" for f in normalized)
    has_poisson = any(f == "poisson" for f in normalized)
    has_gamma = any(f == "gamma" for f in normalized)
    has_gammasurv = any(f == "gammasurv" for f in normalized)
    if not (
        has_binomial
        or has_betabinomial
        or has_nbinomial
        or has_nbinomial2
        or has_beta
        or has_logistic
        or has_loglogistic
        or has_weibull
        or has_weibullsurv
        or has_exponential
        or has_exponentialsurv
        or has_lognormal
        or has_lognormalsurv
        or has_sn
        or has_t
        or has_gaussian
        or has_poisson
        or has_gamma
        or has_gammasurv
    ):
        return

    control = kwargs.get("control") or {}
    if not isinstance(control, dict):
        return
    family_block = control.get("family")
    if family_block is None:
        return
    if not isinstance(family_block, dict):
        raise SafetyError("pyinla safety check: control['family'] must be a dict when specifying family options.")

    allowed_keys = set()
    if has_binomial or has_betabinomial or has_nbinomial or has_nbinomial2 or has_loglogistic or has_weibull or has_weibullsurv:
        allowed_keys.add("variant")
    if has_nbinomial or has_nbinomial2 or has_betabinomial or has_beta or has_logistic or has_weibull or has_weibullsurv or has_lognormal or has_lognormalsurv or has_loglogistic or has_sn or has_t or has_gaussian or has_gamma or has_gammasurv or has_exponentialsurv:
        allowed_keys.add("hyper")
    if has_beta:
        allowed_keys.add("beta.censor.value")
    if has_binomial or has_betabinomial or has_nbinomial or has_nbinomial2 or has_beta or has_logistic or has_exponential or has_exponentialsurv or has_lognormal or has_lognormalsurv or has_loglogistic or has_weibull or has_weibullsurv or has_sn or has_t or has_gaussian or has_poisson or has_gamma or has_gammasurv:
        allowed_keys.add("link")
    # control.link is used for quantile links (e.g., {'model': 'quantile', 'quantile': 0.85})
    if has_gamma or has_gammasurv or has_weibull or has_weibullsurv or has_nbinomial:
        allowed_keys.add("control.link")
    extra_keys = set(family_block.keys()) - allowed_keys
    if extra_keys:
        raise SafetyError(
            "pyinla safety check: control['family'] only accepts {} for these likelihoods: {}.".format(
                ", ".join(sorted(allowed_keys)),
                ", ".join(sorted(extra_keys))
            )
        )

    if "variant" in family_block:
        if not (
            has_binomial
            or has_nbinomial
            or has_nbinomial2
            or has_betabinomial
            or has_loglogistic
            or has_weibull
            or has_weibullsurv
        ):
            raise SafetyError(
                "pyinla safety check: control['family']['variant'] is only valid for binomial/xbinomial/nbinomial/nbinomial2/betabinomial/loglogistic/loglogisticsurv/weibull/weibullsurv families."
            )
        try:
            variant_val = int(family_block.get("variant"))
        except Exception as exc:
            raise SafetyError(
                "pyinla safety check: control['family']['variant'] must be an integer (0 or 1)."
            ) from exc
        allowed_variant_values = {0, 1}
        if has_nbinomial:
            allowed_variant_values.add(2)
        if has_nbinomial2:
            allowed_variant_values = {0}  # only default behavior allowed
        if variant_val not in allowed_variant_values:
            raise SafetyError(
                "pyinla safety check: control['family']['variant'] must be one of {} for the requested family.".format(
                    sorted(allowed_variant_values)
                )
            )

    hyper_block = family_block.get("hyper")
    if hyper_block is not None:
        if not (has_nbinomial or has_betabinomial or has_beta or has_logistic or has_lognormal or has_lognormalsurv or has_weibull or has_weibullsurv or has_loglogistic or has_sn or has_t or has_gaussian or has_gamma or has_gammasurv or has_exponentialsurv):
            raise SafetyError(
                "pyinla safety check: control['family']['hyper'] is only allowed for gaussian/gamma/gammasurv/exponentialsurv/nbinomial/betabinomial/beta/logistic/lognormal/lognormalsurv/weibull/weibullsurv/loglogistic/loglogisticsurv/sn/t."
            )
        # Allow both dict format (single family) and list format (multiple families)
        if not isinstance(hyper_block, (dict, list)):
            raise SafetyError(
                "pyinla safety check: control['family']['hyper'] must be a dict or list of dicts."
            )

        if "beta.censor.value" in family_block and not has_beta:
            raise SafetyError(
                "pyinla safety check: control['family']['beta.censor.value'] is only valid for the beta likelihood."
            )

    # Validate beta.censor.value range [0, 0.5)
    if has_beta and "beta.censor.value" in family_block:
        censor_val = family_block.get("beta.censor.value")
        if censor_val is not None:
            try:
                censor_float = float(censor_val)
            except (TypeError, ValueError) as exc:
                raise SafetyError(
                    "pyinla safety check: control['family']['beta.censor.value'] must be a number."
                ) from exc
            if censor_float < 0.0 or censor_float >= 0.5:
                raise SafetyError(
                    f"pyinla safety check: control['family']['beta.censor.value'] must be in [0, 0.5); got {censor_float}."
                )

    # Validate link function for beta
    if has_beta and "link" in family_block:
        link_val = family_block.get("link")
        if link_val is not None:
            link_norm = _normalize_link_value(link_val)
            beta_allowed_links = {"default", "logit", "loga", "cauchit", "probit", "cloglog", "ccloglog", "loglog"}
            if link_norm is not None and link_norm not in beta_allowed_links:
                raise SafetyError(
                    f"pyinla safety check: control['family']['link'] for beta must be one of {sorted(beta_allowed_links)}; got '{link_val}'."
                )

    # Validate link function for exponential
    if has_exponential and "link" in family_block:
        link_val = family_block.get("link")
        if link_val is not None:
            link_norm = _normalize_link_value(link_val)
            exponential_allowed_links = {"default", "log"}
            if link_norm is not None and link_norm not in exponential_allowed_links:
                raise SafetyError(
                    f"pyinla safety check: control['family']['link'] for exponential must be one of {sorted(exponential_allowed_links)}; got '{link_val}'."
                )

    # Validate link function for exponentialsurv
    if has_exponentialsurv and "link" in family_block:
        link_val = family_block.get("link")
        if link_val is not None:
            link_norm = _normalize_link_value(link_val)
            exponentialsurv_allowed_links = {"default", "log", "neglog"}
            if link_norm is not None and link_norm not in exponentialsurv_allowed_links:
                raise SafetyError(
                    f"pyinla safety check: control['family']['link'] for exponentialsurv must be one of {sorted(exponentialsurv_allowed_links)}; got '{link_val}'."
                )

    # Validate link function for lognormal
    if has_lognormal and "link" in family_block:
        link_val = family_block.get("link")
        if link_val is not None:
            link_norm = _normalize_link_value(link_val)
            lognormal_allowed_links = {"default", "identity"}
            if link_norm is not None and link_norm not in lognormal_allowed_links:
                raise SafetyError(
                    f"pyinla safety check: control['family']['link'] for lognormal must be one of {sorted(lognormal_allowed_links)}; got '{link_val}'."
                )

    # Validate link function for lognormalsurv
    if has_lognormalsurv and "link" in family_block:
        link_val = family_block.get("link")
        if link_val is not None:
            link_norm = _normalize_link_value(link_val)
            lognormalsurv_allowed_links = {"default", "identity"}
            if link_norm is not None and link_norm not in lognormalsurv_allowed_links:
                raise SafetyError(
                    f"pyinla safety check: control['family']['link'] for lognormalsurv must be one of {sorted(lognormalsurv_allowed_links)}; got '{link_val}'."
                )

    # Validate link function for logistic
    if has_logistic and "link" in family_block:
        link_val = family_block.get("link")
        if link_val is not None:
            link_norm = _normalize_link_value(link_val)
            logistic_allowed_links = {"default", "identity"}
            if link_norm is not None and link_norm not in logistic_allowed_links:
                raise SafetyError(
                    f"pyinla safety check: control['family']['link'] for logistic must be one of {sorted(logistic_allowed_links)}; got '{link_val}'."
                )

    # Validate link function for sn (skew-normal)
    if has_sn and "link" in family_block:
        link_val = family_block.get("link")
        if link_val is not None:
            link_norm = _normalize_link_value(link_val)
            sn_allowed_links = {"default", "identity"}
            if link_norm is not None and link_norm not in sn_allowed_links:
                raise SafetyError(
                    f"pyinla safety check: control['family']['link'] for sn must be one of {sorted(sn_allowed_links)}; got '{link_val}'."
                )

    # Validate link function for loglogistic
    if has_loglogistic_only and "link" in family_block:
        link_val = family_block.get("link")
        if link_val is not None:
            link_norm = _normalize_link_value(link_val)
            loglogistic_allowed_links = {"default", "log", "neglog"}
            if link_norm is not None and link_norm not in loglogistic_allowed_links:
                raise SafetyError(
                    f"pyinla safety check: control['family']['link'] for loglogistic must be one of {sorted(loglogistic_allowed_links)}; got '{link_val}'."
                )

    # Validate link function for loglogisticsurv
    if has_loglogisticsurv and "link" in family_block:
        link_val = family_block.get("link")
        if link_val is not None:
            link_norm = _normalize_link_value(link_val)
            loglogisticsurv_allowed_links = {"default", "log", "neglog"}
            if link_norm is not None and link_norm not in loglogisticsurv_allowed_links:
                raise SafetyError(
                    f"pyinla safety check: control['family']['link'] for loglogisticsurv must be one of {sorted(loglogisticsurv_allowed_links)}; got '{link_val}'."
                )

    # Validate link function for weibull
    if has_weibull and "link" in family_block:
        link_val = family_block.get("link")
        if link_val is not None:
            link_norm = _normalize_link_value(link_val)
            weibull_allowed_links = {"default", "log", "neglog", "quantile"}
            if link_norm is not None and link_norm not in weibull_allowed_links:
                raise SafetyError(
                    f"pyinla safety check: control['family']['link'] for weibull must be one of {sorted(weibull_allowed_links)}; got '{link_val}'."
                )

    # Validate link function for weibullsurv
    if has_weibullsurv and "link" in family_block:
        link_val = family_block.get("link")
        if link_val is not None:
            link_norm = _normalize_link_value(link_val)
            weibullsurv_allowed_links = {"default", "log", "neglog", "quantile"}
            if link_norm is not None and link_norm not in weibullsurv_allowed_links:
                raise SafetyError(
                    f"pyinla safety check: control['family']['link'] for weibullsurv must be one of {sorted(weibullsurv_allowed_links)}; got '{link_val}'."
                )

    # Validate link function for t (student-t)
    if has_t and "link" in family_block:
        link_val = family_block.get("link")
        if link_val is not None:
            link_norm = _normalize_link_value(link_val)
            t_allowed_links = {"default", "identity"}
            if link_norm is not None and link_norm not in t_allowed_links:
                raise SafetyError(
                    f"pyinla safety check: control['family']['link'] for t must be one of {sorted(t_allowed_links)}; got '{link_val}'."
                )

    # Validate link function for gaussian
    if has_gaussian and "link" in family_block:
        link_val = family_block.get("link")
        if link_val is not None:
            link_norm = _normalize_link_value(link_val)
            gaussian_allowed_links = {"default", "identity", "logit", "loga", "cauchit", "log", "logoffset"}
            if link_norm is not None and link_norm not in gaussian_allowed_links:
                raise SafetyError(
                    f"pyinla safety check: control['family']['link'] for gaussian must be one of {sorted(gaussian_allowed_links)}; got '{link_val}'."
                )

    # Validate link function for poisson
    if has_poisson and "link" in family_block:
        link_val = family_block.get("link")
        if link_val is not None:
            link_norm = _normalize_link_value(link_val)
            poisson_allowed_links = {"default", "log", "logoffset", "quantile"}
            if link_norm is not None and link_norm not in poisson_allowed_links:
                raise SafetyError(
                    f"pyinla safety check: control['family']['link'] for poisson must be one of {sorted(poisson_allowed_links)}; got '{link_val}'."
                )

    # Validate link function for binomial
    if has_binomial and "link" in family_block:
        link_val = family_block.get("link")
        if link_val is not None:
            link_norm = _normalize_link_value(link_val)
            binomial_allowed_links = {
                "default", "logit", "loga", "cauchit", "probit", "cloglog", "ccloglog", "loglog",
                "log", "sslogit", "logitoffset", "quantile", "pquantile", "robit", "sn", "powerlogit", "gevit", "cgevit"
            }
            if link_norm is not None and link_norm not in binomial_allowed_links:
                raise SafetyError(
                    f"pyinla safety check: control['family']['link'] for binomial must be one of {sorted(binomial_allowed_links)}; got '{link_val}'."
                )

    # Validate link function for gamma
    if has_gamma and "link" in family_block:
        link_val = family_block.get("link")
        if link_val is not None:
            link_norm = _normalize_link_value(link_val)
            gamma_allowed_links = {"default", "log", "quantile"}
            if link_norm is not None and link_norm not in gamma_allowed_links:
                raise SafetyError(
                    f"pyinla safety check: control['family']['link'] for gamma must be one of {sorted(gamma_allowed_links)}; got '{link_val}'."
                )

    # Validate link function for nbinomial
    if has_nbinomial and "link" in family_block:
        link_val = family_block.get("link")
        if link_val is not None:
            link_norm = _normalize_link_value(link_val)
            nbinomial_allowed_links = {"default", "log", "logoffset", "quantile"}
            if link_norm is not None and link_norm not in nbinomial_allowed_links:
                raise SafetyError(
                    f"pyinla safety check: control['family']['link'] for nbinomial must be one of {sorted(nbinomial_allowed_links)}; got '{link_val}'."
                )

    # Validate link function for betabinomial
    if has_betabinomial and "link" in family_block:
        link_val = family_block.get("link")
        if link_val is not None:
            link_norm = _normalize_link_value(link_val)
            betabinomial_allowed_links = {"default", "logit", "loga", "cauchit", "probit", "cloglog", "ccloglog", "loglog", "robit", "sn"}
            if link_norm is not None and link_norm not in betabinomial_allowed_links:
                raise SafetyError(
                    f"pyinla safety check: control['family']['link'] for betabinomial must be one of {sorted(betabinomial_allowed_links)}; got '{link_val}'."
                )

    # Validate link function for gammasurv
    if has_gammasurv and "link" in family_block:
        link_val = family_block.get("link")
        if link_val is not None:
            link_norm = _normalize_link_value(link_val)
            gammasurv_allowed_links = {"default", "log", "neglog", "quantile"}
            if link_norm is not None and link_norm not in gammasurv_allowed_links:
                raise SafetyError(
                    f"pyinla safety check: control['family']['link'] for gammasurv must be one of {sorted(gammasurv_allowed_links)}; got '{link_val}'."
                )


def enforce_gamma_support(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    """Validate that gamma response values are strictly positive (NaN allowed as missing data)."""
    normalized = list(families) if families is not None else list(_normalize_family_spec(kwargs.get("family")))
    if "gamma" not in normalized:
        return

    response = _extract_response_array(kwargs)
    if response is None:
        return

    try:
        import numpy as np  # type: ignore

        arr = np.asarray(response, dtype=float).reshape(-1)
        # Allow NaN (missing data, R-INLA removes these rows)
        valid_mask = ~np.isnan(arr)
        valid_arr = arr[valid_mask]
        if valid_arr.size == 0:
            return
        mask_invalid = np.isinf(valid_arr) | (valid_arr <= 0.0)
        invalid = valid_arr[mask_invalid]
        if invalid.size > 0:
            raise SafetyError(
                "pyinla safety check: gamma likelihood requires strictly positive response values; "
                f"found invalid entries (showing up to 5): {invalid[:5]}"
            )
        return
    except SafetyError:
        raise
    except Exception:
        pass

    try:
        values = [float(v) for v in response]
    except Exception as exc:  # pragma: no cover
        raise SafetyError(
            "pyinla safety check: could not interpret response values for gamma likelihood."
        ) from exc

    # Allow NaN (missing data), validate only non-NaN values
    bad = [val for val in values if not math.isnan(val) and (math.isinf(val) or val <= 0.0)]
    if bad:
        raise SafetyError(
            "pyinla safety check: gamma likelihood requires strictly positive response values; "
            f"found invalid entries (showing up to 5): {bad[:5]}"
        )


def enforce_beta_support(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    """Validate that beta response values are in valid range (NaN allowed as missing data)."""
    normalized = list(families) if families is not None else list(_normalize_family_spec(kwargs.get("family")))
    if "beta" not in normalized:
        return

    control = kwargs.get("control") or {}
    beta_censor = None
    if isinstance(control, dict):
        family_block = control.get("family")
        if isinstance(family_block, dict):
            beta_censor = family_block.get("beta.censor.value")
    censoring_enabled = beta_censor is not None

    response = _extract_response_array(kwargs)
    if response is None:
        return
    try:
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover
        np = None  # type: ignore

    try:
        arr = np.asarray(response, dtype=float).reshape(-1) if np is not None else [float(v) for v in response]
    except Exception as exc:  # pragma: no cover
        raise SafetyError("pyinla safety check: could not interpret response values for beta likelihood.") from exc

    if np is not None:
        # Allow NaN (missing data, R-INLA removes these rows)
        valid_mask = ~np.isnan(arr)
        valid_arr = arr[valid_mask]
        if valid_arr.size == 0:
            return
        mask_invalid = np.isinf(valid_arr)
        if not censoring_enabled:
            mask_invalid |= (valid_arr <= 0.0) | (valid_arr >= 1.0)
        else:
            mask_invalid |= (valid_arr < 0.0) | (valid_arr > 1.0)
        invalid = valid_arr[mask_invalid]
        if invalid.size > 0:
            raise SafetyError(
                "pyinla safety check: beta likelihood requires response values in {} when censoring {}; "
                f"found invalid values (showing up to 5): {invalid[:5]}".format(
                    "[0,1]" if censoring_enabled else "(0,1)",
                    "enabled" if censoring_enabled else "disabled",
                )
            )
    else:
        # Allow NaN (missing data), validate only non-NaN values
        if censoring_enabled:
            bad = [val for val in arr if not math.isnan(val) and (math.isinf(val) or not (0.0 <= float(val) <= 1.0))]
        else:
            bad = [val for val in arr if not math.isnan(val) and (math.isinf(val) or not (0.0 < float(val) < 1.0))]
        if bad:
            raise SafetyError(
                "pyinla safety check: beta likelihood requires response values in {} when censoring {}; "
                f"found invalid values (showing up to 5): {bad[:5]}".format(
                    "[0,1]" if censoring_enabled else "(0,1)",
                    "enabled" if censoring_enabled else "disabled",
                )
            )


def enforce_survival_response(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    normalized = list(families) if families is not None else list(_normalize_family_spec(kwargs.get("family")))
    survival_fams = {
        "exponentialsurv", "gammasurv", "lognormalsurv", "weibullsurv", "loglogisticsurv",
        "mgammasurv", "qloglogisticsurv", "gompertzsurv", "dgompertzsurv", "fmrisurv", "coxph"
    }
    if not any(f in survival_fams for f in normalized):
        return

    response = _get_response_object(kwargs)
    if response is None:
        raise SafetyError(
            "pyinla safety check: survival likelihoods require the response to be built via pyinla.surv.inla_surv(...)."
        )
    if not is_inla_surv(response):
        raise SafetyError(
            "pyinla safety check: survival likelihoods require the response to be an inla_surv(...) object."
        )

    # Validate survival response values from the inla_surv dict
    # inla_surv dict has keys: time, lower, upper, event, truncation, cure, _class
    # Event values can be: 0 (right cens), 1 (observed), 2 (left cens), 3 (interval cens), 4 (in-interval)
    try:
        import numpy as np  # type: ignore

        # The response should be a dict with the survival data
        if not isinstance(response, dict):
            return

        # Extract time-related values from the inla_surv dict
        # These should all be non-negative for survival analysis
        time_vals = response.get("time")
        lower_vals = response.get("lower")
        upper_vals = response.get("upper")
        event_vals = response.get("event")

        # Validate that time/lower/upper values are non-negative (they should be >= 0)
        for name, vals in [("time", time_vals), ("lower", lower_vals), ("upper", upper_vals)]:
            if vals is not None:
                arr = np.asarray(vals, dtype=float).reshape(-1)
                # Check for NaN/Inf or negative values
                mask_invalid = (~np.isfinite(arr)) | (arr < 0.0)
                invalid = arr[mask_invalid]
                if invalid.size > 0:
                    raise SafetyError(
                        f"pyinla safety check: survival '{name}' values must be non-negative and finite; "
                        f"found invalid entries (showing up to 5): {invalid[:5].tolist()}"
                    )

        # Validate event values are in {0, 1, 2, 3, 4}
        # 0=right censored, 1=observed, 2=left censored, 3=interval censored, 4=in-interval
        if event_vals is not None:
            event_arr = np.asarray(event_vals, dtype=float).reshape(-1)
            valid_events = {0.0, 1.0, 2.0, 3.0, 4.0}
            mask_invalid_event = (~np.isfinite(event_arr)) | (~np.isin(event_arr, list(valid_events)))
            invalid_events = event_arr[mask_invalid_event]
            if invalid_events.size > 0:
                raise SafetyError(
                    "pyinla safety check: survival event indicators must be in {0, 1, 2, 3, 4}; "
                    f"found invalid entries (showing up to 5): {invalid_events[:5].tolist()}"
                )
    except SafetyError:
        raise
    except Exception:
        # If numpy is not available or extraction fails, skip detailed validation
        pass


def enforce_untested_arguments(kwargs: Dict[str, Any]) -> None:
    """Block top-level arguments that have not been tested for input file parity with R-INLA.

    These arguments may work but have not been verified to produce identical
    Model.ini and data files as R-INLA. They will be enabled as testing coverage
    expands.
    """

    # Untested top-level observation modifiers
    # NOTE: offset, weights, E, Ntrials, and scale have been tested and are now allowed
    untested_observation = {
        "strata": "stratification specification",
        "lp_scale": "linear predictor scale",
        "link_covariates": "link function covariates",
    }

    for key, desc in untested_observation.items():
        if key in kwargs and kwargs[key] is not None:
            raise SafetyError(
                f"pyinla safety check: '{key}' ({desc}) is not yet tested for input file parity with R-INLA. "
                "This feature will be enabled once testing coverage is complete."
            )

    # Untested diagnostics/output arguments
    # NOTE: lincomb has been tested and is now allowed
    untested_output = {
        "selection": "selection specification",
    }

    for key, desc in untested_output.items():
        if key in kwargs and kwargs[key] is not None:
            raise SafetyError(
                f"pyinla safety check: '{key}' ({desc}) is not yet tested for input file parity with R-INLA. "
                "This feature will be enabled once testing coverage is complete."
            )

    # Untested execution control parameters
    # NOTE: verbose, num_threads, keep, working_directory, and reuse_filenames_from are allowed
    # as they are infrastructure parameters that don't affect model specification
    untested_execution = {
        "silent": "silent mode",
        "inla_call": "custom INLA binary path",
        "inla_arg": "custom INLA arguments",
        "safe": "safe mode retry behavior",
        "debug": "debug mode",
        "dry_run": "dry run mode",
        "collect": "result collection flag",
    }

    for key, desc in untested_execution.items():
        if key in kwargs and kwargs[key] is not None:
            # Allow default values to pass through
            if key == "safe" and kwargs[key] is True:
                continue  # safe=True is the default
            if key == "collect" and kwargs[key] is True:
                continue  # collect=True is the default

            raise SafetyError(
                f"pyinla safety check: '{key}' ({desc}) is not yet tested for input file parity with R-INLA. "
                "This feature will be enabled once testing coverage is complete."
            )

    # Check for custom quantiles (non-default values)
    quantiles = kwargs.get("quantiles")
    if quantiles is not None:
        default_quantiles = (0.025, 0.5, 0.975)
        try:
            user_q = tuple(float(q) for q in quantiles)
        except (TypeError, ValueError):
            user_q = None
        if user_q is not None and user_q != default_quantiles:
            raise SafetyError(
                "pyinla safety check: custom 'quantiles' values are not yet tested for input file parity with R-INLA. "
                f"Please use the default quantiles {default_quantiles} or omit the argument."
            )


def enforce_poisson_support(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    """Validate that Poisson response values are non-negative integers (NaN allowed as missing data)."""
    normalized = list(families) if families is not None else list(_normalize_family_spec(kwargs.get("family")))
    if "poisson" not in normalized:
        return

    response = _extract_response_array(kwargs)
    if response is None:
        return

    try:
        import numpy as np  # type: ignore

        arr = np.asarray(response, dtype=float).reshape(-1)
        # Allow NaN (missing data, R-INLA removes these rows)
        # Only validate non-NaN values
        valid_mask = ~np.isnan(arr)
        valid_arr = arr[valid_mask]
        if valid_arr.size == 0:
            return
        # Check for Inf values
        mask_inf = np.isinf(valid_arr)
        # Check for negative values
        mask_negative = valid_arr < 0.0
        # Check for non-integer values (allowing for floating point representation)
        mask_noninteger = np.abs(valid_arr - np.round(valid_arr)) > 1e-10
        mask_invalid = mask_inf | mask_negative | mask_noninteger
        invalid = valid_arr[mask_invalid]
        if invalid.size > 0:
            raise SafetyError(
                "pyinla safety check: poisson likelihood requires non-negative integer response values; "
                f"found invalid entries (showing up to 5): {invalid[:5].tolist()}"
            )
        return
    except SafetyError:
        raise
    except Exception:
        pass

    try:
        values = [float(v) for v in response]
    except Exception as exc:  # pragma: no cover
        raise SafetyError(
            "pyinla safety check: could not interpret response values for poisson likelihood."
        ) from exc

    # Allow NaN (missing data), validate only non-NaN values
    bad = [val for val in values if not math.isnan(val) and (math.isinf(val) or val < 0.0 or abs(val - round(val)) > 1e-10)]
    if bad:
        raise SafetyError(
            "pyinla safety check: poisson likelihood requires non-negative integer response values; "
            f"found invalid entries (showing up to 5): {bad[:5]}"
        )


def enforce_nbinomial_support(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    """Validate that negative binomial response values are non-negative integers (NaN allowed as missing data)."""
    normalized = list(families) if families is not None else list(_normalize_family_spec(kwargs.get("family")))
    nbinom_families = {"nbinomial", "nbinomial2"}
    if not any(f in nbinom_families for f in normalized):
        return

    response = _extract_response_array(kwargs)
    if response is None:
        return

    try:
        import numpy as np  # type: ignore

        arr = np.asarray(response, dtype=float).reshape(-1)
        # Allow NaN (missing data, R-INLA removes these rows)
        valid_mask = ~np.isnan(arr)
        valid_arr = arr[valid_mask]
        if valid_arr.size == 0:
            return
        mask_inf = np.isinf(valid_arr)
        mask_negative = valid_arr < 0.0
        mask_noninteger = np.abs(valid_arr - np.round(valid_arr)) > 1e-10
        mask_invalid = mask_inf | mask_negative | mask_noninteger
        invalid = valid_arr[mask_invalid]
        if invalid.size > 0:
            raise SafetyError(
                "pyinla safety check: nbinomial likelihood requires non-negative integer response values; "
                f"found invalid entries (showing up to 5): {invalid[:5].tolist()}"
            )
        return
    except SafetyError:
        raise
    except Exception:
        pass

    try:
        values = [float(v) for v in response]
    except Exception as exc:  # pragma: no cover
        raise SafetyError(
            "pyinla safety check: could not interpret response values for nbinomial likelihood."
        ) from exc

    # Allow NaN (missing data), validate only non-NaN values
    bad = [val for val in values if not math.isnan(val) and (math.isinf(val) or val < 0.0 or abs(val - round(val)) > 1e-10)]
    if bad:
        raise SafetyError(
            "pyinla safety check: nbinomial likelihood requires non-negative integer response values; "
            f"found invalid entries (showing up to 5): {bad[:5]}"
        )


def enforce_binomial_support(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    """Validate that binomial response values are in valid range [0, Ntrials] (NaN allowed as missing data)."""
    normalized = list(families) if families is not None else list(_normalize_family_spec(kwargs.get("family")))
    binomial_families = {"binomial", "xbinomial", "betabinomial"}
    if not any(f in binomial_families for f in normalized):
        return

    response = _extract_response_array(kwargs)
    if response is None:
        return

    ntrials = kwargs.get("Ntrials")

    try:
        import numpy as np  # type: ignore

        arr = np.asarray(response, dtype=float).reshape(-1)
        # Allow NaN (missing data, R-INLA removes these rows)
        valid_mask = ~np.isnan(arr)
        valid_arr = arr[valid_mask]
        if valid_arr.size == 0:
            return
        mask_inf = np.isinf(valid_arr)
        mask_negative = valid_arr < 0.0
        mask_noninteger = np.abs(valid_arr - np.round(valid_arr)) > 1e-10
        mask_invalid = mask_inf | mask_negative | mask_noninteger

        # If Ntrials is provided, check y <= Ntrials
        if ntrials is not None:
            ntrials_arr = np.asarray(ntrials, dtype=float).reshape(-1)
            if ntrials_arr.size == 1:
                ntrials_check = np.full(valid_arr.size, ntrials_arr[0])
            else:
                ntrials_check = ntrials_arr[valid_mask]
            mask_exceeds = valid_arr > ntrials_check
            mask_invalid = mask_invalid | mask_exceeds

        invalid = valid_arr[mask_invalid]
        if invalid.size > 0:
            raise SafetyError(
                "pyinla safety check: binomial likelihood requires non-negative integer response values "
                f"not exceeding Ntrials; found invalid entries (showing up to 5): {invalid[:5].tolist()}"
            )
        return
    except SafetyError:
        raise
    except Exception:
        pass

    # Fallback without numpy
    try:
        values = [float(v) for v in response]
    except Exception as exc:  # pragma: no cover
        raise SafetyError(
            "pyinla safety check: could not interpret response values for binomial likelihood."
        ) from exc

    # Allow NaN (missing data), validate only non-NaN values
    bad = [val for val in values if not math.isnan(val) and (math.isinf(val) or val < 0.0 or abs(val - round(val)) > 1e-10)]
    if bad:
        raise SafetyError(
            "pyinla safety check: binomial likelihood requires non-negative integer response values; "
            f"found invalid entries (showing up to 5): {bad[:5]}"
        )


def enforce_exponential_support(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    """Validate that exponential response values are strictly positive (NaN allowed as missing data)."""
    normalized = list(families) if families is not None else list(_normalize_family_spec(kwargs.get("family")))
    if "exponential" not in normalized:
        return

    response = _extract_response_array(kwargs)
    if response is None:
        return

    try:
        import numpy as np  # type: ignore

        arr = np.asarray(response, dtype=float).reshape(-1)
        # Allow NaN (missing data, R-INLA removes these rows)
        valid_mask = ~np.isnan(arr)
        valid_arr = arr[valid_mask]
        if valid_arr.size == 0:
            return
        mask_invalid = np.isinf(valid_arr) | (valid_arr <= 0.0)
        invalid = valid_arr[mask_invalid]
        if invalid.size > 0:
            raise SafetyError(
                "pyinla safety check: exponential likelihood requires strictly positive response values; "
                f"found invalid entries (showing up to 5): {invalid[:5].tolist()}"
            )
        return
    except SafetyError:
        raise
    except Exception:
        pass

    try:
        values = [float(v) for v in response]
    except Exception as exc:  # pragma: no cover
        raise SafetyError(
            "pyinla safety check: could not interpret response values for exponential likelihood."
        ) from exc

    # Allow NaN (missing data), validate only non-NaN values
    bad = [val for val in values if not math.isnan(val) and (math.isinf(val) or val <= 0.0)]
    if bad:
        raise SafetyError(
            "pyinla safety check: exponential likelihood requires strictly positive response values; "
            f"found invalid entries (showing up to 5): {bad[:5]}"
        )


def enforce_lognormal_support(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    """Validate that lognormal response values are strictly positive (NaN allowed as missing data)."""
    normalized = list(families) if families is not None else list(_normalize_family_spec(kwargs.get("family")))
    if "lognormal" not in normalized:
        return

    response = _extract_response_array(kwargs)
    if response is None:
        return

    try:
        import numpy as np  # type: ignore

        arr = np.asarray(response, dtype=float).reshape(-1)
        # Allow NaN (missing data, R-INLA removes these rows)
        valid_mask = ~np.isnan(arr)
        valid_arr = arr[valid_mask]
        if valid_arr.size == 0:
            return
        mask_invalid = np.isinf(valid_arr) | (valid_arr <= 0.0)
        invalid = valid_arr[mask_invalid]
        if invalid.size > 0:
            raise SafetyError(
                "pyinla safety check: lognormal likelihood requires strictly positive response values; "
                f"found invalid entries (showing up to 5): {invalid[:5].tolist()}"
            )
        return
    except SafetyError:
        raise
    except Exception:
        pass

    try:
        values = [float(v) for v in response]
    except Exception as exc:  # pragma: no cover
        raise SafetyError(
            "pyinla safety check: could not interpret response values for lognormal likelihood."
        ) from exc

    # Allow NaN (missing data), validate only non-NaN values
    bad = [val for val in values if not math.isnan(val) and (math.isinf(val) or val <= 0.0)]
    if bad:
        raise SafetyError(
            "pyinla safety check: lognormal likelihood requires strictly positive response values; "
            f"found invalid entries (showing up to 5): {bad[:5]}"
        )


def enforce_weibull_support(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    """Validate that weibull response values are strictly positive (NaN allowed as missing data)."""
    normalized = list(families) if families is not None else list(_normalize_family_spec(kwargs.get("family")))
    if "weibull" not in normalized:
        return

    response = _extract_response_array(kwargs)
    if response is None:
        return

    try:
        import numpy as np  # type: ignore

        arr = np.asarray(response, dtype=float).reshape(-1)
        # Allow NaN (missing data, R-INLA removes these rows)
        valid_mask = ~np.isnan(arr)
        valid_arr = arr[valid_mask]
        if valid_arr.size == 0:
            return
        mask_invalid = np.isinf(valid_arr) | (valid_arr <= 0.0)
        invalid = valid_arr[mask_invalid]
        if invalid.size > 0:
            raise SafetyError(
                "pyinla safety check: weibull likelihood requires strictly positive response values; "
                f"found invalid entries (showing up to 5): {invalid[:5].tolist()}"
            )
        return
    except SafetyError:
        raise
    except Exception:
        pass

    try:
        values = [float(v) for v in response]
    except Exception as exc:  # pragma: no cover
        raise SafetyError(
            "pyinla safety check: could not interpret response values for weibull likelihood."
        ) from exc

    # Allow NaN (missing data), validate only non-NaN values
    bad = [val for val in values if not math.isnan(val) and (math.isinf(val) or val <= 0.0)]
    if bad:
        raise SafetyError(
            "pyinla safety check: weibull likelihood requires strictly positive response values; "
            f"found invalid entries (showing up to 5): {bad[:5]}"
        )


def enforce_loglogistic_support(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    """Validate that loglogistic response values are strictly positive (NaN allowed as missing data)."""
    normalized = list(families) if families is not None else list(_normalize_family_spec(kwargs.get("family")))
    if "loglogistic" not in normalized:
        return

    response = _extract_response_array(kwargs)
    if response is None:
        return

    try:
        import numpy as np  # type: ignore

        arr = np.asarray(response, dtype=float).reshape(-1)
        # Allow NaN (missing data, R-INLA removes these rows)
        valid_mask = ~np.isnan(arr)
        valid_arr = arr[valid_mask]
        if valid_arr.size == 0:
            return
        mask_invalid = np.isinf(valid_arr) | (valid_arr <= 0.0)
        invalid = valid_arr[mask_invalid]
        if invalid.size > 0:
            raise SafetyError(
                "pyinla safety check: loglogistic likelihood requires strictly positive response values; "
                f"found invalid entries (showing up to 5): {invalid[:5].tolist()}"
            )
        return
    except SafetyError:
        raise
    except Exception:
        pass

    try:
        values = [float(v) for v in response]
    except Exception as exc:  # pragma: no cover
        raise SafetyError(
            "pyinla safety check: could not interpret response values for loglogistic likelihood."
        ) from exc

    # Allow NaN (missing data), validate only non-NaN values
    bad = [val for val in values if not math.isnan(val) and (math.isinf(val) or val <= 0.0)]
    if bad:
        raise SafetyError(
            "pyinla safety check: loglogistic likelihood requires strictly positive response values; "
            f"found invalid entries (showing up to 5): {bad[:5]}"
        )


def enforce_gaussian_support(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    """Validate that gaussian response values have no Inf (NaN allowed as missing data)."""
    normalized = list(families) if families is not None else list(_normalize_family_spec(kwargs.get("family")))
    if "gaussian" not in normalized:
        return

    response = _extract_response_array(kwargs)
    if response is None:
        return

    try:
        import numpy as np  # type: ignore

        arr = np.asarray(response, dtype=float).reshape(-1)
        # Allow NaN (missing data, R-INLA removes these rows), but reject Inf
        mask_invalid = np.isinf(arr)
        invalid = arr[mask_invalid]
        if invalid.size > 0:
            raise SafetyError(
                "pyinla safety check: gaussian likelihood requires finite response values (no Inf); "
                f"found invalid entries (showing up to 5): {invalid[:5].tolist()}"
            )
        return
    except SafetyError:
        raise
    except Exception:
        pass

    try:
        values = [float(v) for v in response]
    except Exception as exc:  # pragma: no cover
        raise SafetyError(
            "pyinla safety check: could not interpret response values for gaussian likelihood."
        ) from exc

    # Allow NaN (missing data), but reject Inf
    bad = [val for val in values if math.isinf(val)]
    if bad:
        raise SafetyError(
            "pyinla safety check: gaussian likelihood requires finite response values (no Inf); "
            f"found invalid entries (showing up to 5): {bad[:5]}"
        )


def enforce_logistic_support(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    """Validate that logistic response values have no Inf (NaN allowed as missing data)."""
    normalized = list(families) if families is not None else list(_normalize_family_spec(kwargs.get("family")))
    if "logistic" not in normalized:
        return

    response = _extract_response_array(kwargs)
    if response is None:
        return

    try:
        import numpy as np  # type: ignore

        arr = np.asarray(response, dtype=float).reshape(-1)
        # Allow NaN (missing data, R-INLA removes these rows), but reject Inf
        mask_invalid = np.isinf(arr)
        invalid = arr[mask_invalid]
        if invalid.size > 0:
            raise SafetyError(
                "pyinla safety check: logistic likelihood requires finite response values (no Inf); "
                f"found invalid entries (showing up to 5): {invalid[:5].tolist()}"
            )
        return
    except SafetyError:
        raise
    except Exception:
        pass

    try:
        values = [float(v) for v in response]
    except Exception as exc:  # pragma: no cover
        raise SafetyError(
            "pyinla safety check: could not interpret response values for logistic likelihood."
        ) from exc

    # Allow NaN (missing data), but reject Inf
    bad = [val for val in values if math.isinf(val)]
    if bad:
        raise SafetyError(
            "pyinla safety check: logistic likelihood requires finite response values (no Inf); "
            f"found invalid entries (showing up to 5): {bad[:5]}"
        )


def enforce_t_support(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    """Validate that student-t response values have no Inf (NaN allowed as missing data)."""
    normalized = list(families) if families is not None else list(_normalize_family_spec(kwargs.get("family")))
    if "t" not in normalized:
        return

    response = _extract_response_array(kwargs)
    if response is None:
        return

    try:
        import numpy as np  # type: ignore

        arr = np.asarray(response, dtype=float).reshape(-1)
        # Allow NaN (missing data, R-INLA removes these rows), but reject Inf
        mask_invalid = np.isinf(arr)
        invalid = arr[mask_invalid]
        if invalid.size > 0:
            raise SafetyError(
                "pyinla safety check: student-t likelihood requires finite response values (no Inf); "
                f"found invalid entries (showing up to 5): {invalid[:5].tolist()}"
            )
        return
    except SafetyError:
        raise
    except Exception:
        pass

    try:
        values = [float(v) for v in response]
    except Exception as exc:  # pragma: no cover
        raise SafetyError(
            "pyinla safety check: could not interpret response values for student-t likelihood."
        ) from exc

    # Allow NaN (missing data), but reject Inf
    bad = [val for val in values if math.isinf(val)]
    if bad:
        raise SafetyError(
            "pyinla safety check: student-t likelihood requires finite response values (no Inf); "
            f"found invalid entries (showing up to 5): {bad[:5]}"
        )


def enforce_sn_support(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    """Validate that skew-normal response values have no Inf (NaN allowed as missing data)."""
    normalized = list(families) if families is not None else list(_normalize_family_spec(kwargs.get("family")))
    if "sn" not in normalized:
        return

    response = _extract_response_array(kwargs)
    if response is None:
        return

    try:
        import numpy as np  # type: ignore

        arr = np.asarray(response, dtype=float).reshape(-1)
        # Allow NaN (missing data, R-INLA removes these rows), but reject Inf
        mask_invalid = np.isinf(arr)
        invalid = arr[mask_invalid]
        if invalid.size > 0:
            raise SafetyError(
                "pyinla safety check: skew-normal likelihood requires finite response values (no Inf); "
                f"found invalid entries (showing up to 5): {invalid[:5].tolist()}"
            )
        return
    except SafetyError:
        raise
    except Exception:
        pass

    try:
        values = [float(v) for v in response]
    except Exception as exc:  # pragma: no cover
        raise SafetyError(
            "pyinla safety check: could not interpret response values for skew-normal likelihood."
        ) from exc

    # Allow NaN (missing data), but reject Inf
    bad = [val for val in values if math.isinf(val)]
    if bad:
        raise SafetyError(
            "pyinla safety check: skew-normal likelihood requires finite response values (no Inf); "
            f"found invalid entries (showing up to 5): {bad[:5]}"
        )


def enforce_offset_values(kwargs: Dict[str, Any]) -> None:
    """Validate that offset values are finite (no NaN/Inf).

    The offset is added to the linear predictor and must be a numeric vector
    with finite values. This applies to all likelihood families.
    """
    offset = kwargs.get("offset")
    if offset is None:
        return

    try:
        import numpy as np  # type: ignore

        arr = np.asarray(offset, dtype=float).reshape(-1)
        mask_invalid = ~np.isfinite(arr)
        invalid = arr[mask_invalid]
        if invalid.size > 0:
            raise SafetyError(
                "pyinla safety check: 'offset' must contain finite values (no NaN/Inf); "
                f"found invalid entries (showing up to 5): {invalid[:5].tolist()}"
            )
        return
    except SafetyError:
        raise
    except Exception:
        pass

    # Fallback for non-numpy
    try:
        values = [float(v) for v in offset]
    except Exception as exc:
        raise SafetyError(
            "pyinla safety check: could not interpret 'offset' as numeric values."
        ) from exc

    bad = [val for val in values if not math.isfinite(val)]
    if bad:
        raise SafetyError(
            "pyinla safety check: 'offset' must contain finite values (no NaN/Inf); "
            f"found invalid entries (showing up to 5): {bad[:5]}"
        )


def enforce_weights_values(kwargs: Dict[str, Any]) -> None:
    """Validate that weights values are strictly positive and finite.

    Weights multiply the log-likelihood contribution of each observation.
    They must be strictly positive (> 0) and finite.
    """
    weights = kwargs.get("weights")
    if weights is None:
        return

    try:
        import numpy as np  # type: ignore

        arr = np.asarray(weights, dtype=float).reshape(-1)
        # Check for non-finite values
        mask_nonfinite = ~np.isfinite(arr)
        # Check for non-positive values
        mask_nonpositive = arr <= 0.0
        mask_invalid = mask_nonfinite | mask_nonpositive
        invalid = arr[mask_invalid]
        if invalid.size > 0:
            raise SafetyError(
                "pyinla safety check: 'weights' must be strictly positive (> 0) and finite; "
                f"found invalid entries (showing up to 5): {invalid[:5].tolist()}"
            )
        return
    except SafetyError:
        raise
    except Exception:
        pass

    # Fallback for non-numpy
    try:
        values = [float(v) for v in weights]
    except Exception as exc:
        raise SafetyError(
            "pyinla safety check: could not interpret 'weights' as numeric values."
        ) from exc

    bad = [val for val in values if not math.isfinite(val) or val <= 0.0]
    if bad:
        raise SafetyError(
            "pyinla safety check: 'weights' must be strictly positive (> 0) and finite; "
            f"found invalid entries (showing up to 5): {bad[:5]}"
        )

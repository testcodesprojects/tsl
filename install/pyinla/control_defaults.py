# control_defaults.py
"""
Python equivalents of set.default.arguments.R.

Each control_* function returns a dict of default control arguments,
mimicking the R style where the functions exist largely to enable
tab-completion and to materialize defaults (unless overridden).

Notes
-----
- We keep names close to R but use snake_case (e.g., control.update -> control_update).
- Where R used `INLA::control.*()` as defaults inside another control function,
  we emulate by setting the default to None and filling it in at call-time.
- A small '__control__' tag is attached for traceability.
"""

from __future__ import annotations

import math
import sys
from typing import Any, Dict, List, Optional, Tuple


def _ctrl_object(args: Dict[str, Any], name: str, check: bool = False) -> Dict[str, Any]:
    """
    Mirror the R `ctrl_object(as.list(environment()), name, check=FALSE)` usage.
    We simply attach a tag; no validation since R used check=FALSE here.
    """
    # Remove internal-only keys if present
    args = {k: v for k, v in args.items() if k not in {"check", "name"}}
    args["__control__"] = name
    return args


# ----------------------------
# control.update
# ----------------------------
def control_update(*, result: Any = None) -> Dict[str, Any]:
    """Update the joint posterior for the hyperparameters from result."""
    return _ctrl_object(locals(), "update", check=False)


# ----------------------------
# control.sem
# ----------------------------
def control_sem(*, B: Any = None, idx: int = 0) -> Dict[str, Any]:
    """
    Parameters to family 'sem'
    - B: symbolic B-matrix (strings per entry)
    - idx: which diagonal element to use for the variance
    """
    return _ctrl_object(locals(), "sem", check=False)


# ----------------------------
# control.lincomb
# ----------------------------
def control_lincomb(*, verbose: bool = False) -> Dict[str, Any]:
    """Control for linear combinations."""
    return _ctrl_object(locals(), "lincomb", check=False)


# ----------------------------
# control.group
# ----------------------------
def control_group(
    *,
    model: str = "exchangeable",
    order: Optional[int] = None,
    cyclic: bool = False,
    graph: Any = None,
    scale_model: bool = True,
    adjust_for_con_comp: bool = True,
    hyper: Any = None,
    initial: Any = None,  # OBSOLETE (kept for compatibility)
    fixed: Any = None,    # OBSOLETE
    prior: Any = None,    # OBSOLETE
    param: Any = None,    # OBSOLETE
) -> Dict[str, Any]:
    return _ctrl_object(locals(), "group", check=False)


# ----------------------------
# control.scopy
# ----------------------------
def control_scopy(*, covariate: Any = None, n: int = 11) -> Dict[str, Any]:
    return _ctrl_object(locals(), "scopy", check=False)


# ----------------------------
# control.mix
# ----------------------------
def control_mix(
    *,
    model: Optional[str] = None,  # currently only "gaussian" implemented in R
    hyper: Any = None,
    initial: Any = None,  # OBSOLETE
    fixed: Any = None,    # OBSOLETE
    prior: Any = None,    # OBSOLETE
    param: Any = None,    # OBSOLETE
    npoints: int = 101,
    integrator: str = "default",  # "default" | "quadrature" | "simpson"
) -> Dict[str, Any]:
    return _ctrl_object(locals(), "mix", check=False)


# ----------------------------
# control.pom
# ----------------------------
def control_pom(*, cdf: str = "logit", fast: bool = False) -> Dict[str, Any]:
    return _ctrl_object(locals(), "pom", check=False)


# ----------------------------
# control.link
# ----------------------------
def control_link(
    *,
    model: str = "default",
    order: Optional[int] = None,
    variant: Optional[int] = None,
    hyper: Any = None,
    quantile: Any = None,
    a: float = 1.0,
    initial: Any = None,  # OBSOLETE
    fixed: Any = None,    # OBSOLETE
    prior: Any = None,    # OBSOLETE
    param: Any = None,    # OBSOLETE
) -> Dict[str, Any]:
    return _ctrl_object(locals(), "link", check=False)


# ----------------------------
# inla.set.f.default
# ----------------------------
def inla_set_f_default(*args, **kwargs) -> Dict[str, float]:
    """Equivalent to R `inla.set.f.default()`: returns minimal f() defaults."""
    return {"diagonal": 1.0e-4}


# ----------------------------
# control.expert
# ----------------------------
def control_expert(
    *,
    cpo_manual: bool = False,
    cpo_idx: int = -1,
    disable_gaussian_check: bool = False,
    jp: Any = None,
    dot_product_gain: bool = False,
    globalconstr: Dict[str, Any] = None,
    opt_solve: bool = False,
    opt_num_threads: bool = True,
) -> Dict[str, Any]:
    if globalconstr is None:
        globalconstr = {"A": None, "e": None}
    return _ctrl_object(locals(), "expert", check=False)


# ----------------------------
# control.gcpo
# ----------------------------
def control_gcpo(
    *,
    enable: bool = False,
    num_level_sets: int = -1,
    size_max: int = 32,
    strategy: str = "posterior",  # "posterior" | "prior"
    groups: Any = None,
    selection: Any = None,
    group_selection: Any = None,
    friends: Any = None,
    weights: Any = None,
    verbose: bool = False,
    epsilon: float = 0.005,
    prior_diagonal: float = 1e-4,
    correct_hyperpar: bool = True,
    keep: Any = None,
    remove: Any = None,
    remove_fixed: bool = True,
    type_cv: str = "single",  # "single" | "joint"
) -> Dict[str, Any]:
    return _ctrl_object(locals(), "gcpo", check=False)


# ----------------------------
# control.compute
# ----------------------------
def control_compute(
    *,
    openmp_strategy: str = "default",  # 'small','medium','large','huge','default','pardiso'
    hyperpar: bool = True,
    return_marginals: bool = True,
    return_marginals_predictor: bool = False,
    dic: bool = False,
    mlik: bool = True,
    cpo: bool = False,
    po: bool = False,
    waic: bool = False,
    residuals: bool = False,
    q: bool = False,
    config: bool = False,  # Match R-INLA default
    likelihood_info: bool = False,
    smtp: Optional[str] = None,  # 'default','taucs','band','pardiso'
    graph: bool = False,
    internal_opt: Any = None,
    save_memory: Any = None,
    control_gcpo: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if control_gcpo is None:
        control_gcpo = control_gcpo_default()
    return _ctrl_object(locals(), "compute", check=False)


# ----------------------------
# control.lp.scale
# ----------------------------
def control_lp_scale(*, hyper: Any = None) -> Dict[str, Any]:
    return _ctrl_object(locals(), "lp_scale", check=False)


# ----------------------------
# control.pardiso
# ----------------------------
def control_pardiso(
    *,
    verbose: bool = False,
    debug: bool = False,
    parallel_reordering: bool = True,
    nrhs: int = -1,
) -> Dict[str, Any]:
    return _ctrl_object(locals(), "pardiso", check=False)


# ----------------------------
# control.stiles
# ----------------------------
def control_stiles(
    *,
    verbose: bool = False,
    tile_size: int = 0,
) -> Dict[str, Any]:
    return _ctrl_object(locals(), "stiles", check=False)


# ----------------------------
# control.taucs
# ----------------------------
def control_taucs(*, block_size: int = 64) -> Dict[str, Any]:
    return _ctrl_object(locals(), "taucs", check=False)


# ----------------------------
# control.numa
# ----------------------------
def control_numa(*, enable: Any = None) -> Dict[str, Any]:
    return _ctrl_object(locals(), "numa", check=False)


# ----------------------------
# control.bgev
# ----------------------------
def control_bgev(
    *,
    q_location: float = 0.5,
    q_spread: float = 0.25,               # must be < 0.5
    q_mix: Tuple[float, float] = (0.10, 0.20),
    beta_ab: int = 5,
) -> Dict[str, Any]:
    return _ctrl_object(locals(), "bgev", check=False)


# ----------------------------
# control.family
# ----------------------------
def control_family(
    *,
    dummy: int = 0,
    hyper: Any = None,
    initial: Any = None,     # OBSOLETE
    prior: Any = None,       # OBSOLETE
    param: Any = None,       # OBSOLETE
    fixed: Any = None,       # OBSOLETE
    link: str = "default",   # OBSOLETE! use control_link=list(model=) in R; kept for parity
    sn_shape_max: float = 5.0,
    gev_scale_xi: float = 0.1,
    control_bgev: Any = None,
    cenpoisson_I: Tuple[int, int] = (-1, -1),
    beta_censor_value: float = 0.0,
    variant: int = 0,
    link_simple: str = "default",
    control_mix: Any = None,
    control_pom: Any = None,
    control_link: Optional[Dict[str, Any]] = None,
    control_sem: Any = None,
    cloglike: Any = None,
) -> Dict[str, Any]:
    if control_link is None:
        control_link = control_link_default()
    return _ctrl_object(locals(), "family", check=False)


# ----------------------------
# control.fixed
# ----------------------------
def control_fixed(
    *,
    cdf: Any = None,
    quantiles: Any = None,
    expand_factor_strategy: str = "model.matrix",  # "model.matrix" | "inla"
    mean: float | Dict[str, float] = 0.0,
    mean_intercept: float = 0.0,
    prec: float | Dict[str, float] = 0.001,
    prec_intercept: float = 0.0,
    compute: bool = True,
    correlation_matrix: bool = False,
    remove_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return _ctrl_object(locals(), "fixed", check=False)


# ----------------------------
# control.vb
# ----------------------------
def control_vb(
    *,
    enable: str | bool = "auto",
    strategy: str = "mean",  # "mean" | "variance"
    verbose: bool = True,
    iter_max: int = 25,
    emergency: float = 25.0,
    f_enable_limit: Tuple[int, int, int, int] = (30, 25, 1024, 768),
    hessian_update: int = 2,
    hessian_strategy: str = "default",  # "default" | "full" | "partial" | "diagonal"
) -> Dict[str, Any]:
    return _ctrl_object(locals(), "vb", check=False)


# ----------------------------
# control.inla
# ----------------------------
def control_inla(
    *,
    strategy: str = "auto",          # 'auto','gaussian','simplified.laplace','laplace','adaptive'
    int_strategy: str = "auto",      # 'auto','ccd','grid','eb','user','user.std'
    int_design: Any = None,          # matrix-like of [thetas..., weight]
    interpolator: str = "auto",      # 'auto','nearest','quadratic','weighted.distance','ccd',...
    fast: bool = True,
    linear_correction: Optional[bool] = None,
    h: float = 0.005,
    dz: float = 0.75,
    diff_logdens: float = 6.0,
    print_joint_hyper: bool = True,
    force_diagonal: bool = False,
    skip_configurations: bool = True,
    adjust_weights: bool = True,
    tolerance: float = 0.005,
    tolerance_f: Optional[float] = None,
    tolerance_g: Optional[float] = None,
    tolerance_x: Optional[float] = None,
    tolerance_step: Optional[float] = None,
    restart: int = 0,
    optimiser: str = "default",      # 'gsl'|'default'
    verbose: Optional[bool] = None,
    reordering: str = "auto",
    cpo_diff: Optional[float] = None,
    npoints: int = 9,
    cutoff: float = 1e-4,
    adapt_hessian_mode: Optional[bool] = None,
    adapt_hessian_max_trials: Optional[int] = None,
    adapt_hessian_scale: Optional[float] = None,
    adaptive_max: int = 25,
    huge: bool = False,  # OBSOLETE (kept for parity)
    step_len: float = 0.0,
    stencil: int = 5,
    lincomb_derived_correlation_matrix: bool = False,
    diagonal: float = 0.0,
    numint_maxfeval: int = 100000,
    numint_relerr: float = 1e-5,
    numint_abserr: float = 1e-6,
    cmin: float = float("-inf"),
    b_strategy: str = "keep",   # "keep" | "skip"
    step_factor: float = -0.1,
    global_node_factor: float = 2.0,
    global_node_degree: int = 2**31 - 1,  # mimic .Machine$integer.max
    stupid_search: bool = True,
    stupid_search_max_iter: int = 1000,
    stupid_search_factor: float = 1.05,
    control_vb: Optional[Dict[str, Any]] = None,
    num_gradient: str = "central",  # "forward" | "central"
    num_hessian: str = "central",   # "forward" | "central"
    optimise_strategy: str = "smart",  # "plain" | "smart"
    use_directions: Any = True,     # bool or matrix-like
    constr_marginal_diagonal: float = math.sqrt(sys.float_info.epsilon),
    improved_simplified_laplace: bool = False,
    parallel_linesearch: bool = False,
    compute_initial_values: bool = True,
    hessian_correct_skewness_only: bool = True,
) -> Dict[str, Any]:
    if control_vb is None:
        control_vb = control_vb_default()
    return _ctrl_object(locals(), "inla", check=False)


# ----------------------------
# control.predictor
# ----------------------------
def control_predictor(
    *,
    hyper: Any = None,
    fixed: Any = None,     # OBSOLETE
    prior: Any = None,     # OBSOLETE
    param: Any = None,     # OBSOLETE
    initial: Any = None,   # OBSOLETE
    compute: bool = False,
    cdf: Any = None,
    quantiles: Any = None,
    cross: Any = None,
    A: Any = None,
    precision: float = math.exp(15.0),
    link: Any = None,
) -> Dict[str, Any]:
    return _ctrl_object(locals(), "predictor", check=False)


# ----------------------------
# control.mode
# ----------------------------
def control_mode(
    *,
    result: Any = None,
    theta: Any = None,
    x: Any = None,
    restart: bool = True,
    fixed: bool = False,
) -> Dict[str, Any]:
    return _ctrl_object(locals(), "mode", check=False)


# ----------------------------
# control.hazard
# ----------------------------
def control_hazard(
    *,
    model: str = "rw1",   # 'rw1','rw2','iid'
    hyper: Any = None,
    fixed: bool = False,     # OBSOLETE
    initial: Any = None,     # OBSOLETE
    prior: Any = None,       # OBSOLETE
    param: Any = None,       # OBSOLETE
    constr: bool = True,
    diagonal: Optional[float] = None,
    n_intervals: int = 15,
    cutpoints: Any = None,
    strata_name: Optional[str] = None,
    scale_model: Any = None,  # None -> use inla.getOption("scale.model.default") in R
) -> Dict[str, Any]:
    return _ctrl_object(locals(), "hazard", check=False)


# ------------------------------------------------------------------
# Convenience aliases mirroring the R "inla.set.control.*.default"
# ------------------------------------------------------------------
def inla_set_control_update_default(**kwargs):        return control_update(**kwargs)
def inla_set_control_lincomb_default(**kwargs):       return control_lincomb(**kwargs)
def inla_set_control_group_default(**kwargs):         return control_group(**kwargs)
def inla_set_control_scopy_default(**kwargs):         return control_scopy(**kwargs)
def inla_set_control_mix_default(**kwargs):           return control_mix(**kwargs)
def inla_set_control_pom_default(**kwargs):           return control_pom(**kwargs)
def inla_set_control_link_default(**kwargs):          return control_link(**kwargs)
def inla_set_control_expert_default(**kwargs):        return control_expert(**kwargs)
def inla_set_control_gcpo_default(**kwargs):          return control_gcpo(**kwargs)
def inla_set_control_compute_default(**kwargs):       return control_compute(**kwargs)
def inla_set_control_lp_scale_default(**kwargs):      return control_lp_scale(**kwargs)
def inla_set_control_pardiso_default(**kwargs):       return control_pardiso(**kwargs)
def inla_set_control_stiles_default(**kwargs):        return control_stiles(**kwargs)
def inla_set_control_taucs_default(**kwargs):         return control_taucs(**kwargs)
def inla_set_control_bgev_default(**kwargs):          return control_bgev(**kwargs)
def inla_set_control_family_default(**kwargs):        return control_family(**kwargs)
def inla_set_control_fixed_default(**kwargs):         return control_fixed(**kwargs)
def inla_set_control_vb_default(**kwargs):            return control_vb(**kwargs)
def inla_set_control_inla_default(**kwargs):          return control_inla(**kwargs)
def inla_set_control_predictor_default(**kwargs):     return control_predictor(**kwargs)
def inla_set_control_mode_default(**kwargs):          return control_mode(**kwargs)
def inla_set_control_hazard_default(**kwargs):        return control_hazard(**kwargs)
def inla_set_control_sem_default(**kwargs):           return control_sem(**kwargs)

# Shorthands used internally above for default population
def control_gcpo_default(): return control_gcpo()
def control_link_default(): return control_link()
def control_vb_default():   return control_vb()


__all__ = [name for name in globals().keys() if name.startswith("control_") or name.startswith("inla_set_")] + ["inla_set_f_default"]

"""
INLA Models Database.

This module provides the INLAModels class which serves as the interface to all
INLA model definitions. The actual model definitions have been organized into
separate modules in the models/ package for better maintainability.

Model definitions are split into:
- model_defs/latent.py: Latent model definitions (random effects)
- model_defs/likelihood.py: Likelihood model definitions (observation distributions)
- model_defs/other_sections.py: Other sections (group, scopy, mix, link, etc.)
"""

import os
import numpy as np
import warnings
import copy
import numbers
from typing import Any, Dict, List, Optional, Tuple, Union

# Import model definitions from the model_defs package
from .model_defs import (
    LATENT_MODELS,
    LIKELIHOOD_MODELS,
    GROUP_MODELS,
    SCOPY_MODELS,
    MIX_MODELS,
    LINK_MODELS,
    PREDICTOR_MODELS,
    HAZARD_MODELS,
    PRIOR_MODELS,
    WRAPPER_MODELS,
    LP_SCALE_MODELS,
    copy_clinear_to_theta,
    copy_clinear_from_theta,
)


class INLAModels:
    """
    A Pythonic representation of the R-INLA model definition database.
    """
    _SPECIAL_NUMBER = 1048576.0

    def __init__(self):
        """Initializes the INLAModels database."""
        self._models = None
        self._warnings_shown = set()

    def _get_latent_section(self) -> dict:
        """Returns all available latent models."""
        return LATENT_MODELS

    def _get_group_section(self) -> dict:
        """Returns all available group correlation models."""
        return GROUP_MODELS

    def _get_scopy_section(self) -> dict:
        """Returns allowed models for the 'scopy' model component."""
        return SCOPY_MODELS

    def _get_mix_section(self) -> dict:
        """Returns all available mixture models for likelihoods."""
        return MIX_MODELS

    def _get_link_section(self) -> dict:
        """Returns all available link functions."""
        return LINK_MODELS

    def _get_predictor_section(self) -> dict:
        """Returns properties for the main predictor."""
        return PREDICTOR_MODELS

    def _get_hazard_section(self) -> dict:
        """Returns all available hazard models for survival analysis."""
        return HAZARD_MODELS

    def _get_prior_section(self) -> dict:
        """Returns all available prior distributions for hyperparameters."""
        return PRIOR_MODELS

    def _get_wrapper_section(self) -> dict:
        """Returns wrapper models."""
        return WRAPPER_MODELS

    def _get_lp_scale_section(self) -> dict:
        """Returns the model for the linear predictor scaling."""
        return LP_SCALE_MODELS

    def _get_likelihood_section(self) -> dict:
        """Returns all available likelihood models."""
        return LIKELIHOOD_MODELS

    # Static helper methods for bounded parameters
    @staticmethod
    def _copy_clinear_to_theta(x, low, high):
        """Transform for 'copy'/'clinear' betas with optional bounds."""
        return copy_clinear_to_theta(x, low, high)

    @staticmethod
    def _copy_clinear_from_theta(z, low, high):
        """Inverse transform for 'copy'/'clinear' betas with optional bounds."""
        return copy_clinear_from_theta(z, low, high)

    # ========================================================================
    # Public Methods
    # ========================================================================

    def available_likelihoods(self) -> set:
        return set(self._get_likelihood_section().keys())

    def get_models(self) -> dict:
        """Returns all model sections as a dictionary."""
        if self._models is None:
            if os.environ.get("PYINLA_VERBOSE_MODELS"):
                print("Building model database for the first time...")
            self._models = {
                'latent': self._get_latent_section(),
                'group': self._get_group_section(),
                'scopy': self._get_scopy_section(),
                'mix': self._get_mix_section(),
                'link': self._get_link_section(),
                'predictor': self._get_predictor_section(),
                'hazard': self._get_hazard_section(),
                'likelihood': self._get_likelihood_section(),
                'prior': self._get_prior_section(),
                'wrapper': self._get_wrapper_section(),
                'lp_scale': self._get_lp_scale_section()
            }
        return self._models

    def get_model_properties(self, model_name: str, section: str,
                            stop_on_error: bool = True, ignore_case: bool = True) -> dict | None:
        """
        Retrieves the properties for a specific model from a given section.

        This method is the Python equivalent of R-INLA's `inla.model.properties()`
        and `inla.model.properties.generic()`.

        Args:
            model_name: The name of the model (e.g., 'rw1', 'poisson').
            section: The section to search in (e.g., 'latent', 'likelihood').
            stop_on_error: If True, raises a ValueError for unknown models.
            ignore_case: If True, performs a case-insensitive search.

        Returns:
            A dictionary of model properties, or None if not found and stop_on_error is False.
        """
        if not section:
            raise ValueError("Argument 'section' cannot be empty.")
        if not model_name:
            raise ValueError("Argument 'model_name' cannot be empty.")

        all_models = self.get_models()

        if section not in all_models:
            if stop_on_error:
                raise ValueError(f"Unknown section '{section}'. Valid sections are: {list(all_models.keys())}")
            return None

        section_models = all_models[section]

        # Prepare for case-insensitive search if requested
        search_name = model_name.lower() if ignore_case else model_name

        found_model_key = None
        for key in section_models.keys():
            current_key = key.lower() if ignore_case else key
            if current_key == search_name:
                found_model_key = key
                break

        if found_model_key is None:
            if stop_on_error:
                raise ValueError(
                    f"Model '{model_name}' not found in section '{section}'.\n"
                    f"Valid models are: {list(section_models.keys())}"
                )
            return None

        # Status Check Logic
        model_properties = section_models[found_model_key]
        status = model_properties.get('status')

        if status:
            status_key = f"{section}.{model_name.lower()}"
            status_core = status.split(":")[0].lower()

            if status_core == "experimental":
                if status_key not in self._warnings_shown:
                    warnings.warn(
                        f"Model '{model_name}' in section '{section}' is marked as '{status}'; "
                        "changes may appear at any time. Further warnings for this model are disabled."
                    )
                    self._warnings_shown.add(status_key)

            elif status_core == "disabled":
                raise RuntimeError(
                    f"Model '{model_name}' in section '{section}' is disabled: '{status}'.\n"
                    "Usage is not recommended or supported. Check documentation for alternatives."
                )

            elif status_core == "changed":
                if status_key not in self._warnings_shown:
                    warnings.warn(
                        f"Model '{model_name}' in section '{section}' has changed: '{status}'.\n"
                        "The model definition is not backward compatible. Please review documentation."
                    )
                    self._warnings_shown.add(status_key)

        return model_properties

    def is_model(self, model_name: str, section: str,
                stop_on_error: bool = True, ignore_case: bool = False) -> bool:
        """
        Checks if a model name is valid within a given section.
        Equivalent to R-INLA's `inla.is.model()`.
        """
        all_models = self.get_models()

        if section not in all_models:
            if stop_on_error:
                raise ValueError(f"Unknown section '{section}'.")
            return False

        section_models = all_models[section]

        if ignore_case:
            search_name = model_name.lower()
            available_names = [name.lower() for name in section_models.keys()]
        else:
            search_name = model_name
            available_names = section_models.keys()

        if search_name in available_names:
            return True
        else:
            if stop_on_error:
                raise ValueError(
                    f"Unknown model '{model_name}' in section '{section}'.\n"
                    f"Valid choices are: {list(section_models.keys())}"
                )
            return False

    def validate_link_function(self, model_name: str, link: str) -> str:
        """
        Validates a link function for a given likelihood model.
        Resolves 'default' to the actual default link.
        """
        props = self.get_model_properties(model_name, "likelihood")
        valid_links = props.get('link', [])

        if not valid_links or len(valid_links) < 2 or valid_links[0] != "default":
            raise ValueError(f"Model '{model_name}' has an invalid link definition.")

        link_lower = link.lower()
        if link_lower in valid_links:
            if link_lower == "default":
                return valid_links[1]
            return link_lower
        else:
            raise ValueError(
                f"Link function '{link}' is not valid for likelihood '{model_name}'.\n"
                f"Valid links are: {valid_links}"
            )

    def validate_link_simple_function(self, model_name: str, link: str) -> str | None:
        """
        Validates a 'simple' link function for zero-inflated models.
        """
        props = self.get_model_properties(model_name, "likelihood")
        valid_links = props.get('link_simple')

        if valid_links is None:
            return None

        link_lower = link.lower()
        if link_lower in valid_links:
            if link_lower == "default":
                return valid_links[1]
            return link_lower
        else:
            raise ValueError(
                f"Simple link function '{link}' is not valid for '{model_name}'.\n"
                f"Valid simple links are: {valid_links}"
            )

    # ========================================================================
    # Helper Methods
    # ========================================================================

    @staticmethod
    def _is_scalar(x: Any) -> bool:
        if isinstance(x, (str, bytes)):
            return True
        return isinstance(x, numbers.Number) or isinstance(x, bool)

    @staticmethod
    def _to_list(x: Any) -> Optional[List[Any]]:
        if x is None:
            return None
        if isinstance(x, (list, tuple)):
            return list(x)
        return [x]

    @staticmethod
    def _flat_list(x: Any) -> Optional[List[Any]]:
        """Flatten nested lists/tuples into a flat list; keep strings as scalars."""
        if x is None:
            return None
        out: List[Any] = []
        def rec(v):
            if isinstance(v, (list, tuple)):
                for u in v:
                    rec(u)
            else:
                out.append(v)
        rec(x)
        return out

    @staticmethod
    def _theta_key_for_index(hyper_new: Dict[str, Dict[str, Any]], i: int) -> str:
        """Match R logic: prefer 'theta{i}', unless i==1 and only 'theta' exists."""
        key_num = f"theta{i}"
        if key_num in hyper_new:
            return key_num
        if i == 1 and "theta" in hyper_new:
            return "theta"
        keys = sorted(hyper_new.keys())
        return keys[i - 1]

    @staticmethod
    def _find_h_override_idx(hyper_override: Optional[Dict[str, Any]],
                             ih_name: str,
                             long_name: str,
                             short_name: str,
                             is_first: bool) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Find the matching key in the user `hyper` override for the i-th hyperparameter.
        """
        if not hyper_override:
            return None
        candidates = []
        if ih_name != "theta":
            candidates.append(ih_name)
        if is_first:
            candidates.append("theta")
            candidates.append("theta1")
        candidates.extend([long_name, short_name])

        lower_map = {str(k).lower(): k for k in hyper_override.keys()}

        for cand in candidates:
            if cand is None:
                continue
            lc = str(cand).lower()
            if lc in lower_map:
                k = lower_map[lc]
                v = hyper_override[k]
                if isinstance(v, dict):
                    return (k, v)
                else:
                    raise TypeError(
                        f"Argument `hyper['{k}']` must be a dict with keys among "
                        f"{{'initial','fixed','prior','param'}}; got {type(v)}."
                    )
        return None

    @staticmethod
    def _get_prior_nparams(self_obj, prior_value: Any) -> int:
        """
        Return expected 'param' length for a given prior.
        """
        try:
            from .rprior import RPrior, inla_is_rprior
        except Exception:
            def inla_is_rprior(x):
                return hasattr(x, "code") and getattr(x, "code", None) == "c5c4fee74dc9299b6753b8605e303f59a1236bfa"
            class RPrior:
                pass

        if inla_is_rprior(prior_value):
            return -1

        if isinstance(prior_value, str):
            prior_props = self_obj.get_model_properties(prior_value, "prior", stop_on_error=False)
            if prior_props is None:
                raise ValueError(f"Unknown prior '{prior_value}'. Is it registered in the 'prior' section?")
            return int(prior_props.get("nparameters", 0))

        raise TypeError(f"Unsupported prior spec type: {type(prior_value)}")

    # ========================================================================
    # set_hyper - Main hyperparameter setting method
    # ========================================================================

    def set_hyper(self,
                model: str,
                section: str,
                hyper: Optional[Dict[str, Dict[str, Any]]] = None,
                initial: Optional[Union[List[Any], Any]] = None,
                fixed: Optional[Union[List[Any], Any]] = None,
                prior: Optional[Union[List[Any], Any]] = None,
                param: Optional[Union[List[Any], Any]] = None,
                debug: bool = False,
                hyper_default: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Python conversion of R's `inla.set.hyper()`.

        Parameters
        ----------
        model, section : str
            Identify the model whose default hyper-structure should be fetched.
        hyper : dict or None
            Per-theta overrides, keyed by 'theta', 'theta1'.., or by 'name'/'short_name'.
        initial, fixed, prior, param :
            Top-level vectorised overrides.
        debug : bool
            If True, prints tracing messages.
        hyper_default : dict or None
            If provided, use this dict as the default hyper structure.

        Returns
        -------
        dict
            The merged hyper-structure (deep copy).
        """
        if section is None:
            raise ValueError("No section given; please fix...")

        # 1) Get defaults
        if hyper_default is None:
            props = self.get_model_properties(model, section)
            hyper_new = copy.deepcopy(props.get("hyper", {}))
        else:
            hyper_new = self.set_hyper(model=model, section=section,
                                    hyper=hyper_default, debug=debug)

        nhyper = len(hyper_new)
        if debug:
            print(f"* Get default hyper from model {model} in section {section}")

        if nhyper == 0:
            for nm, arg in [("hyper", hyper), ("initial", initial), ("fixed", fixed),
                            ("param", param), ("prior", prior)]:
                if arg is not None and (arg != [] and arg != {}):
                    raise ValueError(
                        f"Model {model} [{section}] has no hyperparameters, "
                        f"but '{nm}' was provided: {arg}"
                    )
            return hyper_new

        # Validate keys in `hyper`
        if hyper is not None:
            valid_keywords = ["theta"] + [f"theta{i}" for i in range(1, nhyper + 1)]
            valid_keywords += [str(h["short_name"]) for h in hyper_new.values()]
            valid_keywords += [str(h["name"]) for h in hyper_new.values()]
            for nm in hyper.keys():
                if nm is None:
                    raise ValueError("Missing name/keyword in `hyper`; "
                                    f"must be one of {valid_keywords}.")
                nm_lc = str(nm).lower()
                if not any(nm_lc == k.lower() for k in valid_keywords):
                    raise ValueError(
                        f"Unknown keyword in `hyper` '{nm}'. Must be one of {valid_keywords}."
                    )

        # Offsets for top-level vectorised args
        OFF_KEYS = ("initial", "fixed", "prior", "param")
        off = {k: 0 for k in OFF_KEYS}

        # Prepare top-level sequences
        top_initial = self._flat_list(initial)
        top_fixed   = self._flat_list(fixed)
        top_prior   = self._flat_list(prior)

        if isinstance(param, (list, tuple)) and any(isinstance(p, (list, tuple)) for p in param):
            top_param_nested = list(param)
            top_param_flat = None
        else:
            top_param_nested = None
            top_param_flat = self._flat_list(param)

        skip_final_check = False

        # Iterate over each hyperparameter
        for ih in range(1, nhyper + 1):
            ih_key = self._theta_key_for_index(hyper_new, ih)
            ih_slot = hyper_new[ih_key]

            name_long = ih_slot.get("name", f"theta{ih}")
            name_short = ih_slot.get("short_name", f"theta{ih}")
            if debug:
                print(f"** Check hyperparameter {ih} with key={ih_key}, "
                    f"name={name_long}, short_name={name_short}")

            h_pair = self._find_h_override_idx(
                hyper_override=hyper,
                ih_name=ih_key,
                long_name=name_long,
                short_name=name_short,
                is_first=(ih == 1)
            )
            h = h_pair[1] if h_pair else None

            for key in OFF_KEYS:
                if key != "param":
                    top_values = {"initial": top_initial, "fixed": top_fixed, "prior": top_prior}[key]
                    if top_values is not None:
                        if off[key] < len(top_values):
                            val = top_values[off[key]]
                            off[key] += 1
                            if val is not None:
                                ih_slot[key] = val
                                if debug:
                                    print(f"*** top-level: set {ih_key}.{key} <- {val}")
                else:
                    if top_param_nested is not None:
                        if ih - 1 < len(top_param_nested):
                            pval = top_param_nested[ih - 1]
                            if pval is not None:
                                ih_slot["param"] = list(pval)
                                if debug:
                                    print(f"*** top-level: set {ih_key}.param <- {pval}")
                    elif top_param_flat is not None:
                        npar = self._get_prior_nparams(self, ih_slot.get("prior"))
                        if npar < 0:
                            if off["param"] < len(top_param_flat):
                                ih_slot["param"] = top_param_flat[off["param"]:]
                                off["param"] = len(top_param_flat)
                                skip_final_check = True
                                if debug:
                                    print(f"*** top-level: set {ih_key}.param (variable length) <- {ih_slot['param']}")
                        else:
                            need = npar
                            have = len(top_param_flat) - off["param"]
                            if need > 0:
                                take = min(need, max(0, have))
                                if take > 0:
                                    chunk = top_param_flat[off["param"]: off["param"] + take]
                                    off["param"] += take
                                    if len(chunk) < need:
                                        chunk += [None] * (need - len(chunk))
                                    ih_slot["param"] = [
                                        (chunk[j] if chunk[j] is not None else ih_slot.get("param", [None] * need)[j])
                                        for j in range(need)
                                    ]
                                    if debug:
                                        print(f"*** top-level: set {ih_key}.param <- {ih_slot['param']} (need={need})")

                if h is not None and key in h:
                    v = h[key]
                    if key == "param":
                        npar = self._get_prior_nparams(self, ih_slot.get("prior"))
                        if npar < 0:
                            skip_final_check = True
                            ih_slot["param"] = list(v)
                            if debug:
                                print(f"*** hyper: set {ih_key}.param (variable length) <- {v}")
                        else:
                            if not isinstance(v, (list, tuple)):
                                raise TypeError(f"{ih_key}.param must be a sequence of length {npar}")
                            if len(v) != npar:
                                raise ValueError(
                                    f"Wrong length of prior-parameters for {ih_key}; "
                                    f"prior '{ih_slot.get('prior')}' needs {npar} parameters, you gave {len(v)}."
                                )
                            ih_slot["param"] = list(v)
                            if debug:
                                print(f"*** hyper: set {ih_key}.param <- {v}")
                    else:
                        ih_slot[key] = v
                        if debug:
                            print(f"*** hyper: set {ih_key}.{key} <- {v}")

                if key == "param":
                    prior_val = ih_slot.get("prior")
                    try:
                        npar = self._get_prior_nparams(self, prior_val)
                    except Exception:
                        npar = 0

                    if npar >= 0:
                        cur_param = ih_slot.get("param", [])
                        # Normalize scalar param to list for length check
                        if not isinstance(cur_param, (list, tuple)):
                            cur_param = [cur_param] if cur_param is not None else []
                        if len(cur_param) != npar:
                            raise ValueError(
                                f"Wrong length of prior-parameters, prior '{prior_val}' needs "
                                f"{npar} parameters, you have {len(cur_param)}."
                            )

        # Final check
        for key in ("initial", "fixed", "prior"):
            seq = {"initial": top_initial, "fixed": top_fixed, "prior": top_prior}[key]
            if seq is not None and off[key] < len(seq):
                raise ValueError(
                    f"Length of argument '{key}' is {len(seq)}, does not match the "
                    f"expected length {off[key]} consumed across hyperparameters."
                )

        if top_param_flat is not None and not skip_final_check:
            if off["param"] < len(top_param_flat):
                raise ValueError(
                    f"Length of argument 'param' is {len(top_param_flat)}, does not match the "
                    f"total length {off['param']} expected from the priors."
                )

        return hyper_new

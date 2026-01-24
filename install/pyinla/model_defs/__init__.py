"""
INLA model definitions package.

This package contains all model definitions organized into separate modules:

- latent: Latent model definitions (random effects)
- likelihood: Likelihood model definitions (observation distributions)
- other_sections: Other sections (group, scopy, mix, link, predictor, hazard, prior, wrapper, lp_scale)
- transforms: Common hyperparameter transformation functions
"""

from .latent import (
    LATENT_MODELS,
    get_latent_models,
    copy_clinear_to_theta,
    copy_clinear_from_theta,
    _SPECIAL_NUMBER,
)

from .likelihood import (
    LIKELIHOOD_MODELS,
    get_likelihood_models,
)

from .other_sections import (
    GROUP_MODELS,
    SCOPY_MODELS,
    MIX_MODELS,
    LINK_MODELS,
    PREDICTOR_MODELS,
    HAZARD_MODELS,
    PRIOR_MODELS,
    WRAPPER_MODELS,
    LP_SCALE_MODELS,
    get_group_models,
    get_scopy_models,
    get_mix_models,
    get_link_models,
    get_predictor_models,
    get_hazard_models,
    get_prior_models,
    get_wrapper_models,
    get_lp_scale_models,
)

from .transforms import (
    identity,
    log_transform,
    exp_transform,
    logit_transform,
    inv_logit_transform,
    get_transform,
    TRANSFORMS,
)

__all__ = [
    # Latent models
    'LATENT_MODELS',
    'get_latent_models',
    'copy_clinear_to_theta',
    'copy_clinear_from_theta',
    '_SPECIAL_NUMBER',
    # Likelihood models
    'LIKELIHOOD_MODELS',
    'get_likelihood_models',
    # Other sections
    'GROUP_MODELS',
    'SCOPY_MODELS',
    'MIX_MODELS',
    'LINK_MODELS',
    'PREDICTOR_MODELS',
    'HAZARD_MODELS',
    'PRIOR_MODELS',
    'WRAPPER_MODELS',
    'LP_SCALE_MODELS',
    'get_group_models',
    'get_scopy_models',
    'get_mix_models',
    'get_link_models',
    'get_predictor_models',
    'get_hazard_models',
    'get_prior_models',
    'get_wrapper_models',
    'get_lp_scale_models',
    # Transforms
    'identity',
    'log_transform',
    'exp_transform',
    'logit_transform',
    'inv_logit_transform',
    'get_transform',
    'TRANSFORMS',
]

"""
Other model sections for INLA.

This module contains the smaller model sections:
- group: Group correlation models
- scopy: Scopy models  
- mix: Mixture models
- link: Link functions
- predictor: Predictor specification
- hazard: Hazard models for survival
- prior: Prior distributions
- wrapper: Wrapper models
- lp_scale: Linear predictor scaling
"""

import numpy as np

# ============================================================================
# GROUP SECTION - Correlation models for grouped effects
# ============================================================================

# Helper functions for group section
_EPS = 1e-15

def _logit(p):
    """Logit transform with bounds protection."""
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))

def _expit(z):
    """Inverse logit transform."""
    ez = np.exp(z)
    return ez / (1.0 + ez)

def _cor_to_theta(r):
    """Transform correlation to internal scale."""
    r = np.clip(r, -1.0 + _EPS, 1.0 - _EPS)
    return np.log((1.0 + r) / (1.0 - r))

def _theta_to_cor(z):
    """Transform internal scale to correlation."""
    ez = np.exp(z)
    return 2.0 * ez / (1.0 + ez) - 1.0


GROUP_MODELS = {
    'exchangeable': {
        'doc': "Exchangeable correlations",
        'hyper': {
            'theta': {
                'hyperid': 40001,
                'name': "logit correlation",
                'short_name': "rho",
                'initial': 1,
                'fixed': False,
                'prior': "normal",
                'param': [0, 0.2],
                'to_theta': (lambda x, ngroup:
                    (lambda r: np.log((1.0 + r * (ngroup - 1.0)) / (1.0 - r)))(
                        np.clip(x, -1.0 / (ngroup - 1.0) + _EPS, 1.0 - _EPS)
                    )
                ),
                'from_theta': (lambda z, ngroup:
                    (np.exp(z) - 1.0) / (np.exp(z) + ngroup - 1.0)
                )
            }
        }
    },
    'exchangeablepos': {
        'doc': "Exchangeable positive correlations",
        'hyper': {
            'theta': {
                'hyperid': 40101,
                'name': "logit correlation",
                'short_name': "rho",
                'initial': 1,
                'fixed': False,
                'prior': "pc.cor0",
                'param': [0.5, 0.5],
                'to_theta': lambda x: _logit(x),
                'from_theta': lambda z: _expit(z)
            }
        }
    },
    'ar1': {
        'doc': "AR(1) correlations",
        'hyper': {
            'theta': {
                'hyperid': 41001,
                'name': "logit correlation",
                'short_name': "rho",
                'initial': 2,
                'fixed': False,
                'prior': "normal",
                'param': [0, 0.15],
                'to_theta': lambda x: _cor_to_theta(x),
                'from_theta': lambda z: _theta_to_cor(z)
            }
        }
    },
    'ar': {
        'doc': "AR(p) correlations",
        'hyper': {
            'theta1': {
                'hyperid': 42001,
                'name': "log precision",
                'short_name': "prec",
                'initial': 0,
                'fixed': True,
                'prior': "pc.prec",
                'param': [3, 0.01],
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda z: np.exp(z)
            },
            **{
                f'theta{2+i}': {
                    'hyperid': 42002 + i,
                    'name': f"pacf{i+1}",
                    'short_name': f"pacf{i+1}",
                    'initial': 2 if i == 0 else 0,
                    'fixed': False,
                    'prior': "pc.cor0",
                    'param': [0.5, [0.5, 0.4, 0.3, 0.2][i]] if i < 4 else [0.5, 0.1],
                    'to_theta': lambda x: _cor_to_theta(x),
                    'from_theta': lambda z: _theta_to_cor(z)
                } for i in range(10)
            }
        }
    },
    'rw1': {
        'doc': "Random walk of order 1",
        'hyper': {
            'theta': {
                'hyperid': 43001,
                'name': "log precision",
                'short_name': "prec",
                'prior': "loggamma",
                'param': [1, 0.00005],
                'initial': 0,
                'fixed': True,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda z: np.exp(z)
            }
        }
    },
    'rw2': {
        'doc': "Random walk of order 2",
        'hyper': {
            'theta': {
                'hyperid': 44001,
                'name': "log precision",
                'short_name': "prec",
                'prior': "loggamma",
                'param': [1, 0.00005],
                'initial': 0,
                'fixed': True,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda z: np.exp(z)
            }
        }
    },
    'besag': {
        'doc': "Besag model",
        'hyper': {
            'theta': {
                'hyperid': 45001,
                'name': "log precision",
                'short_name': "prec",
                'prior': "loggamma",
                'param': [1, 0.00005],
                'initial': 0,
                'fixed': True,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda z: np.exp(z)
            }
        }
    },
    'iid': {
        'doc': "Independent model",
        'hyper': {
            'theta': {
                'hyperid': 46001,
                'name': "log precision",
                'short_name': "prec",
                'prior': "loggamma",
                'param': [1, 0.00005],
                'initial': 0,
                'fixed': True,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda z: np.exp(z)
            }
        }
    }
}


# ============================================================================
# SCOPY SECTION
# ============================================================================

SCOPY_MODELS = {
    'rw1': {'doc': "Random walk of order 1", 'hyper': {}},
    'rw2': {'doc': "Random walk of order 2", 'hyper': {}}
}


# ============================================================================
# MIX SECTION - Mixture models
# ============================================================================

MIX_MODELS = {
    'gaussian': {
        'doc': "Gaussian mixture",
        'hyper': {
            'theta': {
                'hyperid': 47001,
                'name': "log precision",
                'short_name': "prec",
                'output_name': "Precision for the Gaussian observations",
                'output_name_intern': "Log precision for the Gaussian observations",
                'prior': "pc.prec",
                'param': [1, 0.01],
                'initial': 0,
                'fixed': False,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            }
        }
    },
    'loggamma': {
        'doc': "LogGamma mixture",
        'hyper': {
            'theta': {
                'hyperid': 47101,
                'name': "log precision",
                'short_name': "prec",
                'prior': "pcmgamma",
                'param': [4.8],
                'initial': 4,
                'fixed': False,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            }
        }
    },
    'mloggamma': {
        'doc': "Minus-LogGamma mixture",
        'hyper': {
            'theta': {
                'hyperid': 47201,
                'name': "log precision",
                'short_name': "prec",
                'prior': "pcmgamma",
                'param': [4.8],
                'initial': 4,
                'fixed': False,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            }
        }
    }
}


# ============================================================================
# LINK SECTION - Link functions
# ============================================================================

# Helper for GEVIT transforms
def _link_gevit_to(x, interval):
    return np.log(-(interval[0] - x) / (interval[1] - x))

def _link_gevit_from(x, interval):
    return interval[0] + (interval[1] - interval[0]) * np.exp(x) / (1.0 + np.exp(x))


LINK_MODELS = {
    'default': {'doc': "The default link", 'hyper': {}},
    'cloglog': {'doc': "The complementary log-log link", 'hyper': {}},
    'ccloglog': {'doc': "The complement complementary log-log link", 'hyper': {}},
    'loglog': {'doc': "The log-log link", 'hyper': {}},
    'identity': {'doc': "The identity link", 'hyper': {}},
    'inverse': {'doc': "The inverse link", 'hyper': {}},
    'log': {'doc': "The log-link", 'hyper': {}},
    'loga': {'doc': "The loga-link", 'hyper': {}},
    'neglog': {'doc': "The negative log-link", 'hyper': {}},
    'logit': {'doc': "The logit-link", 'hyper': {}},
    'probit': {'doc': "The probit-link", 'hyper': {}},
    'cauchit': {'doc': "The cauchit-link", 'hyper': {}},
    'tan': {'doc': "The tan-link", 'hyper': {}, 'pdf': "circular"},
    'tanpi': {'doc': "The tanpi-link", 'hyper': {}, 'pdf': "circular"},
    'quantile': {'doc': "The quantile-link", 'hyper': {}},
    'pquantile': {'doc': "The population quantile-link", 'hyper': {}},
    'sslogit': {
        'doc': "Logit link with sensitivity and specificity",
        'hyper': {
            'theta1': {'hyperid': 48001, 'name': "sensitivity", 'short_name': "sens", 'prior': "logitbeta", 'param': [10, 5], 'initial': 1, 'fixed': False, 'to_theta': lambda x: np.log(x / (1 - x)), 'from_theta': lambda x: np.exp(x) / (1 + np.exp(x))},
            'theta2': {'hyperid': 48002, 'name': "specificity", 'short_name': "spec", 'prior': "logitbeta", 'param': [10, 5], 'initial': 1, 'fixed': False, 'to_theta': lambda x: np.log(x / (1 - x)), 'from_theta': lambda x: np.exp(x) / (1 + np.exp(x))}
        },
        'status': "disabled", 'pdf': None
    },
    'logoffset': {'doc': "Log-link with an offset", 'hyper': {'theta': {'hyperid': 49001, 'name': "beta", 'short_name': "b", 'prior': "normal", 'param': [0, 100], 'initial': 0, 'fixed': True, 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)}}, 'pdf': "logoffset"},
    'logitoffset': {'doc': "Logit-link with an offset", 'hyper': {'theta': {'hyperid': 49011, 'name': "prob", 'short_name': "p", 'prior': "normal", 'param': [-1, 100], 'initial': -1, 'fixed': False, 'to_theta': lambda x: np.log(x / (1 - x)), 'from_theta': lambda x: np.exp(x) / (1 + np.exp(x))}}, 'pdf': "logitoffset"},
    'robit': {'doc': "Robit link", 'hyper': {'theta': {'hyperid': 49021, 'name': "log degrees of freedom", 'short_name': "dof", 'initial': np.log(5), 'fixed': True, 'prior': "pc.dof", 'param': [50, 0.5], 'to_theta': lambda x: np.log(x - 2), 'from_theta': lambda x: 2 + np.exp(x)}}, 'pdf': "robit"},
    'sn': {
        'doc': "Skew-normal link",
        'hyper': {
            'theta1': {'hyperid': 49031, 'name': "skewness", 'short_name': "skew", 'initial': 0.00123456789, 'fixed': False, 'prior': "pc.sn", 'param': [10], 'to_theta': lambda x, skew_max=0.988: np.log((1 + x / skew_max) / (1 - x / skew_max)), 'from_theta': lambda x, skew_max=0.988: skew_max * (2 * np.exp(x) / (1 + np.exp(x)) - 1)},
            'theta2': {'hyperid': 49032, 'name': "intercept", 'short_name': "p0", 'initial': 0.0, 'fixed': False, 'prior': "linksnintercept", 'param': [0, 0], 'to_theta': lambda x: np.log(x / (1 - x)), 'from_theta': lambda x: np.exp(x) / (1 + np.exp(x))}
        },
        'pdf': "linksn"
    },
    'gevit': {
        'doc': "GEVIT link",
        'hyper': {
            'theta1': {'hyperid': 49033, 'name': "gev tail", 'short_name': "tail", 'initial': 0.1, 'fixed': False, 'prior': "pc.egptail", 'param': [5, -0.5, 0.5], 'to_theta': _link_gevit_to, 'from_theta': _link_gevit_from},
            'theta2': {'hyperid': 49034, 'name': "gev p0", 'short_name': "p0", 'initial': 0.0, 'fixed': False, 'prior': "normal", 'param': [0, 1], 'to_theta': lambda x: np.log(x / (1 - x)), 'from_theta': lambda x: 1 / (1 + np.exp(-x))}
        },
        'pdf': "gevit"
    },
    'cgevit': {
        'doc': "Complement GEVIT link",
        'hyper': {
            'theta1': {'hyperid': 49035, 'name': "gev tail", 'short_name': "tail", 'initial': -3, 'fixed': False, 'prior': "pc.gevtail", 'param': [7, 0.0, 0.5], 'to_theta': _link_gevit_to, 'from_theta': _link_gevit_from},
            'theta2': {'hyperid': 49036, 'name': "gev p0", 'short_name': "p0", 'initial': 0.0, 'fixed': False, 'prior': "normal", 'param': [0, 1], 'to_theta': lambda x: np.log(x / (1 - x)), 'from_theta': lambda x: 1 / (1 + np.exp(-x))}
        },
        'pdf': "gevit"
    },
    'powerlogit': {
        'doc': "Power logit link",
        'hyper': {
            'theta1': {'hyperid': 49131, 'name': "power", 'short_name': "power", 'initial': 0.00123456789, 'fixed': False, 'prior': "normal", 'param': [0, 10], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)},
            'theta2': {'hyperid': 49132, 'name': "intercept", 'short_name': "p0", 'initial': 0.0, 'fixed': False, 'prior': "logitbeta", 'param': [1, 1], 'to_theta': lambda x: np.log(x / (1 - x)), 'from_theta': lambda x: np.exp(x) / (1 + np.exp(x))}
        },
        'pdf': "powerlogit"
    }
}


# ============================================================================
# PREDICTOR SECTION
# ============================================================================

PREDICTOR_MODELS = {
    'predictor': {
        'doc': "(do not use)",
        'hyper': {
            'theta': {
                'hyperid': 53001,
                'name': "log precision",
                'short_name': "prec",
                'initial': np.log(1 / 0.001**2),
                'fixed': True,
                'prior': "loggamma",
                'param': [1, 0.00001],
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            }
        }
    }
}


# ============================================================================
# HAZARD SECTION - Hazard models for survival analysis
# ============================================================================

HAZARD_MODELS = {
    'rw1': {
        'doc': "A random walk of order 1 for the log-hazard",
        'hyper': {
            'theta': {
                'hyperid': 54001, 'name': "log precision", 'short_name': "prec", 'initial': 4, 'fixed': False,
                'prior': "loggamma", 'param': [1, 0.00005], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            }
        }
    },
    'rw2': {
        'doc': "A random walk of order 2 for the log-hazard",
        'hyper': {
            'theta': {
                'hyperid': 55001, 'name': "log precision", 'short_name': "prec", 'initial': 4, 'fixed': False,
                'prior': "loggamma", 'param': [1, 0.00005], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            }
        }
    },
    'iid': {
        'doc': "An iid model for the log-hazard",
        'hyper': {
            'theta': {
                'hyperid': 55501, 'name': "log precision", 'short_name': "prec", 'initial': 4, 'fixed': False,
                'prior': "loggamma", 'param': [1, 0.00005], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            }
        }
    }
}


# ============================================================================
# PRIOR SECTION - Prior distributions for hyperparameters
# ============================================================================

PRIOR_MODELS = {
    'normal': {'doc': "Normal prior", 'nparameters': 2, 'pdf': "gaussian"},
    'gaussian': {'doc': "Gaussian prior", 'nparameters': 2, 'pdf': "gaussian"},
    'laplace': {'doc': "Laplace prior", 'nparameters': 2, 'pdf': "laplace"},
    'linksnintercept': {'doc': "Skew-normal-link intercept-prior", 'nparameters': 2, 'pdf': "gaussian"},
    'wishart1d': {'doc': "Wishart prior dim=1", 'nparameters': 2, 'pdf': "iid123d"},
    'wishart2d': {'doc': "Wishart prior dim=2", 'nparameters': 4, 'pdf': "iid123d"},
    'wishart3d': {'doc': "Wishart prior dim=3", 'nparameters': 7, 'pdf': "iid123d"},
    'wishart4d': {'doc': "Wishart prior dim=4", 'nparameters': 11, 'pdf': "iid123d"},
    'wishart5d': {'doc': "Wishart prior dim=5", 'nparameters': 16, 'pdf': "iid123d"},
    'loggamma': {'doc': "Log-Gamma prior", 'nparameters': 2, 'pdf': "prior-loggamma"},
    'gamma': {'doc': "Gamma prior", 'nparameters': 2, 'pdf': "prior-loggamma"},
    'minuslogsqrtruncnormal': {'doc': "(obsolete)", 'nparameters': 2, 'pdf': "prior-logtnorm"},
    'logtnormal': {'doc': "Truncated Normal prior", 'nparameters': 2, 'pdf': "prior-logtnorm"},
    'logtgaussian': {'doc': "Truncated Gaussian prior", 'nparameters': 2, 'pdf': "prior-logtnorm"},
    'flat': {'doc': "A constant prior", 'nparameters': 0, 'pdf': "various-flat"},
    'logflat': {'doc': "A constant prior for log(theta)", 'nparameters': 0, 'pdf': "various-flat"},
    'logiflat': {'doc': "A constant prior for log(1/theta)", 'nparameters': 0, 'pdf': "various-flat"},
    'mvnorm': {'doc': "A multivariate Normal prior", 'nparameters': -1, 'pdf': "mvnorm"},
    'pc.alphaw': {'doc': "PC prior for alpha in Weibull", 'nparameters': 1, 'pdf': "pc.alphaw"},
    'pc.ar': {'doc': "PC prior for the AR(p) model", 'nparameters': 1, 'pdf': "pc.ar"},
    'dirichlet': {'doc': "Dirichlet prior", 'nparameters': 1, 'pdf': "dirichlet"},
    'none': {'doc': "No prior", 'nparameters': 0},
    'invalid': {'doc': "Void prior", 'nparameters': 0},
    'betacorrelation': {'doc': "Beta prior for the correlation", 'nparameters': 2, 'pdf': "betacorrelation"},
    'logitbeta': {'doc': "Logit prior for a probability", 'nparameters': 2, 'pdf': "logitbeta"},
    'pc.prec': {'doc': "PC prior for log(precision)", 'nparameters': 2, 'pdf': "pc.prec"},
    'pcprec': {'doc': "PC prior for log(precision)", 'nparameters': 2, 'pdf': "pc.prec"},
    'pc.dof': {'doc': "PC prior for log(dof-2)", 'nparameters': 2, 'pdf': "pc.dof"},
    'pcdof': {'doc': "PC prior for log(dof-2)", 'nparameters': 2, 'pdf': "pcdof"},
    'pc.cor0': {'doc': "PC prior correlation, basemodel cor=0", 'nparameters': 2, 'pdf': "pc.cor0"},
    'pc.cor1': {'doc': "PC prior correlation, basemodel cor=1", 'nparameters': 2, 'pdf': "pc.cor1"},
    'pc.fgnh': {'doc': "PC prior for the Hurst parameter in FGN", 'nparameters': 2, 'pdf': "pc.fgnh"},
    'pcfgnh': {'doc': "PC prior for the Hurst parameter in FGN", 'nparameters': 2, 'pdf': "pc.fgnh"},
    'pc.spde.GA': {'doc': "(experimental)", 'nparameters': 4, 'pdf': None},
    'pc.matern': {'doc': "PC prior for the Matern SPDE", 'nparameters': 3, 'pdf': None},
    'pcmatern': {'doc': "PC prior for the Matern SPDE", 'nparameters': 3, 'pdf': None},
    'pc.range': {'doc': "PC prior for the range in the Matern SPDE", 'nparameters': 2, 'pdf': None},
    'pc.sn': {'doc': "PC prior for the skew-normal", 'nparameters': 1, 'pdf': "pc.sn"},
    'pc.gamma': {'doc': "PC prior for a Gamma parameter", 'nparameters': 1, 'pdf': "pc.gamma"},
    'pc.mgamma': {'doc': "PC prior for a Gamma parameter", 'nparameters': 1, 'pdf': "pc.gamma"},
    'pcmgamma': {'doc': "PC prior for a Gamma parameter", 'nparameters': 1, 'pdf': "pc.gamma"},
    'pc.gammacount': {'doc': "PC prior for the GammaCount likelihood", 'nparameters': 1, 'pdf': "pc.gammacount"},
    'pc.gevtail': {'doc': "PC prior for the tail in the GEV likelihood", 'nparameters': 3, 'pdf': "pc.gevtail"},
    'pc.egptail': {'doc': "PC prior for the tail in the EGP likelihood", 'nparameters': 3, 'pdf': "pc.egptail"},
    'pc': {'doc': "Generic PC prior", 'nparameters': 2, 'pdf': None},
    'ref.ar': {'doc': "Reference prior for the AR(p) model, p<=3", 'nparameters': 0, 'pdf': None},
    'pom': {'doc': "#classes-dependent prior for the POM model", 'nparameters': 0, 'pdf': "pom"},
    'jeffreystdf': {'doc': "Jeffreys prior for the doc", 'nparameters': 0, 'pdf': "jeffreystdf"},
    'wishartkd': {'doc': "Wishart prior", 'nparameters': 301, 'pdf': None},
    'expression:': {'doc': "A generic prior defined using expressions", 'nparameters': -1, 'pdf': "expression"},
    'table:': {'doc': "A generic tabulated prior", 'nparameters': -1, 'pdf': "table"},
    'rprior:': {'doc': "A R-function defining the prior", 'status': "experimental", 'nparameters': 0, 'pdf': "rprior"}
}


# ============================================================================
# WRAPPER SECTION
# ============================================================================

WRAPPER_MODELS = {
    'joint': {
        'doc': "(experimental)",
        'hyper': {
            'theta': {
                'hyperid': 102001, 'name': "log precision", 'short_name': "prec", 'output_name': "NOT IN USE",
                'output_name_intern': "NOT IN USE", 'initial': 0, 'fixed': True, 'prior': "loggamma",
                'param': [1, 0.00005], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            }
        },
        'constr': False, 'nrow_ncol': False, 'augmented': False, 'aug_factor': 1, 'aug_constr': None,
        'n_div_by': None, 'n_required': False, 'set_default_values': False, 'pdf': None
    }
}


# ============================================================================
# LP_SCALE SECTION - Linear predictor scaling
# ============================================================================

LP_SCALE_MODELS = {
    'lp.scale': {
        'hyper': {
            f'theta{i}': {
                'hyperid': 103000 + i,
                'name': f"beta{i}",
                'short_name': f"b{i}",
                'output_name': f"beta[{i}] for lp_scale",
                'output_name_intern': f"beta[{i}] for lp_scale",
                'initial': 1,
                'fixed': False,
                'prior': "normal",
                'param': [1, 10],
                'to_theta': lambda x: x,
                'from_theta': lambda x: x
            } for i in range(1, 101)
        },
        'pdf': "lp.scale"
    }
}


# ============================================================================
# ACCESSOR FUNCTIONS
# ============================================================================

def get_group_models() -> dict:
    """Return all group correlation model definitions."""
    return GROUP_MODELS

def get_scopy_models() -> dict:
    """Return all scopy model definitions."""
    return SCOPY_MODELS

def get_mix_models() -> dict:
    """Return all mixture model definitions."""
    return MIX_MODELS

def get_link_models() -> dict:
    """Return all link function definitions."""
    return LINK_MODELS

def get_predictor_models() -> dict:
    """Return predictor model definitions."""
    return PREDICTOR_MODELS

def get_hazard_models() -> dict:
    """Return all hazard model definitions."""
    return HAZARD_MODELS

def get_prior_models() -> dict:
    """Return all prior distribution definitions."""
    return PRIOR_MODELS

def get_wrapper_models() -> dict:
    """Return all wrapper model definitions."""
    return WRAPPER_MODELS

def get_lp_scale_models() -> dict:
    """Return LP scale model definitions."""
    return LP_SCALE_MODELS

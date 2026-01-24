"""
Likelihood model definitions for INLA.

This module contains all available likelihood models (observation distributions)
that can be used in INLA model formulas.

Each model is defined as a dictionary with the following keys:
- doc: Description of the likelihood
- hyper: Hyperparameter specifications
- survival: Whether it's a survival model
- discrete: Whether it's for discrete data
- link: Available link functions
- status: Optional status (e.g., "experimental")
- pdf: Name of the PDF documentation
"""

import numpy as np

# Special number used as placeholder for unspecified hyperparameters
_SPECIAL_NUMBER = 1048576.0

# Helper transforms for generalized extreme value inverse transform
def _gevit_to(x, interval):
    """Transform to internal scale for gevit link."""
    return np.log(-(interval[0] - x) / (interval[1] - x))

def _gevit_from(x, interval):
    """Transform from internal scale for gevit link."""
    return interval[0] + (interval[1] - interval[0]) * np.exp(x) / (1.0 + np.exp(x))


def _gen_beta_hyper(start_id, n, name_base, initial1, param1, theta_offset=1):
    """Generate beta hyperparameters for cure models.

    Args:
        start_id: Starting hyperid
        n: Number of beta parameters to generate
        name_base: Name base for output names (e.g., 'Gamma-Cure')
        initial1: Initial value for beta1
        param1: Prior parameters for beta1
        theta_offset: Starting theta index (default 1, use 2 when theta1 is precision)
    """
    hyper = {
        f'theta{theta_offset}': {
    'hyperid': start_id, 'name': "beta1", 'short_name': "beta1", 'output_name': f"beta1 for {name_base}",
    'output_name_intern': f"beta1 for {name_base}", 'initial': initial1, 'fixed': False,
    'prior': "normal", 'param': param1, 'to_theta': lambda x: x, 'from_theta': lambda x: x
}
    }
    for i in range(2, n + 1):
        hyper[f'theta{theta_offset + i - 1}'] = {
    'hyperid': start_id + i - 1, 'name': f"beta{i}", 'short_name': f"beta{i}", 'output_name': f"beta{i} for {name_base}",
    'output_name_intern': f"beta{i} for {name_base}", 'initial': 0, 'fixed': False,
    'prior': "normal", 'param': [0, 100], 'to_theta': lambda x: x, 'from_theta': lambda x: x
}
    return hyper


LIKELIHOOD_MODELS = {
    'fl': {
        'doc': "The fl likelihood", 'hyper': {}, 'survival': False, 'discrete': True,
        'link': ["default", "identity"], 'status': "experimental", 'pdf': "fl"
    },
    'poisson': {
        'doc': "The Poisson likelihood", 'hyper': {}, 'survival': False, 'discrete': True,
        'link': ["default", "log", "logoffset", "quantile", "test1", "special1", "special2"], 'pdf': "poisson"
    },
    'npoisson': {
        'doc': "The Normal approximation to the Poisson likelihood", 'hyper': {}, 'survival': False, 'discrete': True,
        'link': ["default", "log", "logoffset"], 'pdf': "poisson"
    },
    'nzpoisson': {
        'doc': "The nzPoisson likelihood", 'hyper': {}, 'survival': False, 'discrete': True,
        'link': ["default", "log", "logoffset"], 'pdf': "nzpoisson"
    },
    'xpoisson': {
        'doc': "The Poisson likelihood (expert version)", 'hyper': {}, 'survival': False, 'discrete': True,
        'link': ["default", "log", "logoffset", "quantile", "test1", "special1", "special2"], 'pdf': "poisson"
    },
    'cenpoisson': {
        'doc': "Then censored Poisson likelihood", 'hyper': {}, 'survival': False, 'discrete': True,
        'link': ["default", "log", "logoffset", "test1", "special1", "special2"], 'pdf': "cenpoisson"
    },
    'cenpoisson2': {
        'doc': "Then censored Poisson likelihood (version 2)", 'hyper': {}, 'survival': False, 'discrete': True,
        'link': ["default", "log", "logoffset", "test1", "special1", "special2"], 'pdf': "cenpoisson2"
    },
    'gpoisson': {
        'doc': "The generalized Poisson likelihood",
        'hyper': {
            'theta1': {
                'hyperid': 56001, 'name': "overdispersion", 'short_name': "phi", 'output_name': "Overdispersion for gpoisson",
                'output_name_intern': "Log overdispersion for gpoisson", 'initial': 0, 'fixed': False, 'prior': "loggamma",
                'param': [1, 1], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 56002, 'name': "p", 'short_name': "p", 'output_name': "Parameter p for gpoisson",
                'output_name_intern': "Parameter p_intern for gpoisson", 'initial': 1, 'fixed': True, 'prior': "normal",
                'param': [1, 100], 'to_theta': lambda x: x, 'from_theta': lambda x: x
            }
        },
        'survival': False, 'discrete': True, 'link': ["default", "log", "logoffset"], 'pdf': "gpoisson"
    },
    'poisson.special1': {
        'doc': "The Poisson.special1 likelihood",
        'hyper': {
            'theta': {
                'hyperid': 56100, 'name': "logit probability", 'short_name': "prob",
                'output_name': "one-probability parameter for poisson.special1",
                'output_name_intern': "intern one-probability parameter for poisson.special1",
                'initial': -1, 'fixed': False, 'prior': "gaussian", 'param': [-1, 0.2],
                'to_theta': lambda x: np.log(x / (1 - x)), 'from_theta': lambda x: np.exp(x) / (1 + np.exp(x))
            }
        },
        'survival': False, 'discrete': True, 'link': ["default", "log"], 'pdf': "poisson-special"
    },
    '0poisson': {
        'doc': "New 0-inflated Poisson",
        'hyper': _gen_beta_hyper(56201, 10, '0poisson', -4, [-4, 10]),
        'survival': False, 'discrete': True,
        'link': ["default", "log", "quantile"],
        'link_simple': ["default", "logit", "cauchit", "probit", "cloglog", "ccloglog"],
        'pdf': "0inflated"
    },
    '0poissonS': {
        'doc': "New 0-inflated Poisson Swap",
        'hyper': _gen_beta_hyper(56301, 10, '0poissonS', -4, [-4, 10]),
        'survival': False, 'discrete': True,
        'link': ["default", "logit", "loga", "cauchit", "probit", "cloglog", "ccloglog", "loglog",
                "log", "sslogit", "logitoffset", "quantile", "pquantile", "robit", "sn",
                "powerlogit"],
        'link_simple': ["default", "log"],
        'pdf': "0inflated"
    },
    'bell': {
        'doc': "The Bell likelihood", 'hyper': {}, 'survival': False, 'discrete': True,
        'link': ["default", "log"], 'pdf': "bell"
    },
    '0binomial': {
        'doc': "New 0-inflated Binomial",
        'hyper': _gen_beta_hyper(56401, 10, '0binomial', -4, [-4, 10]),
        'survival': False, 'discrete': True,
        'link': ["default", "logit", "loga", "cauchit", "probit", "cloglog", "ccloglog", "loglog", "log"],
        'link_simple': ["default", "logit", "cauchit", "probit", "cloglog", "ccloglog"],
        'pdf': "0inflated"
    },
    '0binomialS': {
        'doc': "New 0-inflated Binomial Swap",
        'hyper': _gen_beta_hyper(56501, 10, '0binomialS', -4, [-4, 10]),
        'survival': False, 'discrete': True,
        'link': ["default", "logit", "loga", "cauchit", "probit", "cloglog",
                "ccloglog", "loglog", "log"],
        'link_simple': ["default", "logit", "cauchit", "probit", "cloglog", "ccloglog"],
        'pdf': "0inflated"
    },
    'binomialmix': {
        'doc': "Binomial mixture",
        'hyper': _gen_beta_hyper(56551, 51, 'binomialmix', 0, [0, 100]),
        'status': "experimental", 'survival': False, 'discrete': True,
        'link': ["default", "logit", "probit"],
        'pdf': "binomialmix"
    },
    'binomial': {
        'doc': "The Binomial likelihood", 'hyper': {}, 'survival': False, 'discrete': True,
        'link': ["default", "logit", "loga", "cauchit", "probit", "cloglog",
                "ccloglog", "loglog", "log", "sslogit", "logitoffset", "quantile",
                "pquantile", "robit", "sn", "powerlogit", "gevit", "cgevit"],
        'pdf': "binomial"
    },
    'xbinomial': {
        'doc': "The Binomial likelihood (experimental version)", 'hyper': {}, 'survival': False, 'discrete': True,
        'link': ["default", "logit", "loga", "cauchit", "probit", "cloglog", "ccloglog", "loglog",
                "log", "sslogit", "logitoffset", "quantile", "pquantile", "robit", "sn",
                "powerlogit", "gevit", "cgevit"],
        'pdf': "binomial"
    },
    'occupancy': {
        'doc': "Occupancy likelihood",
        'hyper': _gen_beta_hyper(56601, 10, 'occupancy', -2, [-2, 10]),
        'survival': False, 'discrete': True,
        'link': ["default", "logit", "cloglog"],
        'link_simple': ["default", "logit", "cloglog"],
        'pdf': "occupancy"
    },
    'pom': {
        'doc': "Likelihood for the proportional odds model",
        'hyper': {
            'theta1': {
                'hyperid': 57101, 'name': "theta1", 'short_name': "theta1", 'output_name': "theta1 for POM",
                'output_name_intern': "theta1 for POM", 'initial': None, 'fixed': False, 'prior': "dirichlet",
                'param': [3.0], 'to_theta': lambda x: x, 'from_theta': lambda x: x
            },
            **{f'theta{i}': {
                'hyperid': 57100 + i, 'name': f"theta{i}", 'short_name': f"theta{i}", 'output_name': f"theta{i} for POM",
                'output_name_intern': f"theta{i} for POM", 'initial': None, 'fixed': False, 'prior': "none",
                'param': [], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            } for i in range(2, 11)}
        },
        'survival': False, 'discrete': True, 'link': ["default", "identity"], 'pdf': "pom"
    },
    'bgev': {
        'doc': "The blended Generalized Extreme Value likelihood",
        'hyper': {
            'theta1': {
                'hyperid': 57201, 'name': "spread", 'short_name': "sd", 'output_name': "spread for BGEV observations",
                'output_name_intern': "log spread for BGEV observations", 'initial': 0, 'fixed': False, 'prior': "loggamma",
                'param': [1, 3], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 57202, 'name': "tail", 'short_name': "xi", 'output_name': "tail for BGEV observations",
                'output_name_intern': "intern tail for BGEV observations", 'initial': -4, 'fixed': False, 'prior': "pc.gevtail",
                'param': [7, 0.0, 0.5], 'to_theta': _gevit_to, 'from_theta': _gevit_from # Using the helper defined previously
            },
            **{f'theta{i+2}': {
                'hyperid': 57201 + i, 'name': f"beta{i}", 'short_name': f"beta{i}", 'output_name': "MUST BE FIXED",
                'output_name_intern': "MUST BE FIXED", 'initial': None, 'fixed': False, 'prior': "normal",
                'param': [0, 300], 'to_theta': lambda x: x, 'from_theta': lambda x: x
            } for i in range(1, 11)}
        },
        'survival': False, 'discrete': False, 'link': ["default", "identity", "log"], 'pdf': "bgev"
    },
    'gamma': {
        'doc': "The Gamma likelihood",
        'hyper': {
            'theta': {
                'hyperid': 58001, 'name': "precision parameter", 'short_name': "prec",
                'output_name': "Precision-parameter for the Gamma observations",
                'output_name_intern': "Intern precision-parameter for the Gamma observations",
                'initial': np.log(100), 'fixed': False, 'prior': "loggamma", 'param': [1, 0.01],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            }
        },
        'survival': False, 'discrete': False, 'link': ["default", "log", "quantile"], 'pdf': "gamma"
    },
    'mgamma': {
        'doc': "The modal Gamma likelihood",
        'hyper': {
            'theta': {
                'hyperid': 58002, 'name': "precision parameter", 'short_name': "prec",
                'output_name': "Precision-parameter for the modal Gamma observations",
                'output_name_intern': "Intern precision-parameter for the modal Gamma observations",
                'initial': np.log(100), 'fixed': False, 'prior': "loggamma", 'param': [1, 0.01],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            }
        },
        'survival': False, 'discrete': False, 'link': ["default", "log"], 'pdf': "mgamma"
    },
    'gammasurv': {
        'doc': "The Gamma likelihood (survival)",
        'hyper': {
            'theta1': {
                'hyperid': 58101, 'name': "precision parameter", 'short_name': "prec",
                'output_name': "Precision-parameter for the Gamma surv observations",
                'output_name_intern': "Intern precision-parameter for the Gamma surv observations",
                'initial': np.log(1), 'fixed': False, 'prior': "loggamma", 'param': [1, 0.01],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            **_gen_beta_hyper(58102, 10, 'Gamma-Cure', -7, [-4, 100], theta_offset=2)
        },
        'survival': True, 'discrete': False,
        'link': ["default", "log", "neglog", "quantile"], 'pdf': "gammasurv"
    },
    'mgammasurv': {
        'doc': "The modal Gamma likelihood (survival)",
        'hyper': {
            'theta1': {
                'hyperid': 58121, 'name': "precision parameter", 'short_name': "prec",
                'output_name': "Precision-parameter for the modal Gamma surv observations",
                'output_name_intern': "Intern precision-parameter for the modal Gamma surv observations",
                'initial': np.log(1), 'fixed': False, 'prior': "loggamma", 'param': [1, 0.01],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            **_gen_beta_hyper(58122, 10, 'modal Gamma-Cure', -7, [-4, 100], theta_offset=2)
        },
        'survival': True, 'discrete': False, 'link': ["default", "log", "neglog"], 'pdf': "agamma"
    },
    'gammajw': {
        'doc': "A special case of the Gamma likelihood", 'hyper': {}, 'survival': False, 'discrete': False,
        'link': ["default", "log", "neglog"], 'pdf': "gammajw"
    },
    'gammajwsurv': {
        'doc': "A special case of the Gamma likelihood (survival)",
        'hyper': _gen_beta_hyper(58200, 10, 'GammaJW-Cure', -7, [-4, 100]),
        'survival': True, 'discrete': False, 'link': ["default", "log"], 'pdf': "gammajw"
    },
    'gammacount': {
        'doc': "A Gamma generalisation of the Poisson likelihood",
        'hyper': {
            'theta': {
                'hyperid': 59001, 'name': "log alpha", 'short_name': "alpha",
                'output_name': "Log-alpha parameter for Gammacount observations",
                'output_name_intern': "Alpha parameter for Gammacount observations",
                'initial': np.log(1.0), 'fixed': False, 'prior': "pc.gammacount", 'param': [3],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            }
        },
        'survival': False, 'discrete': False, 'link': ["default", "log"], 'pdf': "gammacount"
    },
    'qkumar': {
        'doc': "A quantile version of the Kumar likelihood",
        'hyper': {
            'theta': {
                'hyperid': 60001, 'name': "precision parameter", 'short_name': "prec",
                'output_name': "precision for qkumar observations",
                'output_name_intern': "log precision for qkumar observations",
                'initial': 1, 'fixed': False, 'prior': "loggamma", 'param': [1, 0.1],
                'to_theta': lambda x, sc=0.1: np.log(x) / sc,
                'from_theta': lambda x, sc=0.1: np.exp(sc * x)
            }
        },
        'survival': False, 'discrete': False, 'link': ["default", "logit", "loga", "cauchit"], 'pdf': "qkumar"
    },
    'qloglogistic': {
        'doc': "A quantile loglogistic likelihood",
        'hyper': {
            'theta': {
                'hyperid': 60011, 'name': "log alpha", 'short_name': "alpha",
                'output_name': "alpha for qloglogistic observations",
                'output_name_intern': "log alpha for qloglogistic observations",
                'initial': 1, 'fixed': False, 'prior': "loggamma", 'param': [25, 25],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            }
        },
        'survival': False, 'discrete': False, 'link': ["default", "log", "neglog"], 'pdf': "qloglogistic"
    },
    'qloglogisticsurv': {
        'doc': "A quantile loglogistic likelihood (survival)",
        'hyper': {
            'theta1': {
                'hyperid': 60021, 'name': "log alpha", 'short_name': "alpha",
                'output_name': "alpha for qloglogisticsurv observations",
                'output_name_intern': "log alpha for qloglogisticsurv observations",
                'initial': 1, 'fixed': False, 'prior': "loggamma", 'param': [25, 25],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            # theta2..theta11 are the cure-model betas; keep theta1 for alpha
            **_gen_beta_hyper(60022, 10, 'qlogLogistic-Cure', -5, [-4, 100], theta_offset=2)
        },
        'survival': True, 'discrete': False, 'link': ["default", "log", "neglog"], 'pdf': "qloglogistic"
    },
    'beta': {
        'doc': "The Beta likelihood",
        'hyper': {
            'theta': {
                'hyperid': 61001, 'name': "precision parameter", 'short_name': "phi",
                'output_name': "precision parameter for the beta observations",
                'output_name_intern': "intern precision-parameter for the beta observations",
                'initial': np.log(10), 'fixed': False, 'prior': "loggamma", 'param': [1, 0.1],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            }
        },
        'survival': False, 'discrete': False,
        'link': ["default", "logit", "loga", "cauchit", "probit", "cloglog", "ccloglog", "loglog"], 'pdf': "beta"
    },
    'obeta': {
        'doc': "The ordered Beta likelihood",
        'hyper': {
            'theta1': {
                'hyperid': 61101, 'name': "precision parameter", 'short_name': "phi",
                'output_name': "precision-parameter for the obeta observations",
                'output_name_intern': "intern precision-parameter for the obeta observations",
                'initial': np.log(10), 'fixed': False, 'prior': "loggamma", 'param': [1, 0.1],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 61102, 'name': "offset location", 'short_name': "loc",
                'output_name': "offset location-parameter for the obeta observations",
                'output_name_intern': "intern offset location-parameter for the obeta observations",
                'initial': 0, 'fixed': False, 'prior': "normal", 'param': [0, 10],
                'to_theta': lambda x: x, 'from_theta': lambda x: x
            },
            'theta3': {
                'hyperid': 61103, 'name': "offset width", 'short_name': "width",
                'output_name': "offset width-parameter for the obeta observations",
                'output_name_intern': "intern offset width-parameter for the obeta observations",
                'initial': 0, 'fixed': False, 'prior': "loggamma", 'param': [1, 1],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            }
        },
        'status': "experimental", 'survival': False, 'discrete': False,
        'link': ["default", "logit", "loga", "cauchit", "probit", "cloglog", "ccloglog", "loglog"], 'pdf': "obeta"
    },
    'betabinomial': {
        'doc': "The Beta-Binomial likelihood",
        'hyper': {
            'theta': {
                'hyperid': 62001, 'name': "overdispersion", 'short_name': "rho",
                'output_name': "overdispersion for the betabinomial observations",
                'output_name_intern': "intern overdispersion for the betabinomial observations",
                'initial': 0, 'fixed': False, 'prior': "gaussian", 'param': [0.0, 0.4],
                'to_theta': lambda x: np.log(x / (1 - x)), 'from_theta': lambda x: np.exp(x) / (1 + np.exp(x))
            }
        },
        'survival': False, 'discrete': True,
        'link': ["default", "logit", "loga", "cauchit", "probit", "cloglog", "ccloglog", "loglog", "robit", "sn"],
        'pdf': "betabinomial"
    },
    'betabinomialna': {
        'doc': "The Beta-Binomial Normal approximation likelihood",
        'hyper': {
            'theta': {
                'hyperid': 62101, 'name': "overdispersion", 'short_name': "rho",
                'output_name': "overdispersion for the betabinomialna observations",
                'output_name_intern': "intern overdispersion for the betabinomialna observations",
                'initial': 0, 'fixed': False, 'prior': "gaussian", 'param': [0.0, 0.4],
                'to_theta': lambda x: np.log(x / (1 - x)), 'from_theta': lambda x: np.exp(x) / (1 + np.exp(x))
            }
        },
        'survival': False, 'discrete': True,
        'link': ["default", "logit", "loga", "cauchit", "probit", "cloglog", "ccloglog", "loglog", "robit", "sn"],
        'pdf': "betabinomialna"
    },
    'cbinomial': {
        'doc': "The clustered Binomial likelihood", 'hyper': {}, 'survival': False, 'discrete': True,
        'link': ["default", "logit", "loga", "cauchit", "probit", "cloglog", "ccloglog", "loglog", "robit", "sn"],
        'pdf': "cbinomial"
    },
    'nbinomial': {
        'doc': "The negBinomial likelihood",
        'hyper': {
            'theta': {
                'hyperid': 63001, 'name': "size", 'short_name': "size",
                'output_name': "size for the nbinomial observations (1/overdispersion)",
                'output_name_intern': "log size for the nbinomial observations (1/overdispersion)",
                'initial': np.log(10), 'fixed': False, 'prior': "pcmgamma", 'param': [7],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            }
        },
        'survival': False, 'discrete': True, 'link': ["default", "log", "logoffset", "quantile"], 'pdf': "nbinomial"
    },
    'nbinomial2': {
        'doc': "The negBinomial2 likelihood", 'hyper': {}, 'survival': False, 'discrete': True,
        'link': ["default", "logit", "loga", "cauchit", "probit", "cloglog", "ccloglog", "loglog"], 'pdf': "nbinomial"
    },
    'cennbinomial2': {
        'doc': "The CenNegBinomial2 likelihood (similar to cenpoisson2)",
        'hyper': {
            'theta': {
                'hyperid': 63101, 'name': "size", 'short_name': "size",
                'output_name': "size for the cennbinomial2 observations (1/overdispersion)",
                'output_name_intern': "log size for the cennbinomial2 observations (1/overdispersion)",
                'initial': np.log(10), 'fixed': False, 'prior': "pcmgamma", 'param': [7],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            }
        },
        'survival': False, 'discrete': True, 'link': ["default", "log", "logoffset", "quantile"], 'pdf': "cennbinomial2"
    },
    'simplex': {
        'doc': "The simplex likelihood",
        'hyper': {
            'theta': {
                'hyperid': 64001, 'name': "log precision", 'short_name': "prec",
                'output_name': "Precision for the Simplex observations",
                'output_name_intern': "Log precision for the Simplex observations",
                'initial': 4, 'fixed': False, 'prior': "loggamma", 'param': [1, 0.00005],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            }
        },
        'survival': False, 'discrete': False,
        'link': ["default", "logit", "loga", "cauchit", "probit", "cloglog", "ccloglog", "loglog"], 'pdf': "simplex"
    },
    'gaussian': {
        'doc': "The Gaussian likelihoood",
        'hyper': {
            'theta1': {
                'hyperid': 65001, 'name': "log precision", 'short_name': "prec",
                'output_name': "Precision for the Gaussian observations",
                'output_name_intern': "Log precision for the Gaussian observations",
                'initial': 4, 'fixed': False, 'prior': "loggamma", 'param': [1, 0.00005],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 65002, 'name': "log precision offset", 'short_name': "precoffset",
                'output_name': "NOT IN USE", 'output_name_intern': "NOT IN USE",
                'initial': -2.0 * np.log(np.finfo(float).eps), 'fixed': True, 'prior': "none",
                'param': [], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            }
        },
        'survival': False, 'discrete': False,
        'link': ["default", "identity", "logit", "loga", "cauchit", "log", "logoffset"], 'pdf': "gaussian"
    },
    'stdgaussian': {
        'doc': "The stdGaussian likelihoood", 'hyper': {}, 'survival': False, 'discrete': False,
        'link': ["default", "identity", "logit", "loga", "cauchit", "log", "logoffset"], 'pdf': "gaussian"
    },
    'gaussianjw': {
        'doc': "The GaussianJW likelihoood",
        'hyper': {
            'theta1': {'hyperid': 65101, 'name': "beta1", 'short_name': "beta1", 'output_name': "beta1 for GaussianJW observations", 'output_name_intern': "beta1 for GaussianJW observations", 'initial': 0, 'fixed': False, 'prior': "normal", 'param': [0, 100], 'to_theta': lambda x: x, 'from_theta': lambda x: x},
            'theta2': {'hyperid': 65102, 'name': "beta2", 'short_name': "beta2", 'output_name': "beta2 for GaussianJW observations", 'output_name_intern': "beta2 for GaussianJW observations", 'initial': 1, 'fixed': False, 'prior': "normal", 'param': [1, 100], 'to_theta': lambda x: x, 'from_theta': lambda x: x},
            'theta3': {'hyperid': 65103, 'name': "beta3", 'short_name': "beta3", 'output_name': "beta3 for GaussianJW observations", 'output_name_intern': "beta3 for GaussianJW observations", 'initial': -1, 'fixed': False, 'prior': "normal", 'param': [-1, 100], 'to_theta': lambda x: x, 'from_theta': lambda x: x}
        },
        'survival': False, 'discrete': False, 'link': ["default", "logit", "probit"], 'pdf': "gaussianjw"
    },
    'agaussian': {
        'doc': "The aggregated Gaussian likelihoood",
        'hyper': {
            'theta': {
                'hyperid': 66001, 'name': "log precision", 'short_name': "prec",
                'output_name': "Precision for the AggGaussian observations",
                'output_name_intern': "Log precision for the AggGaussian observations",
                'initial': 4, 'fixed': False, 'prior': "loggamma", 'param': [1, 0.00005],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            }
        },
        'survival': False, 'discrete': False,
        'link': ["default", "identity", "logit", "loga", "cauchit", "log", "logoffset"], 'pdf': "agaussian"
    },
    'ggaussian': {
        'doc': "Generalized Gaussian",
        'hyper': _gen_beta_hyper(66501, 10, 'ggaussian', 4, [9.33, 0.61]),
        'survival': False, 'discrete': False,
        'link': ["default", "identity"], 'link_simple': ["default", "log"], 'pdf': "ggaussian"
    },
    'ggaussianS': {
        'doc': "Generalized GaussianS",
        'hyper': _gen_beta_hyper(66601, 10, 'ggaussianS', 0, [0, 0.001]),
        'survival': False, 'discrete': False,
        'link': ["default", "log"], 'link_simple': ["default", "identity"], 'pdf': "ggaussian"
    },
    'bcgaussian': {
        'doc': "The Box-Cox Gaussian likelihoood",
        'hyper': {
            'theta1': {
                'hyperid': 65010, 'name': "log precision", 'short_name': "prec",
                'output_name': "Precision for the Box-Cox Gaussian observations",
                'output_name_intern': "Log precision for the Box-Cox Gaussian observations",
                'initial': 4, 'fixed': False, 'prior': "loggamma", 'param': [1, 0.00005],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 65011, 'name': "Box-Cox transformation parameter", 'short_name': "lambda",
                'output_name': "NOT IN USE", 'output_name_intern': "NOT IN USE",
                'initial': 1, 'fixed': False, 'prior': "gaussian", 'param': [1, 8],
                'to_theta': lambda x: x, 'from_theta': lambda x: x
            }
        },
        'status': "disabled", 'survival': False, 'discrete': False,
        'link': ["default", "identity"], 'pdf': "bcgaussian"
    },
    'exppower': {
        'doc': "The exponential power likelihoood",
        'hyper': {
            'theta1': {
                'hyperid': 65021, 'name': "log precision", 'short_name': "prec", 'output_name': "NOT IN USE",
                'output_name_intern': "NOT IN USE", 'initial': 4, 'fixed': False, 'prior': "loggamma",
                'param': [1, 0.00005], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 65022, 'name': "power", 'short_name': "beta", 'output_name': "NOT IN USE",
                'output_name_intern': "NOT IN USE", 'initial': 0, 'fixed': False, 'prior': "gaussian",
                'param': [0, 100], 'to_theta': lambda x: np.log(x-1), 'from_theta': lambda x: 1+np.exp(x)
            }
        },
        'status': "experimental", 'survival': False, 'discrete': False,
        'link': ["default", "identity", "quantile"], 'pdf': "exppower"
    },
    'sem': {
        'doc': "The SEM likelihoood", 'hyper': {}, 'survival': False, 'discrete': False,
        'link': ["default", "identity"], 'pdf': "sem"
    },
    'rcpoisson': {
        'doc': "Randomly censored Poisson",
        'hyper': _gen_beta_hyper(66701, 10, 'rcpoisson', 0, [0, 100]),
        'status': "experimental", 'survival': False, 'discrete': True,
        'link': ["default", "log"], 'pdf': "rcpoisson"
    },
    'tpoisson': {
        'doc': "Thinned Poisson",
        'hyper': _gen_beta_hyper(66721, 10, 'tpoisson', 0, [0, 100]),
        'status': "experimental", 'survival': False, 'discrete': True,
        'link': ["default", "log"], 'pdf': "tpoisson"
    },
    'circularnormal': {
        'doc': "The circular Gaussian likelihoood",
        'hyper': {
            'theta': {
                'hyperid': 67001, 'name': "log precision parameter", 'short_name': "prec",
                'output_name': "Precision parameter for the Circular Normal observations",
                'output_name_intern': "Log precision parameter for the Circular Normal observations",
                'initial': 2, 'fixed': False, 'prior': "loggamma", 'param': [1, 0.01],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            }
        },
        'survival': False, 'discrete': False, 'link': ["default", "tan", "tan.pi"],
        'pdf': "circular-normal", 'status': "disabled"
    },
    'wrappedcauchy': {
        'doc': "The wrapped Cauchy likelihoood",
        'hyper': {
            'theta': {
                'hyperid': 68001, 'name': "log precision parameter", 'short_name': "prec",
                'output_name': "Precision parameter for the Wrapped Cauchy observations",
                'output_name_intern': "Log precision parameter for the Wrapped Cauchy observations",
                'initial': 2, 'fixed': False, 'prior': "loggamma", 'param': [1, 0.005],
                'to_theta': lambda x: np.log(x / (1 - x)), 'from_theta': lambda x: np.exp(x) / (1 + np.exp(x))
            }
        },
        'survival': False, 'discrete': False, 'link': ["default", "tan", "tan.pi"],
        'pdf': "wrapped-cauchy", 'status': "disabled"
    },
    'iidgamma': {
        'doc': "(experimental)",
        'hyper': {
            'theta1': {
                'hyperid': 69001, 'name': "logshape", 'short_name': "shape", 'output_name': "Shape parameter for iid-gamma",
                'output_name_intern': "Log shape parameter for iid-gamma", 'initial': 0, 'fixed': False,
                'prior': "loggamma", 'param': [100, 100], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 69002, 'name': "lograte", 'short_name': "rate", 'output_name': "Rate parameter for iid-gamma",
                'output_name_intern': "Log rate parameter for iid-gamma", 'initial': 0, 'fixed': False,
                'prior': "loggamma", 'param': [100, 100], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            }
        },
        'survival': False, 'discrete': False, 'link': ["default", "identity"],
        'pdf': "iidgamma", 'status': "experimental"
    },
    'iidlogitbeta': {
        'doc': "(experimental)",
        'hyper': {
            'theta1': {
                'hyperid': 70001, 'name': "log.a", 'short_name': "a", 'output_name': "a parameter for iid-beta",
                'output_name_intern': "Log a parameter for iid-beta", 'initial': 1, 'fixed': False,
                'prior': "loggamma", 'param': [1, 1], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 70002, 'name': "log.b", 'short_name': "b", 'output_name': "Rate parameter for iid-gamma",
                'output_name_intern': "Log rate parameter for iid-gamma", 'initial': 1, 'fixed': False,
                'prior': "loggamma", 'param': [1, 1], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            }
        },
        'survival': False, 'discrete': False, 'link': ["default", "logit", "loga"],
        'pdf': "iidlogitbeta", 'status': "experimental"
    },
    'loggammafrailty': {
        'doc': "(experimental)",
        'hyper': {
            'theta': {
                'hyperid': 71001, 'name': "log precision", 'short_name': "prec",
                'output_name': "precision for the gamma frailty",
                'output_name_intern': "log precision for the gamma frailty", 'initial': 4, 'fixed': False,
                'prior': "loggamma", 'param': [1, 0.00005], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            }
        },
        'survival': False, 'discrete': False, 'link': ["default", "identity"],
        'pdf': "loggammafrailty", 'status': "experimental"
    },
    'logistic': {
        'doc': "The Logistic likelihoood",
        'hyper': {
            'theta': {
                'hyperid': 72001, 'name': "log precision", 'short_name': "prec",
                'output_name': "precision for the logistic observations",
                'output_name_intern': "log precision for the logistic observations",
                'initial': 1, 'fixed': False, 'prior': "loggamma", 'param': [1, 0.00005],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            }
        },
        'survival': False, 'discrete': False, 'link': ["default", "identity"], 'pdf': "logistic"
    },
    'sn': {
        'doc': "The Skew-Normal likelihoood",
        'hyper': {
            'theta1': {
                'hyperid': 74001, 'name': "log precision", 'short_name': "prec",
                'output_name': "precision for skew-normal observations",
                'output_name_intern': "log precision for skew-normal observations",
                'initial': 4, 'fixed': False, 'prior': "loggamma", 'param': [1, 0.00005],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 74002, 'name': "logit skew", 'short_name': "skew",
                'output_name': "Skewness for skew-normal observations",
                'output_name_intern': "Intern skewness for skew-normal observations",
                'initial': 0.00123456789, 'fixed': False, 'prior': "pc.sn", 'param': 10,
                'to_theta': lambda x, skew_max=0.988: np.log((1 + x / skew_max) / (1 - x / skew_max)),
                'from_theta': lambda x, skew_max=0.988: skew_max * (2 * np.exp(x) / (1 + np.exp(x)) - 1)
            }
        },
        'survival': False, 'discrete': False, 'link': ["default", "identity"], 'pdf': "sn"
    },
    'gev': {
        'doc': "The Generalized Extreme Value likelihood",
        'hyper': {
            'theta1': {
                'hyperid': 76001, 'name': "log precision", 'short_name': "prec",
                'output_name': "precision for GEV observations", 'output_name_intern': "log precision for GEV observations",
                'initial': 4, 'fixed': False, 'prior': "loggamma", 'param': [1, 0.00005],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 76002, 'name': "tail parameter", 'short_name': "tail",
                'output_name': "tail parameter for GEV observations", 'output_name_intern': "tail parameter for GEV observations",
                'initial': 0, 'fixed': False, 'prior': "gaussian", 'param': [0, 25],
                'to_theta': lambda x: x, 'from_theta': lambda x: x
            }
        },
        'survival': False, 'discrete': False, 'link': ["default", "identity"],
        'status': "disabled: Use likelihood model 'bgev' instead; see inla.doc('bgev')", 'pdf': "gev"
    },
    'lognormal': {
        'doc': "The log-Normal likelihood",
        'hyper': {
            'theta': {
                'hyperid': 77101, 'name': "log precision", 'short_name': "prec",
                'output_name': "Precision for the lognormal observations",
                'output_name_intern': "Log precision for the lognormal observations",
                'initial': 0, 'fixed': False, 'prior': "loggamma", 'param': [1, 0.00005],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            }
        },
        'survival': False, 'discrete': False, 'link': ["default", "identity"], 'pdf': "lognormal"
    },
    'lognormalsurv': {
        'doc': "The log-Normal likelihood (survival)",
        'hyper': {
            'theta1': {
                'hyperid': 78001, 'name': "log precision", 'short_name': "prec",
                'output_name': "Precision for the lognormalsurv observations",
                'output_name_intern': "Log precision for the lognormalsurv observations",
                'initial': 0, 'fixed': False, 'prior': "loggamma", 'param': [1, 0.00005],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            **_gen_beta_hyper(78002, 10, 'logNormal-Cure', -7, [-4, 100], theta_offset=2)
        },
        'survival': True, 'discrete': False, 'link': ["default", "identity"], 'pdf': "lognormal"
    },
    'exponential': {
        'doc': "The Exponential likelihood", 'hyper': {}, 'survival': False, 'discrete': False,
        'link': ["default", "log"], 'pdf': "exponential"
    },
    'exponentialsurv': {
        'doc': "The Exponential likelihood (survival)",
        'hyper': _gen_beta_hyper(78020, 10, 'Exp-Cure', -4, [-1, 100]),
        'survival': True, 'discrete': False, 'link': ["default", "log", "neglog"], 'pdf': "exponential"
    },
    'coxph': {
        'doc': "Cox-proportional hazard likelihood", 'hyper': {}, 'survival': True, 'discrete': True,
        'link': ["default", "log", "neglog"], 'pdf': "coxph"
    },
    'weibull': {
        'doc': "The Weibull likelihood",
        'hyper': {
            'theta': {
                'hyperid': 79001, 'name': "log alpha", 'short_name': "alpha",
                'output_name': "alpha parameter for weibull", 'output_name_intern': "alpha_intern for weibull",
                'initial': -2, 'fixed': False, 'prior': "pc.alphaw", 'param': [5],
                'to_theta': lambda x, sc=0.1: np.log(x) / sc,
                'from_theta': lambda x, sc=0.1: np.exp(sc * x)
            }
        },
        'survival': False, 'discrete': False, 'link': ["default", "log", "neglog", "quantile"], 'pdf': "weibull"
    },
    'weibullsurv': {
        'doc': "The Weibull likelihood (survival)",
        'hyper': {
            'theta': { # R code calls this 'theta', not 'theta1', so let's match that.
                'hyperid': 79101, 'name': "log alpha", 'short_name': "alpha",
                'output_name': "alpha parameter for weibullsurv", 'output_name_intern': "alpha_intern for weibullsurv",
                'initial': -2, 'fixed': False, 'prior': "pc.alphaw", 'param': [5],
                'to_theta': lambda x, sc=0.1: np.log(x) / sc,
                'from_theta': lambda x, sc=0.1: np.exp(sc * x)
            },
            **_gen_beta_hyper(79102, 10, 'Weibull-Cure', -7, [-4, 100])
        },
        'survival': True, 'discrete': False, 'link': ["default", "log", "neglog", "quantile"], 'pdf': "weibull"
    },
    # NOTE: Duplicate block removed (sn, gev, lognormal, lognormalsurv, exponential, exponentialsurv, coxph, weibull, weibullsurv)
    # The correct definitions are above (lines 662-768)
    'zeroinflatednbinomial2': {
        'doc': "Zero inflated negBinomial, type 2",
        'hyper': {
            'theta1': {
                'hyperid': 99001, 'name': "log size", 'short_name': "size",
                'output_name': "size for nbinomial zero-inflated observations",
                'output_name_intern': "log size for nbinomial zero-inflated observations",
                'initial': np.log(10), 'fixed': False, 'prior': "pcmgamma", 'param': [7],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 99002, 'name': "log alpha", 'short_name': "a",
                'output_name': "parameter alpha for zero-inflated nbinomial2",
                'output_name_intern': "parameter alpha.intern for zero-inflated nbinomial2",
                'initial': np.log(2), 'fixed': False, 'prior': "gaussian", 'param': [np.log(2), 1],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            }
        },
        'survival': False, 'discrete': False, 'link': ["default", "log"], 'pdf': "zeroinflated"
    },
    't': {
        'doc': "Student-t likelihood",
        'hyper': {
            'theta1': {
                'hyperid': 100001, 'name': "log precision", 'short_name': "prec",
                'output_name': "precision for the student-t observations",
                'output_name_intern': "log precision for the student-t observations",
                'initial': 0, 'fixed': False, 'prior': "loggamma", 'param': [1, 0.00005],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 100002, 'name': "log degrees of freedom", 'short_name': "dof",
                'output_name': "degrees of freedom for student-t",
                'output_name_intern': "dof_intern for student-t",
                'initial': 5, 'fixed': False, 'prior': "pcdof", 'param': [15, 0.5],
                'to_theta': lambda x: np.log(x - 2), 'from_theta': lambda x: 2 + np.exp(x)
            }
        },
        'survival': False, 'discrete': False, 'link': ["default", "identity"], 'pdf': "student-t"
    },
    'tstrata': {
        'doc': "A stratified version of the Student-t likelihood",
        'hyper': {
            'theta1': {
                'hyperid': 101001, 'name': "log degrees of freedom", 'short_name': "dof",
                'output_name_intern': "dof_intern for tstrata",
                'output_name': "degrees of freedom for tstrata",
                'initial': 4, 'fixed': False, 'prior': "pcdof", 'param': [15, 0.5],
                'to_theta': lambda x: np.log(x - 5), 'from_theta': lambda x: 5 + np.exp(x)
            },
            **{f'theta{i+1}': {
                'hyperid': 101001 + i,
                'name': f"log precision{i}", 'short_name': f"prec{i}",
                'output_name': f"Prec for tstrata strata[{i}]",
                'output_name_intern': f"Log prec for tstrata strata[{i}]",
                'initial': 2, 'fixed': False, 'prior': "loggamma", 'param': [1, 0.00005],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            } for i in range(1, 11)}
        },
        'survival': False, 'discrete': False, 'link': ["default", "identity"], 'pdf': "tstrata"
    },
    'nmix': {
        'doc': "Binomial-Poisson mixture",
        'hyper': _gen_beta_hyper(101101, 15, 'NMix', np.log(10), [0, 0.5]),
        'survival': False, 'discrete': True, 'link': ["default", "logit", "loga", "probit"], 'pdf': "nmix"
    },
    'nmixnb': {
        'doc': "NegBinomial-Poisson mixture",
        'hyper': {
            **_gen_beta_hyper(101121, 15, 'NMixNB', np.log(10), [0, 0.5]),
            'theta16': {
                'hyperid': 101136, 'name': "overdispersion", 'short_name': "overdispersion",
                'output_name': "overdispersion for NMixNB observations",
                'output_name_intern': "log_overdispersion for NMixNB observations",
                'initial': 0, 'fixed': False, 'prior': "pc.gamma", 'param': [7],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            }
        },
        'survival': False, 'discrete': True, 'link': ["default", "logit", "loga", "probit"], 'pdf': "nmixnb"
    },
    'gp': {
        'doc': "Generalized Pareto likelihood",
        'hyper': {
            'theta': {
                'hyperid': 101201, 'name': "tail", 'short_name': "xi",
                'output_name': "Tail parameter for the gp observations",
                'output_name_intern': "Intern tail parameter for the gp observations",
                'initial': 2, 'fixed': False, 'prior': "pc.gevtail", 'param': [7, 0.0, 0.5],
                'to_theta': _gevit_to, 'from_theta': _gevit_from
            }
        },
        'survival': False, 'discrete': True, 'link': ["default", "quantile"], 'pdf': "genPareto"
    },
    'egp': {
        'doc': "Exteneded Generalized Pareto likelihood",
        'hyper': {
            'theta1': {
                'hyperid': 101211, 'name': "tail", 'short_name': "xi",
                'output_name': "Tail parameter for egp observations",
                'output_name_intern': "Intern tail parameter for egp observations",
                'initial': 0, 'fixed': False, 'prior': "pc.egptail", 'param': [5, -0.5, 0.5],
                'to_theta': _gevit_to, 'from_theta': _gevit_from
            },
            'theta2': {
                'hyperid': 101212, 'name': "shape", 'short_name': "kappa",
                'output_name': "Shape parameter for the egp observations",
                'output_name_intern': "Intern shape parameter for the egp observations",
                'initial': 0, 'fixed': False, 'prior': "loggamma", 'param': [100, 100],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            }
        },
        'status': "experimental", 'survival': False, 'discrete': False,
        'link': ["default", "quantile"], 'pdf': "egp"
    },
    'dgp': {
        'doc': "Discrete generalized Pareto likelihood",
        'hyper': {
            'theta': {
                'hyperid': 101301, 'name': "tail", 'short_name': "xi",
                'output_name': "Tail parameter for the dgp observations",
                'output_name_intern': "Intern tail parameter for the dgp observations",
                'initial': 2, 'fixed': False, 'prior': "pc.gevtail", 'param': [7, 0.0, 0.5],
                'to_theta': _gevit_to, 'from_theta': _gevit_from
            }
        },
        'survival': False, 'discrete': True, 'link': ["default", "quantile"], 'pdf': "dgp"
    },
    'logperiodogram': {
        'doc': "Likelihood for the log-periodogram", 'hyper': {}, 'survival': False, 'discrete': False,
        'link': ["default", "identity"], 'pdf': None
    },
    'tweedie': {
        'doc': "Tweedie distribution",
        'hyper': {
            'theta1': {
                'hyperid': 102101, 'name': "p", 'short_name': "p",
                'output_name': "p parameter for Tweedie", 'output_name_intern': "p_intern parameter for Tweedie",
                'initial': 0, 'fixed': False, 'prior': "normal", 'param': [0, 100],
                'to_theta': lambda x, interval=(1.0, 2.0): np.log(-(interval[0] - x) / (interval[1] - x)),
                'from_theta': lambda x, interval=(1.0, 2.0): interval[0] + (interval[1] - interval[0]) * np.exp(x) / (1.0 + np.exp(x))
            },
            'theta2': {
                'hyperid': 102201, 'name': "dispersion", 'short_name': "phi",
                'output_name': "Dispersion parameter for Tweedie",
                'output_name_intern': "Log dispersion parameter for Tweedie",
                'initial': -4, 'fixed': False, 'prior': "loggamma", 'param': [100, 100],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            }
        },
        'survival': False, 'discrete': False, 'link': ["default", "log"], 'pdf': "tweedie"
    },
    'fmri': {
        'doc': "fmri distribution (special nc-chi)",
        'hyper': {
            'theta1': {
                'hyperid': 103101, 'name': "precision", 'short_name': "prec",
                'output_name': "Precision for fmri", 'output_name_intern': "Log precision for fmri",
                'initial': 0, 'fixed': False, 'prior': "loggamma", 'param': [10, 10],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 103202, 'name': "dof", 'short_name': "df",
                'output_name': "NOT IN USE", 'output_name_intern': "NOT IN USE",
                'initial': 4, 'fixed': True, 'prior': "normal", 'param': [0, 1],
                'to_theta': lambda x: x, 'from_theta': lambda x: x
            }
        },
        'survival': False, 'discrete': False, 'link': ["default", "log"], 'pdf': "fmri"
    },
    'fmrisurv': {
        'doc': "fmri distribution (special nc-chi)",
        'hyper': {
            'theta1': {
                'hyperid': 104101, 'name': "precision", 'short_name': "prec",
                'output_name': "Precision for fmrisurv", 'output_name_intern': "Log precision for fmrisurv",
                'initial': 0, 'fixed': False, 'prior': "loggamma", 'param': [10, 10],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 104201, 'name': "dof", 'short_name': "df",
                'output_name': "NOT IN USE", 'output_name_intern': "NOT IN USE",
                'initial': 4, 'fixed': True, 'prior': "normal", 'param': [0, 1],
                'to_theta': lambda x: x, 'from_theta': lambda x: x
            }
        },
        'survival': True, 'discrete': False, 'link': ["default", "log"], 'pdf': "fmri"
    },
    'gompertz': {
        'doc': "gompertz distribution",
        'hyper': {
            'theta': {
                'hyperid': 105101, 'name': "shape", 'short_name': "alpha",
                'output_name_intern': "alpha_intern for Gompertz", 'output_name': "alpha parameter for Gompertz",
                'initial': -1, 'fixed': False, 'prior': "normal", 'param': [0, 1],
                'to_theta': lambda x, sc=0.1: np.log(x) / sc,
                'from_theta': lambda x, sc=0.1: np.exp(sc * x)
            }
        },
        'survival': False, 'discrete': False, 'link': ["default", "log", "neglog"], 'pdf': "gompertz"
    },
    'gompertzsurv': {
        'doc': "gompertz distribution",
        'hyper': {
            'theta1': {
                'hyperid': 106101, 'name': "shape", 'short_name': "alpha",
                'output_name_intern': "alpha_intern for Gompertz-surv",
                'output_name': "alpha parameter for Gompertz-surv", 'initial': -10, 'fixed': False,
                'prior': "normal", 'param': [0, 1], 'to_theta': lambda x, sc=0.1: np.log(x) / sc,
                'from_theta': lambda x, sc=0.1: np.exp(sc * x)
            },
            **_gen_beta_hyper(106102, 10, 'Gompertz-Cure', -5, [-4, 100])
        },
        'survival': True, 'discrete': False, 'link': ["default", "log", "neglog"], 'pdf': "gompertz"
    },
    'dgompertzsurv': {
        'doc': "destructive gompertz (survival) distribution",
        'hyper': {
            'theta': {
                'hyperid': 108101, 'name': "shape", 'short_name': "alpha",
                'output_name_intern': "alpha_intern for dGompertz", 'output_name': "alpha parameter for dGompertz",
                'initial': -1, 'fixed': False, 'prior': "normal", 'param': [0, 10],
                'to_theta': lambda x: x, 'from_theta': lambda x: x
            }
        },
        'experimental': True, 'survival': True, 'discrete': False,
        'link': ["default", "log", "neglog"], 'pdf': "dgompertz"
    },
    'vm': {
        'doc': "von Mises circular distribution",
        'hyper': {
            'theta': {
                'hyperid': 109101, 'name': "precision", 'short_name': "prec",
                'output_name_intern': "prec_intern for vm", 'output_name': "precision parameter for vm",
                'initial': 2, 'fixed': False, 'prior': "loggamma", 'param': [1, 0.01],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            }
        },
        'experimental': True, 'survival': False, 'discrete': False,
        'link': ["default", "circular", "tan", "tan.pi", "identity"], 'pdf': "vm"
    },
    'cloglike': {
        'doc': "User-defined likelihood", 'hyper': {}, 'experimental': True, 'survival': False,
        'discrete': False, 'link': ["default", "identity"], 'pdf': "cloglike"
    },
    'logistic': {
        'doc': "The Logistic likelihood",
        'hyper': {
            'theta': {
                'hyperid': 72001, 'name': "log precision", 'short_name': "prec",
                'output_name': "precision for the logistic observations",
                'output_name_intern': "log precision for the logistic observations",
                'initial': 1, 'fixed': False, 'prior': "loggamma", 'param': [1, 0.00005],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            }
        },
        'survival': False, 'discrete': False, 'link': ["default", "identity"], 'pdf': "logistic"
    },
    'loglogistic': {
        'doc': "The loglogistic likelihood",
        'hyper': {
            'theta': {
                'hyperid': 80001, 'name': "log alpha", 'short_name': "alpha",
                'output_name': "alpha for loglogistic observations",
                'output_name_intern': "log alpha for loglogistic observations",
                'initial': 1, 'fixed': False, 'prior': "loggamma", 'param': [25, 25],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            }
        },
        'survival': False, 'discrete': False, 'link': ["default", "log", "neglog"], 'pdf': "loglogistic"
    },
    'loglogisticsurv': {
        'doc': "The loglogistic likelihood (survival)",
        'hyper': {
            'theta1': {
                'hyperid': 80011, 'name': "log alpha", 'short_name': "alpha",
                'output_name': "alpha for loglogisticsurv observations",
                'output_name_intern': "log alpha for loglogisticsurv observations",
                'initial': 1, 'fixed': False, 'prior': "loggamma", 'param': [25, 25],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            # Keep theta1 for alpha; place cure betas in theta2..theta11 like R-INLA
            **_gen_beta_hyper(80012, 10, 'logLogistic-Cure', -5, [-4, 100], theta_offset=2)
        },
        'survival': True, 'discrete': False, 'link': ["default", "log", "neglog"], 'pdf': "loglogistic"
    },
    'stochvol': {
        'doc': "The Gaussian stochvol likelihood",
        'hyper': {
            'theta': {
                'hyperid': 82001, 'name': "log precision", 'short_name': "prec",
                'output_name': "Offset precision for stochvol",
                'output_name_intern': "Log offset precision for stochvol",
                'initial': 500, 'fixed': True, 'prior': "loggamma", 'param': [1, 0.005],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            }
        },
        'survival': False, 'discrete': False, 'link': ["default", "log"], 'pdf': "stochvolgaussian"
    },
    'stochvolln': {
        'doc': "The Log-Normal stochvol likelihood",
        'hyper': {
            'theta': {
                'hyperid': 82011, 'name': "offset", 'short_name': "c",
                'output_name': "Mean offset for stochvolln",
                'output_name_intern': "Mean offset for stochvolln",
                'initial': 0, 'fixed': False, 'prior': "normal", 'param': [0, 10],
                'to_theta': lambda x: x, 'from_theta': lambda x: x
            }
        },
        'survival': False, 'discrete': False, 'link': ["default", "log"], 'pdf': "stochvolln"
    },
    'stochvolsn': {
        'doc': "The SkewNormal stochvol likelihood",
        'hyper': {
            'theta1': {
                'hyperid': 82101, 'name': "logit skew", 'short_name': "skew",
                'output_name': "Skewness for stochvol_sn observations",
                'output_name_intern': "Intern skewness for stochvol_sn observations",
                'initial': 0.00123456789, 'fixed': False, 'prior': "pc.sn", 'param': 10,
                'to_theta': lambda x, skew_max=0.988: np.log((1 + x / skew_max) / (1 - x / skew_max)),
                'from_theta': lambda x, skew_max=0.988: skew_max * (2 * np.exp(x) / (1 + np.exp(x)) - 1)
            },
            'theta2': {
                'hyperid': 82102, 'name': "log precision", 'short_name': "prec",
                'output_name': "Offset precision for stochvol_sn",
                'output_name_intern': "Log offset precision for stochvol_sn",
                'initial': 500, 'fixed': True, 'prior': "loggamma", 'param': [1, 0.005],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            }
        },
        'survival': False, 'discrete': False, 'link': ["default", "log"], 'pdf': "stochvolsn"
    },
    'stochvolt': {
        'doc': "The Student-t stochvol likelihood",
        'hyper': {
            'theta': {
                'hyperid': 83001, 'name': "log degrees of freedom", 'short_name': "dof",
                'output_name': "degrees of freedom for stochvol student-t",
                'output_name_intern': "dof_intern for stochvol student-t",
                'initial': 4, 'fixed': False, 'prior': "pc.dof", 'param': [15, 0.5],
                'to_theta': lambda x: np.log(x - 2), 'from_theta': lambda x: 2 + np.exp(x)
            }
        },
        'survival': False, 'discrete': False, 'link': ["default", "log"], 'pdf': "stochvolt"
    },
    'stochvolnig': {
        'doc': "The Normal inverse Gaussian stochvol likelihood",
        'hyper': {
            'theta1': {
                'hyperid': 84001, 'name': "skewness", 'short_name': "skew",
                'output_name_intern': "skewness_param_intern for stochvol-nig",
                'output_name': "skewness parameter for stochvol-nig",
                'initial': 0, 'fixed': False, 'prior': "gaussian", 'param': [0, 10],
                'to_theta': lambda x: x, 'from_theta': lambda x: x
            },
            'theta2': {
                'hyperid': 84002, 'name': "shape", 'short_name': "shape",
                'output_name': "shape parameter for stochvol-nig",
                'output_name_intern': "shape_param_intern for stochvol-nig",
                'initial': 0, 'fixed': False, 'prior': "loggamma", 'param': [1, 0.5],
                'to_theta': lambda x: np.log(x - 1), 'from_theta': lambda x: 1 + np.exp(x)
            }
        },
        'survival': False, 'discrete': False, 'link': ["default", "log"], 'pdf': "stochvolnig"
    },
    'zeroinflatedpoisson0': {
        'doc': "Zero-inflated Poisson, type 0",
        'hyper': {
            'theta': {
                'hyperid': 85001, 'name': "logit probability", 'short_name': "prob",
                'output_name': "zero-probability parameter for zero-inflated poisson_0",
                'output_name_intern': "intern zero-probability parameter for zero-inflated poisson_0",
                'initial': -1, 'fixed': False, 'prior': "gaussian", 'param': [-1, 0.2],
                'to_theta': lambda x: np.log(x / (1 - x)), 'from_theta': lambda x: np.exp(x) / (1 + np.exp(x))
            }
        },
        'survival': False, 'discrete': False, 'link': ["default", "log"], 'pdf': "zeroinflated"
    },
    'zeroinflatedpoisson1': {
        'doc': "Zero-inflated Poisson, type 1",
        'hyper': {
            'theta': {
                'hyperid': 86001, 'name': "logit probability", 'short_name': "prob",
                'output_name': "zero-probability parameter for zero-inflated poisson_1",
                'output_name_intern': "intern zero-probability parameter for zero-inflated poisson_1",
                'initial': -1, 'fixed': False, 'prior': "gaussian", 'param': [-1, 0.2],
                'to_theta': lambda x: np.log(x / (1 - x)), 'from_theta': lambda x: np.exp(x) / (1 + np.exp(x))
            }
        },
        'survival': False, 'discrete': False, 'link': ["default", "log"], 'pdf': "zeroinflated"
    },
    'zeroinflatedpoisson2': {
        'doc': "Zero-inflated Poisson, type 2",
        'hyper': {
            'theta': {
                'hyperid': 87001, 'name': "log alpha", 'short_name': "a",
                'output_name': "zero-probability parameter for zero-inflated poisson_2",
                'output_name_intern': "intern zero-probability parameter for zero-inflated poisson_2",
                'initial': np.log(2), 'fixed': False, 'prior': "gaussian", 'param': [np.log(2), 1],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            }
        },
        'survival': False, 'discrete': False, 'link': ["default", "log"], 'pdf': "zeroinflated"
    },
    'zeroinflatedcenpoisson0': {
        'doc': "Zero-inflated censored Poisson, type 0",
        'hyper': {
            'theta': {
                'hyperid': 87101, 'name': "logit probability", 'short_name': "prob",
                'output_name': "zero-probability parameter for zero-inflated poisson_0",
                'output_name_intern': "intern zero-probability parameter for zero-inflated poisson_0",
                'initial': -1, 'fixed': False, 'prior': "gaussian", 'param': [-1, 0.2],
                'to_theta': lambda x: np.log(x / (1 - x)), 'from_theta': lambda x: np.exp(x) / (1 + np.exp(x))
            }
        },
        'survival': False, 'discrete': False, 'link': ["default", "log"], 'pdf': "zeroinflated"
    },
    'zeroinflatedcenpoisson1': {
        'doc': "Zero-inflated censored Poisson, type 1",
        'hyper': {
            'theta': {
                'hyperid': 87201, 'name': "logit probability", 'short_name': "prob",
                'output_name': "zero-probability parameter for zero-inflated poisson_1",
                'output_name_intern': "intern zero-probability parameter for zero-inflated poisson_1",
                'initial': -1, 'fixed': False, 'prior': "gaussian", 'param': [-1, 0.2],
                'to_theta': lambda x: np.log(x / (1 - x)), 'from_theta': lambda x: np.exp(x) / (1 + np.exp(x))
            }
        },
        'survival': False, 'discrete': False, 'link': ["default", "log"], 'pdf': "zeroinflated"
    },
    'zeroinflatedbetabinomial0': {
        'doc': "Zero-inflated Beta-Binomial, type 0",
        'hyper': {
            'theta1': {
                'hyperid': 88001, 'name': "overdispersion", 'short_name': "rho",
                'output_name': "rho for zero-inflated betabinomial_0",
                'output_name_intern': "rho_intern for zero-inflated betabinomial_0",
                'initial': 0, 'fixed': False, 'prior': "gaussian", 'param': [0.0, 0.4],
                'to_theta': lambda x: np.log(x / (1 - x)), 'from_theta': lambda x: np.exp(x) / (1 + np.exp(x))
            },
            'theta2': {
                'hyperid': 88002, 'name': "logit probability", 'short_name': "prob",
                'output_name': "zero-probability parameter for zero-inflated betabinomial_0",
                'output_name_intern': "intern zero-probability parameter for zero-inflated betabinomial_0",
                'initial': -1, 'fixed': False, 'prior': "gaussian", 'param': [-1, 0.2],
                'to_theta': lambda x: np.log(x / (1 - x)), 'from_theta': lambda x: np.exp(x) / (1 + np.exp(x))
            }
        },
        'survival': False, 'discrete': True,
        'link': ["default", "logit", "loga", "cauchit", "probit", "cloglog", "ccloglog", "loglog", "robit", "sn"],
        'pdf': "zeroinflated"
    },
    'zeroinflatedbetabinomial1': {
        'doc': "Zero-inflated Beta-Binomial, type 1",
        'hyper': {
            'theta1': {
                'hyperid': 89001, 'name': "overdispersion", 'short_name': "rho",
                'output_name': "rho for zero-inflated betabinomial_1",
                'output_name_intern': "rho_intern for zero-inflated betabinomial_1",
                'initial': 0, 'fixed': False, 'prior': "gaussian", 'param': [0.0, 0.4],
                'to_theta': lambda x: np.log(x / (1 - x)), 'from_theta': lambda x: np.exp(x) / (1 + np.exp(x))
            },
            'theta2': {
                'hyperid': 89002, 'name': "logit probability", 'short_name': "prob",
                'output_name': "zero-probability parameter for zero-inflated betabinomial_1",
                'output_name_intern': "intern zero-probability parameter for zero-inflated betabinomial_1",
                'initial': -1, 'fixed': False, 'prior': "gaussian", 'param': [-1, 0.2],
                'to_theta': lambda x: np.log(x / (1 - x)), 'from_theta': lambda x: np.exp(x) / (1 + np.exp(x))
            }
        },
        'survival': False, 'discrete': True,
        'link': ["default", "logit", "loga", "cauchit", "probit", "cloglog", "ccloglog", "loglog", "robit", "sn"],
        'pdf': "zeroinflated"
    },
    'zeroinflatedbinomial0': {
        'doc': "Zero-inflated Binomial, type 0",
        'hyper': {
            'theta': {
                'hyperid': 90001, 'name': "logit probability", 'short_name': "prob",
                'output_name': "zero-probability parameter for zero-inflated binomial_0",
                'output_name_intern': "intern zero-probability parameter for zero-inflated binomial_0",
                'initial': -1, 'fixed': False, 'prior': "gaussian", 'param': [-1, 0.2],
                'to_theta': lambda x: np.log(x / (1 - x)), 'from_theta': lambda x: np.exp(x) / (1 + np.exp(x))
            }
        },
        'survival': False, 'discrete': False,
        'link': ["default", "logit", "loga", "cauchit", "probit", "cloglog", "ccloglog", "loglog", "robit", "sn"],
        'pdf': "zeroinflated"
    },
    'zeroinflatedbinomial1': {
        'doc': "Zero-inflated Binomial, type 1",
        'hyper': {
            'theta': {
                'hyperid': 91001, 'name': "logit probability", 'short_name': "prob",
                'output_name': "zero-probability parameter for zero-inflated binomial_1",
                'output_name_intern': "intern zero-probability parameter for zero-inflated binomial_1",
                'initial': -1, 'fixed': False, 'prior': "gaussian", 'param': [-1, 0.2],
                'to_theta': lambda x: np.log(x / (1 - x)), 'from_theta': lambda x: np.exp(x) / (1 + np.exp(x))
            }
        },
        'survival': False, 'discrete': False,
        'link': ["default", "logit", "loga", "cauchit", "probit", "cloglog", "ccloglog", "loglog", "robit", "sn"],
        'pdf': "zeroinflated"
    },
    'zeroinflatedbinomial2': {
        'doc': "Zero-inflated Binomial, type 2",
        'hyper': {
            'theta': {
                'hyperid': 92001, 'name': "alpha", 'short_name': "alpha",
                'output_name': "zero-probability parameter for zero-inflated binomial_2",
                'output_name_intern': "intern zero-probability parameter for zero-inflated binomial_2",
                'initial': -1, 'fixed': False, 'prior': "gaussian", 'param': [-1, 0.2],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            }
        },
        'survival': False, 'discrete': False,
        'link': ["default", "logit", "loga", "cauchit", "probit", "cloglog", "ccloglog", "loglog", "robit", "sn"],
        'pdf': "zeroinflated"
    },
    'zeroninflatedbinomial2': {
        'doc': "Zero and N inflated binomial, type 2",
        'hyper': {
            'theta1': {
                'hyperid': 93001, 'name': "alpha1", 'short_name': "alpha1",
                'output_name': "alpha1 parameter for zero-n-inflated binomial_2",
                'output_name_intern': "intern alpha1 parameter for zero-n-inflated binomial_2",
                'initial': -1, 'fixed': False, 'prior': "gaussian", 'param': [-1, 0.2],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 93002, 'name': "alpha2", 'short_name': "alpha2",
                'output_name': "alpha2 parameter for zero-n-inflated binomial_2",
                'output_name_intern': "intern alpha2 parameter for zero-n-inflated binomial_2",
                'initial': -1, 'fixed': False, 'prior': "gaussian", 'param': [-1, 0.2],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            }
        },
        'survival': False, 'discrete': False,
        'link': ["default", "logit", "loga", "cauchit", "probit", "cloglog", "ccloglog", "loglog", "robit", "sn"],
        'pdf': None
    },
    'zeroninflatedbinomial3': {
        'doc': "Zero and N inflated binomial, type 3",
        'hyper': {
            'theta1': {
                'hyperid': 93101, 'name': "alpha0", 'short_name': "alpha0",
                'output_name': "alpha0 parameter for zero-n-inflated binomial_3",
                'output_name_intern': "intern alpha0 parameter for zero-n-inflated binomial_3",
                'initial': 1, 'fixed': False, 'prior': "loggamma", 'param': [1, 1],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 93102, 'name': "alphaN", 'short_name': "alphaN",
                'output_name_intern': "intern alphaN parameter for zero-n-inflated binomial_3",
                'output_name': "alphaN parameter for zero-n-inflated binomial_3",
                'initial': 1, 'fixed': False, 'prior': "loggamma", 'param': [1, 1],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            }
        },
        'survival': False, 'discrete': False,
        'link': ["default", "logit", "loga", "cauchit", "probit", "cloglog", "ccloglog", "loglog", "robit", "sn"],
        'pdf': "zeroinflated"
    },
    'zeroinflatedbetabinomial2': {
        'doc': "Zero inflated Beta-Binomial, type 2",
        'hyper': {
            'theta1': {
                'hyperid': 94001, 'name': "log alpha", 'short_name': "a",
                'output_name': "zero-probability parameter for zero-inflated betabinomial_2",
                'output_name_intern': "intern zero-probability parameter for zero-inflated betabinomial_2",
                'initial': np.log(2), 'fixed': False, 'prior': "gaussian", 'param': [np.log(2), 1],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 94002, 'name': "beta", 'short_name': "b",
                'output_name': "overdispersion parameter for zero-inflated betabinomial_2",
                'output_name_intern': "intern overdispersion parameter for zero-inflated betabinomial_2",
                'initial': np.log(1), 'fixed': False, 'prior': "gaussian", 'param': [0, 1],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            }
        },
        'survival': False, 'discrete': False,
        'link': ["default", "logit", "loga", "cauchit", "probit", "cloglog", "ccloglog", "loglog", "robit", "sn"],
        'pdf': "zeroinflated"
    },
    'zeroinflatednbinomial0': {
        'doc': "Zero inflated negBinomial, type 0",
        'hyper': {
            'theta1': {
                'hyperid': 95001, 'name': "log size", 'short_name': "size",
                'output_name': "size for nbinomial_0 zero-inflated observations",
                'output_name_intern': "log size for nbinomial_0 zero-inflated observations",
                'initial': np.log(10), 'fixed': False, 'prior': "pcmgamma", 'param': [7],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 95002, 'name': "logit probability", 'short_name': "prob",
                'output_name': "zero-probability parameter for zero-inflated nbinomial_0",
                'output_name_intern': "intern zero-probability parameter for zero-inflated nbinomial_0",
                'initial': -1, 'fixed': False, 'prior': "gaussian", 'param': [-1, 0.2],
                'to_theta': lambda x: np.log(x / (1 - x)), 'from_theta': lambda x: np.exp(x) / (1 + np.exp(x))
            }
        },
        'survival': False, 'discrete': False, 'link': ["default", "log"], 'pdf': "zeroinflated"
    },
    'zeroinflatednbinomial1': {
        'doc': "Zero inflated negBinomial, type 1",
        'hyper': {
            'theta1': {
                'hyperid': 96001, 'name': "log size", 'short_name': "size",
                'output_name': "size for nbinomial_1 zero-inflated observations",
                'output_name_intern': "log size for nbinomial_1 zero-inflated observations",
                'initial': np.log(10), 'fixed': False, 'prior': "pcmgamma", 'param': [7],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 96002, 'name': "logit probability", 'short_name': "prob",
                'output_name': "zero-probability parameter for zero-inflated nbinomial_1",
                'output_name_intern': "intern zero-probability parameter for zero-inflated nbinomial_1",
                'initial': -1, 'fixed': False, 'prior': "gaussian", 'param': [-1, 0.2],
                'to_theta': lambda x: np.log(x / (1 - x)), 'from_theta': lambda x: np.exp(x) / (1 + np.exp(x))
            }
        },
        'survival': False, 'discrete': False, 'link': ["default", "log"], 'pdf': "zeroinflated"
    },
    'zeroinflatednbinomial1strata2': {
        'doc': "Zero inflated negBinomial, type 1, strata 2",
        'hyper': {
            'theta1': {
                'hyperid': 97001, 'name': "log size", 'short_name': "size",
                'output_name': "size for zero-inflated nbinomial_1_strata2",
                'output_name_intern': "log size for zero-inflated nbinomial_1_strata2",
                'initial': np.log(10), 'fixed': False, 'prior': "pcmgamma", 'param': [7],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            **{f'theta{i+1}': {
                'hyperid': 97001 + i, 'name': f"logit probability {i}", 'short_name': f"prob{i}",
                'output_name': f"zero-probability{i} for zero-inflated nbinomial_1_strata2",
                'output_name_intern': f"intern zero-probability{i} for zero-inflated nbinomial_1_strata2",
                'initial': -1, 'fixed': i > 2, 'prior': "gaussian", 'param': [-1, 0.2],
                'to_theta': lambda x: np.log(x / (1 - x)), 'from_theta': lambda x: np.exp(x) / (1 + np.exp(x))
            } for i in range(1, 11)}
        },
        'survival': False, 'discrete': False, 'link': ["default", "log"], 'pdf': "zeroinflated"
    },
    'zeroinflatednbinomial1strata3': {
        'doc': "Zero inflated negBinomial, type 1, strata 3",
        'hyper': {
            'theta1': {
                'hyperid': 98001, 'name': "logit probability", 'short_name': "prob",
                'output_name': "zero-probability for zero-inflated nbinomial_1_strata3",
                'output_name_intern': "intern zero-probability for zero-inflated nbinomial_1_strata3",
                'initial': -1, 'fixed': False, 'prior': "gaussian", 'param': [-1, 0.2],
                'to_theta': lambda x: np.log(x / (1 - x)), 'from_theta': lambda x: np.exp(x) / (1 + np.exp(x))
            },
            **{f'theta{i+1}': {
                'hyperid': 98001 + i, 'name': f"log size {i}", 'short_name': f"size{i}",
                'output_name': f"size{i} for zero-inflated nbinomial_1_strata3",
                'output_name_intern': f"log_size{i} for zero-inflated nbinomial_1_strata3",
                'initial': np.log(10), 'fixed': i > 2, 'prior': "pcmgamma", 'param': [7],
                'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            } for i in range(1, 11)}
        },
        'survival': False, 'discrete': False, 'link': ["default", "log"], 'pdf': "zeroinflated"
    }
}

        #likelihood_to_return = {'gaussian', 'poisson', 'binomial'}
        #subset_likelihood = {key: all_likelihoods[key] for key in likelihood_to_return}



def get_likelihood_models() -> dict:
    """Return all likelihood model definitions."""
    return LIKELIHOOD_MODELS

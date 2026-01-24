"""
Latent model definitions for INLA.

This module contains all available latent models (random effects) that can be
used with f() in INLA model formulas.

Each model is defined as a dictionary with the following keys:
- doc: Description of the model
- hyper: Hyperparameter specifications (dict of theta definitions)
- constr: Whether the model has a sum-to-zero constraint
- nrow_ncol: Whether the model uses nrow/ncol specifications
- augmented: Whether the model augments the latent field
- aug_factor: Augmentation factor
- aug_constr: Augmented constraint
- n_div_by: Division factor for n
- n_required: Whether n must be specified
- set_default_values: Whether to set default values
- pdf: Name of the PDF documentation
"""

import numpy as np

# Special number used as placeholder for unspecified hyperparameters
_SPECIAL_NUMBER = 1048576.0

LATENT_MODELS = {
    'linear': {
        'doc': "Alternative interface to a fixed effect",
        'hyper': {},
        'constr': False,
        'nrow_ncol': False,
        'augmented': False,
        'aug_factor': 1,
        'aug_constr': None,
        'n_div_by': None,
        'n_required': False,
        'set_default_values': False,
        'pdf': "linear"
    },
    'iid': {
        'doc': "Gaussian random effects in dim=1",
        'hyper': {
            'theta': {
                'hyperid': 1001,
                'name': "log precision",
                'short_name': "prec",
                'prior': "loggamma",
                'param': [1, 0.00005],
                'initial': 4,
                'fixed': False,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            }
        },
        'constr': False,
        'nrow_ncol': False,
        'augmented': False,
        'aug_factor': 1,
        'aug_constr': None,
        'n_div_by': None,
        'n_required': False,
        'set_default_values': False,
        'pdf': "indep"
    },
    'mec': {
        'doc': "Classical measurement error model",
        'hyper': {
            'theta1': {
                'hyperid': 2001,
                'name': "beta",
                'short_name': "b",
                'prior': "gaussian",
                'param': [1, 0.001],
                'initial': 1,
                'fixed': False,
                'to_theta': lambda x: x,
                'from_theta': lambda x: x
            },
            'theta2': {
                'hyperid': 2002,
                'name': "prec.u",
                'short_name': "prec",
                'prior': "loggamma",
                'param': [1, 0.0001],
                'initial': np.log(1 / 0.0001),
                'fixed': True,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            },
            'theta3': {
                'hyperid': 2003,
                'name': "mean.x",
                'short_name': "mu.x",
                'prior': "gaussian",
                'param': [0, 0.0001],
                'initial': 0,
                'fixed': True,
                'to_theta': lambda x: x,
                'from_theta': lambda x: x
            },
            'theta4': {
                'hyperid': 2004,
                'name': "prec.x",
                'short_name': "prec.x",
                'prior': "loggamma",
                'param': [1, 10000],
                'initial': np.log(1 / 10000),
                'fixed': True,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            }
        },
        'constr': False,
        'nrow_ncol': False,
        'augmented': False,
        'aug_factor': 1,
        'aug_constr': None,
        'n_div_by': None,
        'n_required': False,
        'set_default_values': False,
        'pdf': "mec"
    },
    'meb': {
        'doc': "Berkson measurement error model",
        'hyper': {
            'theta1': {
                'hyperid': 3001,
                'name': "beta",
                'short_name': "b",
                'prior': "gaussian",
                'param': [1, 0.001],
                'initial': 1,
                'fixed': False,
                'to_theta': lambda x: x,
                'from_theta': lambda x: x
            },
            'theta2': {
                'hyperid': 3002,
                'name': "prec.u",
                'short_name': "prec",
                'prior': "loggamma",
                'param': [1, 0.0001],
                'initial': np.log(1000),
                'fixed': False,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            }
        },
        'constr': False,
        'nrow_ncol': False,
        'augmented': False,
        'aug_factor': 1,
        'aug_constr': None,
        'n_div_by': None,
        'n_required': False,
        'set_default_values': False,
        'pdf': "meb"
    },
    'rgeneric': {
        'doc': "Generic latent model specified using R",
        'hyper': {},
        'constr': False,
        'nrow_ncol': False,
        'augmented': False,
        'aug_factor': 1,
        'aug_constr': None,
        'n_div_by': None,
        'n_required': True,
        'set_default_values': True,
        'pdf': "rgeneric"
    },
    'cgeneric': {
        'doc': "Generic latent model specified using C",
        'hyper': {},
        'constr': False,
        'nrow_ncol': False,
        'augmented': False,
        'aug_factor': 1,
        'aug_constr': None,
        'n_div_by': None,
        'n_required': True,
        'set_default_values': True,
        'pdf': "rgeneric"  # Note: This intentionally points to 'rgeneric' as in the R source
    },
    'rw1': {
        'doc': "Random walk of order 1",
        'hyper': {
            'theta': {
                'hyperid': 4001,
                'name': "log precision",
                'short_name': "prec",
                'prior': "loggamma",
                'param': [1, 0.00005],
                'initial': 4,
                'fixed': False,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            }
        },
        'constr': True,
        'nrow_ncol': False,
        'augmented': False,
        'aug_factor': 1,
        'aug_constr': None,
        'n_div_by': None,
        'n_required': False,
        'set_default_values': False,
        'min_diff': 1e-06,
        'pdf': "rw1"
    },
    'rw2': {
        'doc': "Random walk of order 2",
        'hyper': {
            'theta': {
                'hyperid': 5001,
                'name': "log precision",
                'short_name': "prec",
                'prior': "loggamma",
                'param': [1, 0.00005],
                'initial': 4,
                'fixed': False,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            }
        },
        'constr': True,
        'nrow_ncol': False,
        'augmented': False,
        'aug_factor': 1,
        'aug_constr': None,
        'n_div_by': None,
        'n_required': False,
        'set_default_values': False,
        'min_diff': 1e-04,
        'pdf': "rw2"
    },
    'crw2': {
        'doc': "Exact solution to the random walk of order 2",
        'hyper': {
            'theta': {
                'hyperid': 6001,
                'name': "log precision",
                'short_name': "prec",
                'prior': "loggamma",
                'param': [1, 0.00005],
                'initial': 4,
                'fixed': False,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            }
        },
        'constr': True,
        'nrow_ncol': False,
        'augmented': False,
        'aug_factor': 2,
        'aug_constr': 1,
        'n_div_by': None,
        'n_required': False,
        'set_default_values': False,
        'min_diff': 1e-04,
        'pdf': "crw2"
    },
    'seasonal': {
        'doc': "Seasonal model for time series",
        'hyper': {
            'theta': {
                'hyperid': 7001,
                'name': "log precision",
                'short_name': "prec",
                'prior': "loggamma",
                'param': [1, 0.00005],
                'initial': 4,
                'fixed': False,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            }
        },
        'constr': False,
        'nrow_ncol': False,
        'augmented': False,
        'aug_factor': 1,
        'aug_constr': None,
        'n_div_by': None,
        'n_required': False,
        'set_default_values': False,
        'pdf': "seasonal"
    },
    'besag': {
        'doc': "The Besag area model (CAR-model)",
        'hyper': {
            'theta': {
                'hyperid': 8001,
                'name': "log precision",
                'short_name': "prec",
                'prior': "loggamma",
                'param': [1, 0.00005],
                'initial': 4,
                'fixed': False,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            }
        },
        'constr': True,
        'nrow_ncol': False,
        'augmented': False,
        'aug_factor': 1,
        'aug_constr': None,
        'n_div_by': None,
        'n_required': True,
        'set_default_values': True,
        'pdf': "besag"
    },
    'besag2': {
        'doc': "The shared Besag model",
        'hyper': {
            'theta1': {
                'hyperid': 9001,
                'name': "log precision",
                'short_name': "prec",
                'prior': "loggamma",
                'param': [1, 0.00005],
                'initial': 4,
                'fixed': False,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 9002,
                'name': "scaling parameter",
                'short_name': "a",
                'prior': "loggamma",
                'param': [10, 10],
                'initial': 0,
                'fixed': False,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            }
        },
        'constr': True,
        'nrow_ncol': False,
        'augmented': False,
        'aug_factor': 1,
        'aug_constr': [1, 2],
        'n_div_by': 2,
        'n_required': True,
        'set_default_values': True,
        'pdf': "besag2"
    },
    'bym': {
        'doc': "The BYM-model (Besag-York-Mollier model)",
        'hyper': {
            'theta1': {
                'hyperid': 10001,
                'name': "log unstructured precision",
                'short_name': "prec.unstruct",
                'prior': "loggamma",
                'param': [1, 0.0005],
                'initial': 4,
                'fixed': False,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 10002,
                'name': "log spatial precision",
                'short_name': "prec.spatial",
                'prior': "loggamma",
                'param': [1, 0.0005],
                'initial': 4,
                'fixed': False,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            }
        },
        'constr': True,
        'nrow_ncol': False,
        'augmented': True,
        'aug_factor': 2,
        'aug_constr': 2,
        'n_div_by': None,
        'n_required': True,
        'set_default_values': True,
        'pdf': "bym"
    },
    'bym2': {
        'doc': "The BYM-model with the PC priors",
        'hyper': {
            'theta1': {
                'hyperid': 11001,
                'name': "log precision",
                'short_name': "prec",
                'prior': "pc.prec",
                'param': [1, .01],
                'initial': 4,
                'fixed': False,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 11002,
                'name': "logit phi",
                'short_name': "phi",
                'prior': "pc",
                'param': [0.5, 0.5],
                'initial': -3,
                'fixed': False,
                'to_theta': lambda x: np.log(x / (1 - x)),
                'from_theta': lambda x: np.exp(x) / (1 + np.exp(x))
            }
        },
        'constr': True,
        'nrow_ncol': False,
        'augmented': True,
        'aug_factor': 2,
        'aug_constr': 2,
        'n_div_by': None,
        'n_required': True,
        'set_default_values': True,
        'pdf': "bym2"
    },
    'besagproper': {
        'doc': "A proper version of the Besag model",
        'hyper': {
            'theta1': {
                'hyperid': 12001,
                'name': "log precision",
                'short_name': "prec",
                'prior': "loggamma",
                'param': [1, 0.0005],
                'initial': 2,
                'fixed': False,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 12002,
                'name': "log diagonal",
                'short_name': "diag",
                'prior': "loggamma",
                'param': [1, 1],
                'initial': 1,
                'fixed': False,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            }
        },
        'constr': False,
        'nrow_ncol': False,
        'augmented': False,
        'aug_factor': 1,
        'aug_constr': None,
        'n_div_by': None,
        'n_required': True,
        'set_default_values': True,
        'pdf': "besagproper"
    },
    'besagproper2': {
        'doc': "An alternative proper version of the Besag model",
        'hyper': {
            'theta1': {
                'hyperid': 13001,
                'name': "log precision",
                'short_name': "prec",
                'prior': "loggamma",
                'param': [1, 0.0005],
                'initial': 2,
                'fixed': False,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 13002,
                'name': "logit lambda",
                'short_name': "lambda",
                'prior': "gaussian",
                'param': [0, 0.45],
                'initial': 3,
                'fixed': False,
                'to_theta': lambda x: np.log(x / (1 - x)),
                'from_theta': lambda x: np.exp(x) / (1 + np.exp(x))
            }
        },
        'constr': False,
        'nrow_ncol': False,
        'augmented': False,
        'aug_factor': 1,
        'aug_constr': None,
        'n_div_by': None,
        'n_required': True,
        'set_default_values': True,
        'pdf': "besagproper2"
    },
    'fgn': {
        'doc': "Fractional Gaussian noise model",
        'hyper': {
            'theta1': {
                'hyperid': 13101,
                'name': "log precision",
                'short_name': "prec",
                'prior': "pc.prec",
                'param': [3, 0.01],
                'initial': 1,
                'fixed': False,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 13102,
                'name': "logit H",
                'short_name': "H",
                'prior': "pcfgnh",
                'param': [0.9, 0.1],
                'initial': 2,
                'fixed': False,
                'to_theta': lambda x: np.log((2 * x - 1) / (2 * (1 - x))),
                'from_theta': lambda x: 0.5 + 0.5 * np.exp(x) / (1 + np.exp(x))
            }
        },
        'constr': False,
        'nrow_ncol': False,
        'augmented': True,
        'aug_factor': 5,
        'aug_constr': 1,
        'n_div_by': None,
        'n_required': False,
        'set_default_values': True,
        'order_default': 4,
        'order_defined': list(range(3, 5)), # R's 3L:4L becomes list(range(3, 5))
        'pdf': "fgn"
    },
    'fgn2': {
        'doc': "Fractional Gaussian noise model (alt 2)",
        'hyper': {
            'theta1': {
                'hyperid': 13111,
                'name': "log precision",
                'short_name': "prec",
                'prior': "pc.prec",
                'param': [3, 0.01],
                'initial': 1,
                'fixed': False,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 13112,
                'name': "logit H",
                'short_name': "H",
                'prior': "pcfgnh",
                'param': [0.9, 0.1],
                'initial': 2,
                'fixed': False,
                'to_theta': lambda x: np.log((2 * x - 1) / (2 * (1 - x))),
                'from_theta': lambda x: 0.5 + 0.5 * np.exp(x) / (1 + np.exp(x))
            }
        },
        'constr': False,
        'nrow_ncol': False,
        'augmented': True,
        'aug_factor': 4,
        'aug_constr': 1,
        'n_div_by': None,
        'n_required': False,
        'set_default_values': True,
        'order_default': 4,
        'order_defined': list(range(3, 5)),
        'pdf': "fgn"
    },
    'ar1': {
        'doc': "Auto-regressive model of order 1 (AR(1))",
        'hyper': {
            'theta1': {
                'hyperid': 14001,
                'name': "log precision",
                'short_name': "prec",
                'prior': "loggamma",
                'param': [1, 0.00005],
                'initial': 4,
                'fixed': False,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 14002,
                'name': "logit lag one correlation",
                'short_name': "rho",
                'prior': "normal",
                'param': [0, 0.15],
                'initial': 2,
                'fixed': False,
                'to_theta': lambda x: np.log((1 + x) / (1 - x)),
                'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1
            },
            'theta3': {
                'hyperid': 14003,
                'name': "mean",
                'short_name': "mean",
                'prior': "normal",
                'param': [0, 1],
                'initial': 0,
                'fixed': True,
                'to_theta': lambda x: x,
                'from_theta': lambda x: x
            }
        },
        'constr': False,
        'nrow_ncol': False,
        'augmented': False,
        'aug_factor': 1,
        'aug_constr': None,
        'n_div_by': None,
        'n_required': False,
        'set_default_values': False,
        'pdf': "ar1"
    },
    'ar1c': {
        'doc': "Auto-regressive model of order 1 w/covariates",
        'hyper': {
            'theta1': {
                'hyperid': 14101,
                'name': "log precision",
                'short_name': "prec",
                'prior': "pc.prec",
                'param': [1, 0.01],
                'initial': 4,
                'fixed': False,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 14102,
                'name': "logit lag one correlation",
                'short_name': "rho",
                'prior': "pc.cor0",
                'param': [0.5, 0.5],
                'initial': 2,
                'fixed': False,
                'to_theta': lambda x: np.log((1 + x) / (1 - x)),
                'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1
            }
        },
        'constr': False,
        'nrow_ncol': False,
        'augmented': False,
        'aug_factor': 1,
        'aug_constr': None,
        'n_div_by': None,
        'n_required': False,
        'set_default_values': True,
        'pdf': "ar1c"
    },
    'ar': {
        'doc': "Auto-regressive model of order p (AR(p))",
        'hyper': {
            'theta1': {
                'hyperid': 15001,
                'name': "log precision",
                'short_name': "prec",
                'initial': 4,
                'fixed': False,
                'prior': "pc.prec",
                'param': [3, 0.01],
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            },
            **(lambda _:
                (lambda pacf_scales:
                    {
                        f'theta{i}': {
                            'hyperid': 15000 + i,
                            'name': f"pacf{i-1}",
                            'short_name': f"pacf{i-1}",
                            'initial': 1 if i == 2 else 0,   # pacf1 often starts at 1 here; keep whatever you use consistently
                            'fixed': False,
                            'prior': "pc.cor0",
                            'param': [0.5, pacf_scales[i-2]],
                            'to_theta': lambda x: np.log((1 + x) / (1 - x)),
                            'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1
                        }
                        for i in range(2, 12)  # theta2..theta11 -> pacf1..pacf10
                    }
                )([0.5, 0.4, 0.3, 0.2] + [0.1] * 6)
            )(None)
        },
        'constr': False,
        'nrow_ncol': False,
        'augmented': False,
        'aug_factor': 1,
        'aug_constr': None,
        'n_div_by': None,
        'n_required': False,
        'set_default_values': False,
        'pdf': "ar"
    },
    'ou': {
        'doc': "The Ornstein-Uhlenbeck process",
        'hyper': {
            'theta1': {
                'hyperid': 16001,
                'name': "log precision",
                'short_name': "prec",
                'prior': "loggamma",
                'param': [1, 0.00005],
                'initial': 4,
                'fixed': False,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 16002,
                'name': "log phi",
                'short_name': "phi",
                'prior': "normal",
                'param': [0, 0.2],
                'initial': -1,
                'fixed': False,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            }
        },
        'constr': False,
        'nrow_ncol': False,
        'augmented': False,
        'aug_factor': 1,
        'aug_constr': None,
        'n_div_by': None,
        'n_required': False,
        'set_default_values': False,
        'pdf': "ou"
    },
    'intslope': {
        'doc': "Intecept-slope model with Wishart-prior",
        'hyper': {
            # First three thetas are unique
            'theta1': {
                'hyperid': 16101, 
                'name': "log precision1", 
                'short_name': "prec1", 
                'initial': 4, 
                'fixed': False,
                'prior': "wishart2d", 
                'param': [4, 1, 1, 0], 
                'to_theta': lambda x: np.log(x), 
                'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 16102, 
                'name': "log precision2", 
                'short_name': "prec2", 
                'initial': 4, 
                'fixed': False,
                'prior': "none", 
                'param': [], 
                'to_theta': lambda x: np.log(x), 
                'from_theta': lambda x: np.exp(x)
            },
            'theta3': {
                'hyperid': 16103, 
                'name': "logit correlation", 
                'short_name': "cor", 
                'initial': 4, 
                'fixed': False,
                'prior': "none", 
                'param': [], 
                'to_theta': lambda x: np.log((1 + x) / (1 - x)), 
                'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1
            },
            # The next 50 thetas (gamma1 to gamma50) are identical in structure
            **{f'theta{i+4}': {
                'hyperid': 16101 + i + 3,
                'name': f"gamma{i+1}",
                'short_name': f"g{i+1}",
                'initial': 1,
                'fixed': True,
                'prior': "normal",
                'param': [1, 36],
                'to_theta': lambda x: x,
                'from_theta': lambda x: x
            } for i in range(50)}
        },
        'constr': False,
        'nrow_ncol': False,
        'augmented': False,
        'aug_factor': 1,
        'aug_constr': None,
        'n_div_by': None,
        'n_required': False,
        'set_default_values': True,
        'pdf': "intslope"
    },
    'generic': {
        'doc': "A generic model",
        'hyper': {
            'theta': {
                'hyperid': 17001, 
                'name': "log precision", 
                'short_name': "prec", 
                'prior': "loggamma", 
                'param': [1, 0.00005],
                'initial': 4, 
                'fixed': False, 
                'to_theta': lambda x: np.log(x), 
                'from_theta': lambda x: np.exp(x)
            }
        },
        'constr': False, 
        'nrow_ncol': False, 
        'augmented': False, 
        'aug_factor': 1, 
        'aug_constr': None,
        'n_div_by': None, 
        'n_required': True, 
        'set_default_values': True, 
        'pdf': "generic0"
    },
    'generic0': {
        'doc': "A generic model (type 0)",
        'hyper': {
            'theta': {
                'hyperid': 18001, 
                'name': "log precision", 
                'short_name': "prec", 
                'prior': "loggamma", 
                'param': [1, 0.00005],
                'initial': 4, 
                'fixed': False, 
                'to_theta': lambda x: np.log(x), 
                'from_theta': lambda x: np.exp(x)
            }
        },
        'constr': False, 
        'nrow_ncol': False, 
        'augmented': False, 
        'aug_factor': 1, 
        'aug_constr': None,
        'n_div_by': None, 
        'n_required': True, 
        'set_default_values': True, 
        'pdf': "generic0"
    },
    'generic1': {
        'doc': "A generic model (type 1)",
        'hyper': {
            'theta1': {
                'hyperid': 19001, 
                'name': "log precision", 
                'short_name': "prec", 
                'prior': "loggamma", 
                'param': [1, 0.00005],
                'initial': 4, 
                'fixed': False, 
                'to_theta': lambda x: np.log(x), 
                'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 19002, 
                'name': "beta", 
                'short_name': "beta", 
                'initial': 2, 
                'fixed': False,
                'prior': "gaussian", 
                'param': [0, 0.1], 
                'to_theta': lambda x: np.log(x / (1 - x)), 
                'from_theta': lambda x: np.exp(x) / (1 + np.exp(x))
            }
        },
        'constr': False, 
        'nrow_ncol': False, 
        'augmented': False, 
        'aug_factor': 1, 
        'aug_constr': None,
        'n_div_by': None, 
        'n_required': True, 
        'set_default_values': True, 
        'pdf': "generic1"
    },
    'generic2': {
        'doc': "A generic model (type 2)",
        'hyper': {
            'theta1': {
                'hyperid': 20001,
                'name': "log precision cmatrix",
                'short_name': "prec",
                'initial': 4,
                'fixed': False,
                'prior': "loggamma",
                'param': [1, 0.00005],
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 20002,
                'name': "log precision random",
                'short_name': "prec.random",
                'initial': 4,
                'fixed': False,
                'prior': "loggamma",
                'param': [1, 0.001],
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            }
        },
        'constr': False,
        'nrow_ncol': False,
        'augmented': True,
        'aug_factor': 2,
        'aug_constr': 2,
        'n_div_by': None,
        'n_required': True,
        'set_default_values': True,
        'pdf': "generic2"
    },
    'generic3': {
        'doc': "A generic model (type 3)",
        'hyper': {
            f'theta{i}': {
                'hyperid': 21000 + i,
                'name': f"log precision{i if i <= 10 else ' common'}",
                'short_name': f"prec{i if i <= 10 else '.common'}",
                'initial': 4 if i <= 10 else 0,
                'fixed': i > 10,
                'prior': "loggamma",
                'param': [1, 0.00005],
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            } for i in range(1, 12)
        },
        'constr': False, 'nrow_ncol': False, 'augmented': False, 'aug_factor': 1, 'aug_constr': None,
        'n_div_by': None, 'n_required': True, 'set_default_values': True, 'pdf': "generic3"
    },
    'spde': {
        'doc': "A SPDE model",
        'hyper': {
            'theta1': {
                'hyperid': 22001,
                'name': "theta.T", 
                'short_name': "T", 
                'initial': 2, 
                'fixed': False,
                'prior': "normal", 
                'param': [0, 1], 
                'to_theta': lambda x: x, 
                'from_theta': lambda x: x
            },
            'theta2': {
                'hyperid': 22002,
                'name': "theta.K",
                'short_name': "K",
                'initial': -2,
                'fixed': False,
                'prior': "normal",
                'param': [0, 1],
                'to_theta': lambda x: x,
                'from_theta': lambda x: x
            },
            'theta3': {
                'hyperid': 22003,
                'name': "theta.KT",
                'short_name': "KT",
                'initial': 0,
                'fixed': False,
                'prior': "normal",
                'param': [0, 1],
                'to_theta': lambda x: x,
                'from_theta': lambda x: x
            },
            'theta4': {
                'hyperid': 22004,
                'name': "theta.OC",
                'short_name': "OC",
                'initial': -20,
                'fixed': True,
                'prior': "normal",
                'param': [0, 0.2],
                'to_theta': lambda x: np.log(x / (1 - x)),
                'from_theta': lambda x: np.exp(x) / (1 + np.exp(x))
            }
        },
        'constr': False,
        'nrow_ncol': False,
        'augmented': False,
        'aug_factor': 1,
        'aug_constr': None,
        'n_div_by': None,
        'n_required': True,
        'set_default_values': True, 
        'pdf': "spde"
    },
    'spde2': {
        'doc': "A SPDE2 model",
        'hyper': {
            'theta1': {
                'hyperid': 23001, 
                'name': "theta1", 
                'short_name': "t1", 
                'initial': 0, 
                'fixed': False,
                'prior': "mvnorm", 
                'param': [1, 1], 
                'to_theta': lambda x: x, 
                'from_theta': lambda x: x
            },
            **{f'theta{i}': {
                'hyperid': 23000 + i,
                'name': f"theta{i}",
                'short_name': f"t{i}",
                'initial': 0,
                'fixed': False,
                'prior': "none",
                'param': [],
                'to_theta': lambda x: x,
                'from_theta': lambda x: x
            } for i in range(2, 101)}
        },
        'constr': False, 
        'nrow_ncol': False, 
        'augmented': False, 
        'aug_factor': 1, 
        'aug_constr': None,
        'n_div_by': None, 
        'n_required': True, 
        'set_default_values': True, 
        'pdf': "spde2"
    },
    'generic': {
        'doc': "A generic model",
        'hyper': {
            'theta': {
                'hyperid': 17001, 
                'name': "log precision", 
                'short_name': "prec", 
                'prior': "loggamma", 
                'param': [1, 0.00005],
                'initial': 4, 
                'fixed': False, 
                'to_theta': lambda x: np.log(x), 
                'from_theta': lambda x: np.exp(x)
            }
        },
        'constr': False, 
        'nrow_ncol': False, 
        'augmented': False, 
        'aug_factor': 1, 
        'aug_constr': None,
        'n_div_by': None, 
        'n_required': True, 
        'set_default_values': True, 
        'pdf': "generic0"
    },
    'generic0': {
        'doc': "A generic model (type 0)",
        'hyper': {
            'theta': {
                'hyperid': 18001, 
                'name': "log precision", 
                'short_name': "prec", 
                'prior': "loggamma", 
                'param': [1, 0.00005],
                'initial': 4, 
                'fixed': False, 
                'to_theta': lambda x: np.log(x), 
                'from_theta': lambda x: np.exp(x)
            }
        },
        'constr': False, 
        'nrow_ncol': False, 
        'augmented': False, 
        'aug_factor': 1, 
        'aug_constr': None,
        'n_div_by': None, 
        'n_required': True, 
        'set_default_values': True, 
        'pdf': "generic0"
    },
    'generic1': {
        'doc': "A generic model (type 1)",
        'hyper': {
            'theta1': {
                'hyperid': 19001, 
                'name': "log precision", 
                'short_name': "prec", 
                'prior': "loggamma", 
                'param': [1, 0.00005],
                'initial': 4, 
                'fixed': False, 
                'to_theta': lambda x: np.log(x), 
                'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 19002, 
                'name': "beta", 
                'short_name': "beta", 
                'initial': 2, 
                'fixed': False,
                'prior': "gaussian", 
                'param': [0, 0.1], 
                'to_theta': lambda x: np.log(x / (1 - x)), 
                'from_theta': lambda x: np.exp(x) / (1 + np.exp(x))
            }
        },
        'constr': False, 
        'nrow_ncol': False, 
        'augmented': False, 
        'aug_factor': 1, 
        'aug_constr': None,
        'n_div_by': None, 
        'n_required': True, 
        'set_default_values': True, 
        'pdf': "generic1"
    },
    'generic2': {
        'doc': "A generic model (type 2)",
        'hyper': {
            'theta1': {
                'hyperid': 20001, 
                'name': "log precision cmatrix", 
                'short_name': "prec", 
                'initial': 4, 
                'fixed': False,
                'prior': "loggamma", 
                'param': [1, 0.00005], 
                'to_theta': lambda x: np.log(x), 
                'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 20002, 
                'name': "log precision random", 
                'short_name': "prec.random", 
                'initial': 4, 
                'fixed': False,
                'prior': "loggamma", 
                'param': [1, 0.001], 
                'to_theta': lambda x: np.log(x), 
                'from_theta': lambda x: np.exp(x)
            }
        },
        'constr': False,
        'nrow_ncol': False,
        'augmented': True,
        'aug_factor': 2,
        'aug_constr': 2,
        'n_div_by': None,
        'n_required': True,
        'set_default_values': True,
        'pdf': "generic2"
    },
    'generic3': {
        'doc': "A generic model (type 3)",
        'hyper': {
            f'theta{i}': {
                'hyperid': 21000 + i,
                'name': f"log precision{i if i <= 10 else ' common'}",
                'short_name': f"prec{i if i <= 10 else '.common'}",
                'initial': 4 if i <= 10 else 0,
                'fixed': i > 10,
                'prior': "loggamma",
                'param': [1, 0.00005],
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            } for i in range(1, 12)
        },
        'constr': False, 'nrow_ncol': False, 'augmented': False, 'aug_factor': 1, 'aug_constr': None,
        'n_div_by': None, 'n_required': True, 'set_default_values': True, 'pdf': "generic3"
    },
    'spde': {
        'doc': "A SPDE model",
        'hyper': {
            'theta1': {
                'hyperid': 22001, 'name': "theta.T", 'short_name': "T", 'initial': 2, 'fixed': False,
                'prior': "normal", 'param': [0, 1], 'to_theta': lambda x: x, 'from_theta': lambda x: x
            },
            'theta2': {
                'hyperid': 22002, 'name': "theta.K", 'short_name': "K", 'initial': -2, 'fixed': False,
                'prior': "normal", 'param': [0, 1], 'to_theta': lambda x: x, 'from_theta': lambda x: x
            },
            'theta3': {
                'hyperid': 22003, 'name': "theta.KT", 'short_name': "KT", 'initial': 0, 'fixed': False,
                'prior': "normal", 'param': [0, 1], 'to_theta': lambda x: x, 'from_theta': lambda x: x
            },
            'theta4': {
                'hyperid': 22004, 'name': "theta.OC", 'short_name': "OC", 'initial': -20, 'fixed': True,
                'prior': "normal", 'param': [0, 0.2], 'to_theta': lambda x: np.log(x / (1 - x)), 'from_theta': lambda x: np.exp(x) / (1 + np.exp(x))
            }
        },
        'constr': False, 
        'nrow_ncol': False, 
        'augmented': False, 
        'aug_factor': 1, 
        'aug_constr': None,
        'n_div_by': None, 
        'n_required': True, 
        'set_default_values': True, 
        'pdf': "spde"
    },
    'spde2': {
        'doc': "A SPDE2 model",
        'hyper': {
            'theta1': {
                'hyperid': 23001, 
                'name': "theta1", 
                'short_name': "t1", 
                'initial': 0, 
                'fixed': False,
                'prior': "mvnorm", 
                'param': [1, 1], 
                'to_theta': lambda x: x, 
                'from_theta': lambda x: x
            },
            **{f'theta{i}': {
                'hyperid': 23000 + i,
                'name': f"theta{i}",
                'short_name': f"t{i}",
                'initial': 0,
                'fixed': False,
                'prior': "none",
                'param': [],
                'to_theta': lambda x: x,
                'from_theta': lambda x: x
            } for i in range(2, 101)}
        },
        'constr': False, 
        'nrow_ncol': False, 
        'augmented': False, 
        'aug_factor': 1, 
        'aug_constr': None,
        'n_div_by': None, 
        'n_required': True, 
        'set_default_values': True, 
        'pdf': "spde2"
    },
    'spde3': {
        'doc': "A SPDE3 model",
        'hyper': {
            'theta1': {
                'hyperid': 24001, 'name': "theta1", 'short_name': "t1", 'initial': 0, 'fixed': False,
                'prior': "mvnorm", 'param': [1, 1], 'to_theta': lambda x: x, 'from_theta': lambda x: x
            },
            **{f'theta{i}': {
                'hyperid': 24000 + i,
                'name': f"theta{i}",
                'short_name': f"t{i}",
                'initial': 0,
                'fixed': False,
                'prior': "none",
                'param': [],
                'to_theta': lambda x: x,
                'from_theta': lambda x: x
            } for i in range(2, 101)}
        },
        'constr': False, 'nrow_ncol': False, 'augmented': False, 'aug_factor': 1, 'aug_constr': None,
        'n_div_by': None, 'n_required': True, 'set_default_values': True, 'pdf': "spde3"
    },
    'iid1d': {
        'doc': "Gaussian random effect in dim=1 with Wishart prior",
        'hyper': {
            'theta': {
                'hyperid': 25001,
                'name': "precision",
                'short_name': "prec",
                'initial': 4,
                'fixed': False,
                'prior': "wishart1d",
                'param': [2.0, 2.0 * 0.00005],
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            }
        },
        'constr': False, 'nrow_ncol': False, 'augmented': False, 'aug_factor': 1, 'aug_constr': None,
        'n_div_by': None, 'n_required': False, 'set_default_values': True, 'pdf': "iid123d"
    },
    'iid2d': {
        'doc': "Gaussian random effect in dim=2 with Wishart prior",
        'hyper': {
            'theta1': {
                'hyperid': 26001, 'name': "log precision1", 'short_name': "prec1", 'initial': 4, 'fixed': False,
                'prior': "wishart2d", 'param': [4, 1, 1, 0], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 26002, 'name': "log precision2", 'short_name': "prec2", 'initial': 4, 'fixed': False,
                'prior': "none", 'param': [], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            'theta3': {
                'hyperid': 26003, 'name': "logit correlation", 'short_name': "cor", 'initial': 4, 'fixed': False,
                'prior': "none", 'param': [], 'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1
            }
        },
        'constr': False, 'nrow_ncol': False, 'augmented': True, 'aug_factor': 1,
        'aug_constr': [1, 2], 'n_div_by': 2, 'n_required': True, 'set_default_values': True, 'pdf': "iid123d"
    },
    'iid3d': {
        'doc': "Gaussian random effect in dim=3 with Wishart prior",
        'hyper': {
            'theta1': {
                'hyperid': 27001, 'name': "log precision1", 'short_name': "prec1", 'initial': 4, 'fixed': False,
                'prior': "wishart3d", 'param': [7, 1, 1, 1, 0, 0, 0], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 27002, 'name': "log precision2", 'short_name': "prec2", 'initial': 4, 'fixed': False,
                'prior': "none", 'param': [], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            'theta3': {
                'hyperid': 27003, 'name': "log precision3", 'short_name': "prec3", 'initial': 4, 'fixed': False,
                'prior': "none", 'param': [], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            'theta4': {
                'hyperid': 27004, 'name': "logit correlation12", 'short_name': "cor12", 'initial': 0, 'fixed': False,
                'prior': "none", 'param': [], 'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1
            },
            'theta5': {
                'hyperid': 27005, 'name': "logit correlation13", 'short_name': "cor13", 'initial': 0, 'fixed': False,
                'prior': "none", 'param': [], 'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1
            },
            'theta6': {
                'hyperid': 27006, 'name': "logit correlation23", 'short_name': "cor23", 'initial': 0, 'fixed': False,
                'prior': "none", 'param': [], 'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1
            }
        },
        'constr': False, 'nrow_ncol': False, 'augmented': True, 'aug_factor': 1,
        'aug_constr': [1, 2, 3], 'n_div_by': 3, 'n_required': True, 'set_default_values': True, 'pdf': "iid123d"
    },
    'iid4d': {
        'doc': "Gaussian random effect in dim=4 with Wishart prior",
        'hyper': {
            'theta1': {
                'hyperid': 28001, 'name': "log precision1", 'short_name': "prec1", 'initial': 4, 'fixed': False,
                'prior': "wishart4d", 'param': [11, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 28002, 'name': "log precision2", 'short_name': "prec2", 'initial': 4, 'fixed': False,
                'prior': "none", 'param': [], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            'theta3': {
                'hyperid': 28003, 'name': "log precision3", 'short_name': "prec3", 'initial': 4, 'fixed': False,
                'prior': "none", 'param': [], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            'theta4': {
                'hyperid': 28004, 'name': "log precision4", 'short_name': "prec4", 'initial': 4, 'fixed': False,
                'prior': "none", 'param': [], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            'theta5': {
                'hyperid': 28005, 'name': "logit correlation12", 'short_name': "cor12", 'initial': 0, 'fixed': False,
                'prior': "none", 'param': [], 'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1
            },
            'theta6': {
                'hyperid': 28006, 'name': "logit correlation13", 'short_name': "cor13", 'initial': 0, 'fixed': False,
                'prior': "none", 'param': [], 'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1
            },
            'theta7': {
                'hyperid': 28007, 'name': "logit correlation14", 'short_name': "cor14", 'initial': 0, 'fixed': False,
                'prior': "none", 'param': [], 'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1
            },
            'theta8': {
                'hyperid': 28008, 'name': "logit correlation23", 'short_name': "cor23", 'initial': 0, 'fixed': False,
                'prior': "none", 'param': [], 'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1
            },
            'theta9': {
                'hyperid': 28009, 'name': "logit correlation24", 'short_name': "cor24", 'initial': 0, 'fixed': False,
                'prior': "none", 'param': [], 'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1
            },
            'theta10': {
                'hyperid': 28010, 'name': "logit correlation34", 'short_name': "cor34", 'initial': 0, 'fixed': False,
                'prior': "none", 'param': [], 'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1
            }
        },
        'constr': False, 'nrow_ncol': False, 'augmented': True, 'aug_factor': 1,
        'aug_constr': [1, 2, 3, 4], 'n_div_by': 4, 'n_required': True, 'set_default_values': True, 'pdf': "iid123d"
    },
    'iid5d': {
        'doc': "Gaussian random effect in dim=5 with Wishart prior",
        'hyper': {
            'theta1': {
                'hyperid': 29001, 'name': "log precision1", 'short_name': "prec1", 'initial': 4, 'fixed': False,
                'prior': "wishart5d", 'param': [16, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            'theta2': {'hyperid': 29002, 'name': "log precision2", 'short_name': "prec2", 'initial': 4, 'fixed': False, 'prior': "none", 'param': [], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)},
            'theta3': {'hyperid': 29003, 'name': "log precision3", 'short_name': "prec3", 'initial': 4, 'fixed': False, 'prior': "none", 'param': [], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)},
            'theta4': {'hyperid': 29004, 'name': "log precision4", 'short_name': "prec4", 'initial': 4, 'fixed': False, 'prior': "none", 'param': [], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)},
            'theta5': {'hyperid': 29005, 'name': "log precision5", 'short_name': "prec5", 'initial': 4, 'fixed': False, 'prior': "none", 'param': [], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)},
            'theta6': {'hyperid': 29006, 'name': "logit correlation12", 'short_name': "cor12", 'initial': 0, 'fixed': False, 'prior': "none", 'param': [], 'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1},
            'theta7': {'hyperid': 29007, 'name': "logit correlation13", 'short_name': "cor13", 'initial': 0, 'fixed': False, 'prior': "none", 'param': [], 'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1},
            'theta8': {'hyperid': 29008, 'name': "logit correlation14", 'short_name': "cor14", 'initial': 0, 'fixed': False, 'prior': "none", 'param': [], 'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1},
            'theta9': {'hyperid': 29009, 'name': "logit correlation15", 'short_name': "cor15", 'initial': 0, 'fixed': False, 'prior': "none", 'param': [], 'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1},
            'theta10': {'hyperid': 29010, 'name': "logit correlation23", 'short_name': "cor23", 'initial': 0, 'fixed': False, 'prior': "none", 'param': [], 'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1},
            'theta11': {'hyperid': 29011, 'name': "logit correlation24", 'short_name': "cor24", 'initial': 0, 'fixed': False, 'prior': "none", 'param': [], 'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1},
            'theta12': {'hyperid': 29012, 'name': "logit correlation25", 'short_name': "cor25", 'initial': 0, 'fixed': False, 'prior': "none", 'param': [], 'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1},
            'theta13': {'hyperid': 29013, 'name': "logit correlation34", 'short_name': "cor34", 'initial': 0, 'fixed': False, 'prior': "none", 'param': [], 'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1},
            'theta14': {'hyperid': 29014, 'name': "logit correlation35", 'short_name': "cor35", 'initial': 0, 'fixed': False, 'prior': "none", 'param': [], 'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1},
            'theta15': {'hyperid': 29015, 'name': "logit correlation45", 'short_name': "cor45", 'initial': 0, 'fixed': False, 'prior': "none", 'param': [], 'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1}
        },
        'constr': False, 'nrow_ncol': False, 'augmented': True, 'aug_factor': 1,
        'aug_constr': [1, 2, 3, 4, 5], 'n_div_by': 5, 'n_required': True, 'set_default_values': True, 'pdf': "iid123d"
    },
    'iidkd': {
        'doc': "Gaussian random effect in dim=k with Wishart prior",
        'hyper': {
            'theta1': {
                'hyperid': 29101, 'name': "theta1", 'short_name': "theta1", 'initial': _SPECIAL_NUMBER, 'fixed': False,
                'prior': "wishartkd", 'param': [30] + [_SPECIAL_NUMBER] * int((24*25)/2), 'to_theta': lambda x: x, 'from_theta': lambda x: x
            },
            **{f'theta{i}': {
                'hyperid': 29100 + i,
                'name': f"theta{i}",
                'short_name': f"theta{i}",
                'initial': _SPECIAL_NUMBER,
                'fixed': False,
                'prior': "none",
                'param': [],
                'to_theta': lambda x: x,
                'from_theta': lambda x: x
            } for i in range(2, 301)}
        },
        'constr': False, 'nrow_ncol': False, 'augmented': True, 'aug_factor': 1,
        'aug_constr': list(range(1, 25)), 'n_div_by': -1, 'n_required': True, 'set_default_values': True, 'pdf': "iidkd"
    },
    '2diid': {
        'doc': "(This model is obsolete)",
        'hyper': {
            'theta1': {
                'hyperid': 30001, 'name': "log precision1", 'short_name': "prec1", 'initial': 4, 'fixed': False,
                'prior': "loggamma", 'param': [1, 0.00005], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 30002, 'name': "log precision2", 'short_name': "prec2", 'initial': 4, 'fixed': False,
                'prior': "loggamma", 'param': [1, 0.00005], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            'theta3': {
                'hyperid': 30003, 'name': "correlation", 'short_name': "cor", 'initial': 4, 'fixed': False,
                'prior': "normal", 'param': [0, 0.15], 'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1
            }
        },
        'constr': False, 'nrow_ncol': False, 'augmented': False, 'aug_factor': 1,
        'aug_constr': [1, 2], 'n_div_by': 2, 'n_required': True, 'set_default_values': True, 'pdf': "iid123d"
    },
    'z': {
        'doc': "The z-model in a classical mixed model formulation",
        'hyper': {
            'theta': {
                'hyperid': 31001, 'name': "log precision", 'short_name': "prec", 'initial': 4, 'fixed': False,
                'prior': "loggamma", 'param': [1, 0.00005], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            }
        },
        'constr': False, 'nrow_ncol': False, 'augmented': False, 'aug_factor': 1, 'aug_constr': None,
        'n_div_by': None, 'n_required': True, 'set_default_values': True, 'pdf': "z"
    },
    'rw2d': {
        'doc': "Thin-plate spline model",
        'hyper': {
            'theta': {
                'hyperid': 32001, 'name': "log precision", 'short_name': "prec", 'initial': 4, 'fixed': False,
                'prior': "loggamma", 'param': [1, 0.00005], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            }
        },
        'constr': True, 'nrow_ncol': True, 'augmented': False, 'aug_factor': 1, 'aug_constr': None,
        'n_div_by': None, 'n_required': False, 'set_default_values': True, 'pdf': "rw2d"
    },
    'rw2diid': {
        'doc': "Thin-plate spline with iid noise",
        'hyper': {
            'theta1': {
                'hyperid': 33001, 'name': "log precision", 'short_name': "prec", 'prior': "pc.prec",
                'param': [1, .01], 'initial': 4, 'fixed': False, 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 33002, 'name': "logit phi", 'short_name': "phi", 'prior': "pc",
                'param': [0.5, 0.5], 'initial': 3, 'fixed': False, 'to_theta': lambda x: np.log(x / (1 - x)), 'from_theta': lambda x: np.exp(x) / (1 + np.exp(x))
            }
        },
        'constr': True, 'nrow_ncol': True, 'augmented': True, 'aug_factor': 2, 'aug_constr': 2,
        'n_div_by': None, 'n_required': False, 'set_default_values': True, 'pdf': "rw2diid"
    },
    'slm': {
        'doc': "Spatial lag model",
        'hyper': {
            'theta1': {
                'hyperid': 34001, 'name': "log precision", 'short_name': "prec", 'initial': 4, 'fixed': False,
                'prior': "loggamma", 'param': [1, 0.00005], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 34002, 'name': "rho", 'short_name': "rho", 'initial': 0, 'fixed': False,
                'prior': "normal", 'param': [0, 10], 'to_theta': lambda x: np.log(x / (1 - x)), 'from_theta': lambda x: 1 / (1 + np.exp(-x))
            }
        },
        'constr': False, 'nrow_ncol': False, 'augmented': False, 'aug_factor': 1, 'aug_constr': None,
        'n_div_by': None, 'n_required': True, 'set_default_values': True, 'pdf': "slm"
    },
    'matern2d': {
        'doc': "Matern covariance function on a regular grid",
        'hyper': {
            'theta1': {
                'hyperid': 35001, 'name': "log precision", 'short_name': "prec", 'initial': 4, 'fixed': False,
                'prior': "loggamma", 'param': [1, 0.00005], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 35002, 'name': "log range", 'short_name': "range", 'initial': 2, 'fixed': False,
                'prior': "loggamma", 'param': [1, 0.01], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            }
        },
        'constr': False, 'nrow_ncol': True, 'augmented': False, 'aug_factor': 1, 'aug_constr': None,
        'n_div_by': None, 'n_required': False, 'set_default_values': True, 'pdf': "matern2d"
    },
    'dmatern': {
        'doc': "Dense Matern field",
        'hyper': {
            'theta1': {
                'hyperid': 35101, 'name': "log precision", 'short_name': "prec", 'initial': 3, 'fixed': False,
                'prior': "pc.prec", 'param': [1, 0.01], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 35102, 'name': "log range", 'short_name': "range", 'initial': 0, 'fixed': False,
                'prior': "pc.range", 'param': [1, 0.5], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            'theta3': {
                'hyperid': 35103, 'name': "log nu", 'short_name': "nu", 'initial': np.log(0.5), 'fixed': True,
                'prior': "loggamma", 'param': [0.5, 1], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            }
        },
        'constr': False, 'nrow_ncol': False, 'augmented': False, 'aug_factor': 1, 'aug_constr': None,
        'n_div_by': None, 'n_required': True, 'set_default_values': True, 'pdf': "dmatern"
    },
    'copy': {
        'doc': "Create a copy of a model component",
        'hyper': {
            'theta': {
                'hyperid': 36001, 'name': "beta", 'short_name': "b", 'initial': 0.0, 'fixed': True,
                'prior': "normal", 'param': [1, 10],
                'to.theta': (
                    "function (x, REPLACE.ME.low, REPLACE.ME.high) <<NEWLINE>>"
                    "{<<NEWLINE>>"
                    "    if (all(is.infinite(c(low, high))) || low == high) {<<NEWLINE>>"
                    "        stopifnot(low < high)<<NEWLINE>>"
                    "        return(x)<<NEWLINE>>"
                    "    }<<NEWLINE>>"
                    "    else if (all(is.finite(c(low, high)))) {<<NEWLINE>>"
                    "        stopifnot(low < high)<<NEWLINE>>"
                    "        return(log(-(low - x)/(high - x)))<<NEWLINE>>"
                    "    }<<NEWLINE>>"
                    "    else if (is.finite(low) && is.infinite(high) && high > low) {<<NEWLINE>>"
                    "        return(log(x - low))<<NEWLINE>>"
                    "    }<<NEWLINE>>"
                    "    else {<<NEWLINE>>"
                    "        stop(\"Condition not yet implemented\")<<NEWLINE>>"
                    "    }<<NEWLINE>>"
                    "}"
                ),
                'from.theta': (
                    "function (x, REPLACE.ME.low, REPLACE.ME.high) <<NEWLINE>>"
                    "{<<NEWLINE>>"
                    "    if (all(is.infinite(c(low, high))) || low == high) {<<NEWLINE>>"
                    "        stopifnot(low < high)<<NEWLINE>>"
                    "        return(x)<<NEWLINE>>"
                    "    }<<NEWLINE>>"
                    "    else if (all(is.finite(c(low, high)))) {<<NEWLINE>>"
                    "        stopifnot(low < high)<<NEWLINE>>"
                    "        return(low + exp(x)/(1 + exp(x)) * (high - low))<<NEWLINE>>"
                    "    }<<NEWLINE>>"
                    "    else if (is.finite(low) && is.infinite(high) && high > low) {<<NEWLINE>>"
                    "        return(low + exp(x))<<NEWLINE>>"
                    "    }<<NEWLINE>>"
                    "    else {<<NEWLINE>>"
                    "        stop(\"Condition not yet implemented\")<<NEWLINE>>"
                    "    }<<NEWLINE>>"
                    "}"
                )
            }
        },
        'constr': False, 'nrow_ncol': False, 'augmented': False, 'aug_factor': 1, 'aug_constr': None,
        'n_div_by': None, 'n_required': False, 'set_default_values': False, 'pdf': "copy"
    },
    'scopy': {
        'doc': "Create a scaled copy of a model component",
        'hyper': {
            'theta1': {
                'hyperid': 36101, 'name': "mean", 'short_name': "mean", 'initial': 1.0, 'fixed': False,
                'prior': "normal", 'param': [1, 10], 'to_theta': lambda x: x, 'from_theta': lambda x: x
            },
            'theta2': {
                'hyperid': 36102, 'name': "slope", 'short_name': "slope", 'initial': 0, 'fixed': False,
                'prior': "normal", 'param': [0, 10], 'to_theta': lambda x: x, 'from_theta': lambda x: x
            },
            **{f'theta{i+2}': {
                'hyperid': 36102 + i,
                'name': f"spline.theta{i}",
                'short_name': f"spline{i if i > 1 else ''}",
                'initial': 0,
                'fixed': False,
                'prior': 'laplace' if i == 1 else 'none',
                'param': [0, 10] if i == 1 else [],
                'to_theta': lambda x: x,
                'from_theta': lambda x: x
            } for i in range(1, 14)}
        },
        'constr': False, 'nrow_ncol': False, 'augmented': False, 'aug_factor': 1, 'aug_constr': None,
        'n_div_by': None, 'n_required': False, 'set_default_values': False, 'pdf': "scopy"
    },
    'clinear': {
        'doc': "Constrained linear effect",
        'hyper': {
            'theta': {
                'hyperid': 37001, 'name': "beta", 'short_name': "b", 'initial': 1, 'fixed': False,
                'prior': "normal", 'param': [1, 10],
                'to.theta': (
                    "function (x, REPLACE.ME.low, REPLACE.ME.high) <<NEWLINE>>"
                    "{<<NEWLINE>>"
                    "    if (all(is.infinite(c(low, high))) || low == high) {<<NEWLINE>>"
                    "        stopifnot(low < high)<<NEWLINE>>"
                    "        return(x)<<NEWLINE>>"
                    "    }<<NEWLINE>>"
                    "    else if (all(is.finite(c(low, high)))) {<<NEWLINE>>"
                    "        stopifnot(low < high)<<NEWLINE>>"
                    "        return(log(-(low - x)/(high - x)))<<NEWLINE>>"
                    "    }<<NEWLINE>>"
                    "    else if (is.finite(low) && is.infinite(high) && high > low) {<<NEWLINE>>"
                    "        return(log(x - low))<<NEWLINE>>"
                    "    }<<NEWLINE>>"
                    "    else {<<NEWLINE>>"
                    "        stop(\"Condition not yet implemented\")<<NEWLINE>>"
                    "    }<<NEWLINE>>"
                    "}"
                ),
                'from.theta': (
                    "function (x, REPLACE.ME.low, REPLACE.ME.high) <<NEWLINE>>"
                    "{<<NEWLINE>>"
                    "    if (all(is.infinite(c(low, high))) || low == high) {<<NEWLINE>>"
                    "        stopifnot(low < high)<<NEWLINE>>"
                    "        return(x)<<NEWLINE>>"
                    "    }<<NEWLINE>>"
                    "    else if (all(is.finite(c(low, high)))) {<<NEWLINE>>"
                    "        stopifnot(low < high)<<NEWLINE>>"
                    "        return(low + exp(x)/(1 + exp(x)) * (high - low))<<NEWLINE>>"
                    "    }<<NEWLINE>>"
                    "    else if (is.finite(low) && is.infinite(high) && high > low) {<<NEWLINE>>"
                    "        return(low + exp(x))<<NEWLINE>>"
                    "    }<<NEWLINE>>"
                    "    else {<<NEWLINE>>"
                    "        stop(\"Condition not yet implemented\")<<NEWLINE>>"
                    "    }<<NEWLINE>>"
                    "}"
                )
            }
        },
        'constr': False, 'nrow_ncol': False, 'augmented': False, 'aug_factor': 1, 'aug_constr': None,
        'n_div_by': None, 'n_required': False, 'set_default_values': False, 'pdf': "clinear"
    },
    'sigm': {
        'doc': "Sigmoidal effect of a covariate",
        'hyper': {
            'theta1': {
                'hyperid': 38001, 'name': "beta", 'short_name': "b", 'initial': 1, 'fixed': False,
                'prior': "normal", 'param': [1, 10], 'to_theta': lambda x: x, 'from_theta': lambda x: x
            },
            'theta2': {
                'hyperid': 38002, 'name': "loghalflife", 'short_name': "halflife", 'initial': 3, 'fixed': False,
                'prior': "loggamma", 'param': [3, 1], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            'theta3': {
                'hyperid': 38003, 'name': "logshape", 'short_name': "shape", 'initial': 0, 'fixed': False,
                'prior': "loggamma", 'param': [10, 10], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            }
        },
        'constr': False, 'nrow_ncol': False, 'augmented': False, 'aug_factor': 1, 'aug_constr': None,
        'n_div_by': None, 'n_required': False, 'set_default_values': False, 'pdf': "sigm"
    },
    'revsigm': {
        'doc': "Reverse sigmoidal effect of a covariate",
        'hyper': {
            'theta1': {
                'hyperid': 39001, 'name': "beta", 'short_name': "b", 'initial': 1, 'fixed': False,
                'prior': "normal", 'param': [1, 10], 'to_theta': lambda x: x, 'from_theta': lambda x: x
            },
            'theta2': {
                'hyperid': 39002, 'name': "loghalflife", 'short_name': "halflife", 'initial': 3, 'fixed': False,
                'prior': "loggamma", 'param': [3, 1], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            'theta3': {
                'hyperid': 39003, 'name': "logshape", 'short_name': "shape", 'initial': 0, 'fixed': False,
                'prior': "loggamma", 'param': [10, 10], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            }
        },
        'constr': False, 'nrow_ncol': False, 'augmented': False, 'aug_factor': 1, 'aug_constr': None,
        'n_div_by': None, 'n_required': False, 'set_default_values': False, 'pdf': "sigm"
    },
    'log1exp': {
        'doc': "A nonlinear model of a covariate",
        'hyper': {
            'theta1': {
                'hyperid': 39011, 'name': "beta", 'short_name': "b", 'initial': 1, 'fixed': False,
                'prior': "normal", 'param': [0, 1], 'to_theta': lambda x: x, 'from_theta': lambda x: x
            },
            'theta2': {
                'hyperid': 39012, 'name': "alpha", 'short_name': "a", 'initial': 0, 'fixed': False,
                'prior': "normal", 'param': [0, 1], 'to_theta': lambda x: x, 'from_theta': lambda x: x
            },
            'theta3': {
                'hyperid': 39013, 'name': "gamma", 'short_name': "g", 'initial': 0, 'fixed': False,
                'prior': "normal", 'param': [0, 1], 'to_theta': lambda x: x, 'from_theta': lambda x: x
            }
        },
        'constr': False, 'nrow_ncol': False, 'augmented': False, 'aug_factor': 1, 'aug_constr': None,
        'n_div_by': None, 'n_required': False, 'set_default_values': False, 'pdf': "log1exp"
    },
    'logdist': {
        'doc': "A nonlinear model of a covariate",
        'hyper': {
            'theta1': {
                'hyperid': 39021, 'name': "beta", 'short_name': "b", 'initial': 1, 'fixed': False,
                'prior': "normal", 'param': [0, 1], 'to_theta': lambda x: x, 'from_theta': lambda x: x
            },
            'theta2': {
                'hyperid': 39022, 'name': "alpha1", 'short_name': "a1", 'initial': 0, 'fixed': False,
                'prior': "loggamma", 'param': [0.1, 1], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            },
            'theta3': {
                'hyperid': 39023, 'name': "alpha2", 'short_name': "a2", 'initial': 0, 'fixed': False,
                'prior': "loggamma", 'param': [0.1, 1], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)
            }
        },
        'constr': False, 'nrow_ncol': False, 'augmented': False, 'aug_factor': 1, 'aug_constr': None,
        'n_div_by': None, 'n_required': False, 'set_default_values': False, 'pdf': "logdist"
    }
}


def get_latent_models() -> dict:
    """Return all latent model definitions."""
    return LATENT_MODELS


# Helper functions for bounded parameters (used by 'copy' and 'clinear' models)

def copy_clinear_to_theta(x, low, high):
    """
    Transform for 'copy'/'clinear' betas with optional bounds.
    Cases:
    1) (-inf, +inf)  or  low == high  -> identity
    2) (low, high) both finite        -> log((x - low) / (high - x))
    3) (low, +inf)                    -> log(x - low)
    4) (-inf, high)                   -> log(high - x)
    """
    eps = 1e-15
    low_inf = np.isinf(low)
    high_inf = np.isinf(high)

    # Case 1: unbounded (or degenerate)
    if (low_inf and high_inf) or (low == high):
        return x

    # Case 2: both finite (strict interior)
    if np.isfinite(low) and np.isfinite(high):
        if not (low < high):
            raise ValueError("low must be less than high")
        if not (low < x < high):
            raise ValueError(f"x must satisfy low < x < high; got x={x}, low={low}, high={high}")
        return np.log((x - low + eps) / (high - x + eps))

    # Case 3: lower finite, upper infinite
    if np.isfinite(low) and high_inf:
        if not (x > low):
            raise ValueError(f"x must be > low; got x={x}, low={low}")
        return np.log(x - low + eps)

    # Case 4: lower infinite, upper finite
    if low_inf and np.isfinite(high):
        if not (x < high):
            raise ValueError(f"x must be < high; got x={x}, high={high}")
        return np.log(high - x + eps)

    raise NotImplementedError("Unhandled bound configuration in copy_clinear_to_theta().")


def copy_clinear_from_theta(z, low, high):
    """
    Inverse transform for 'copy'/'clinear' betas with optional bounds.
    Cases:
    1) (-inf, +inf)  or  low == high  -> identity
    2) (low, high) both finite        -> low + sigmoid(z) * (high - low)
    3) (low, +inf)                    -> low + exp(z)
    4) (-inf, high)                   -> high - exp(z)
    """
    low_inf = np.isinf(low)
    high_inf = np.isinf(high)

    # Case 1: unbounded (or degenerate)
    if (low_inf and high_inf) or (low == high):
        return z

    # Case 2: both finite
    if np.isfinite(low) and np.isfinite(high):
        if not (low < high):
            raise ValueError("low must be less than high")
        ez = np.exp(z)
        return low + ez / (1.0 + ez) * (high - low)

    # Case 3: lower finite, upper infinite
    if np.isfinite(low) and high_inf:
        return low + np.exp(z)

    # Case 4: lower infinite, upper finite
    if low_inf and np.isfinite(high):
        return high - np.exp(z)

    raise NotImplementedError("Unhandled bound configuration in copy_clinear_from_theta().")

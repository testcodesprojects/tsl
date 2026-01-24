"""
Common hyperparameter transformation functions for INLA models.

These functions define the mapping between internal (theta) and user-scale
representations of hyperparameters.
"""

import numpy as np


# Identity transform (no transformation)
def identity(x):
    """Identity transform: returns x unchanged."""
    return x


# Log/Exp transforms (for precision parameters)
def log_transform(x):
    """Log transform: theta = log(x)."""
    return np.log(x)


def exp_transform(x):
    """Exp transform: x = exp(theta)."""
    return np.exp(x)


# Logit/Inverse-logit transforms (for probability parameters)
def logit_transform(x):
    """Logit transform: theta = log(x/(1-x))."""
    return np.log(x / (1 - x))


def inv_logit_transform(x):
    """Inverse logit transform: x = exp(theta)/(1+exp(theta))."""
    return np.exp(x) / (1 + np.exp(x))


# Log1p transforms (for parameters that must be > -1)
def log1p_transform(x):
    """Log(1+x) transform."""
    return np.log(1 + x)


def expm1_transform(x):
    """Exp(x)-1 transform (inverse of log1p)."""
    return np.exp(x) - 1


# Squared transforms
def sqrt_transform(x):
    """Square root transform."""
    return np.sqrt(x)


def square_transform(x):
    """Square transform."""
    return x ** 2


# Negative log transform (for correlation parameters)
def neg_log_transform(x):
    """Negative log transform: theta = -log(x)."""
    return -np.log(x)


def neg_exp_transform(x):
    """Negative exp transform: x = exp(-theta)."""
    return np.exp(-x)


# Standard transform sets (commonly used combinations)
TRANSFORMS = {
    'identity': (identity, identity),
    'log': (log_transform, exp_transform),
    'logit': (logit_transform, inv_logit_transform),
    'log1p': (log1p_transform, expm1_transform),
    'sqrt': (sqrt_transform, square_transform),
    'neg_log': (neg_log_transform, neg_exp_transform),
}


def get_transform(name: str):
    """
    Get a (to_theta, from_theta) transform pair by name.

    Parameters
    ----------
    name : str
        Transform name: 'identity', 'log', 'logit', 'log1p', 'sqrt', 'neg_log'

    Returns
    -------
    tuple
        (to_theta, from_theta) function pair
    """
    if name not in TRANSFORMS:
        raise ValueError(f"Unknown transform '{name}'. Available: {list(TRANSFORMS.keys())}")
    return TRANSFORMS[name]

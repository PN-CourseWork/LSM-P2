"""Numerical norms and error computations."""

from __future__ import annotations

import numpy as np


def discrete_l2_norm(values: np.ndarray, h: float) -> float:
    """Approximate L2 norm using composite trapezoidal rule.

    Parameters
    ----------
    values : np.ndarray
        Function values at discrete points
    h : float
        Grid spacing

    Returns
    -------
    float
        Approximate L2 norm
    """
    return np.sqrt(h * np.sum(np.abs(values) ** 2))


def discrete_l2_error(
    f_exact: np.ndarray, f_num: np.ndarray, interval_length: float
) -> float:
    """Compute discrete L2 error between exact and numerical solutions.
    #TODO: Change to use Mass Matrix instead

    Parameters
    ----------
    f_exact : np.ndarray
        Exact function values
    f_num : np.ndarray
        Numerical approximation values
    interval_length : float
        Length of the interval

    Returns
    -------
    float
        Discrete L2 error norm
    """
    diff = f_num - f_exact
    h = interval_length / f_exact.size
    return np.sqrt(h) * np.linalg.norm(diff)


def discrete_linf_error(f_exact: np.ndarray, f_num: np.ndarray) -> float:
    """Compute discrete :math:`L^\\infty` (maximum) error.

    Parameters
    ----------
    f_exact : np.ndarray
        Exact function values
    f_num : np.ndarray
        Numerical approximation values

    Returns
    -------
    float
        Maximum absolute error
    """
    return np.max(np.abs(f_num - f_exact))

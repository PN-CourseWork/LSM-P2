"""
Numerical Integration
======================

Simple numerical integration methods.
"""

from typing import Callable

import numpy as np
from numpy.typing import NDArray


def integrate_trapz(
    f: Callable[[float], float], a: float, b: float, n: int = 100
) -> float:
    """
    Integrate a function using the trapezoidal rule.

    Parameters
    ----------
    f : Callable
        Function to integrate
    a : float
        Lower bound of integration
    b : float
        Upper bound of integration
    n : int, optional
        Number of subintervals (default: 100)

    Returns
    -------
    float
        Approximate integral value

    Examples
    --------
    >>> from numutils import integrate_trapz
    >>> import numpy as np
    >>> result = integrate_trapz(lambda x: x**2, 0, 1, n=1000)
    >>> np.isclose(result, 1/3, atol=1e-4)
    True
    """
    x = np.linspace(a, b, n + 1)
    y = np.array([f(xi) for xi in x])
    h = (b - a) / n
    return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])


def integrate_simpson(
    f: Callable[[float], float], a: float, b: float, n: int = 100
) -> float:
    """
    Integrate a function using Simpson's rule.

    Parameters
    ----------
    f : Callable
        Function to integrate
    a : float
        Lower bound of integration
    b : float
        Upper bound of integration
    n : int, optional
        Number of subintervals (must be even, default: 100)

    Returns
    -------
    float
        Approximate integral value

    Examples
    --------
    >>> from numutils import integrate_simpson
    >>> import numpy as np
    >>> result = integrate_simpson(lambda x: x**2, 0, 1, n=1000)
    >>> np.isclose(result, 1/3, atol=1e-6)
    True
    """
    if n % 2 != 0:
        raise ValueError("n must be even for Simpson's rule")

    x = np.linspace(a, b, n + 1)
    y = np.array([f(xi) for xi in x])
    h = (b - a) / n

    return h / 3 * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]) + y[-1])

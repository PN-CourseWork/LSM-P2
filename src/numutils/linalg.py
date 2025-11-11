"""
Linear Algebra Utilities
=========================

Simple linear algebra operations for numerical computing.
"""

import numpy as np
from numpy.typing import NDArray


def norm(x: NDArray, p: int = 2) -> float:
    """
    Compute the p-norm of a vector.

    Parameters
    ----------
    x : NDArray
        Input vector
    p : int, optional
        Order of the norm (default: 2)

    Returns
    -------
    float
        The p-norm of the vector

    Examples
    --------
    >>> import numpy as np
    >>> from numutils import norm
    >>> x = np.array([3, 4])
    >>> norm(x)
    5.0
    """
    if p == np.inf:
        return np.max(np.abs(x))
    return np.sum(np.abs(x) ** p) ** (1 / p)


def solve_linear(A: NDArray, b: NDArray) -> NDArray:
    """
    Solve a linear system Ax = b.

    Parameters
    ----------
    A : NDArray
        Coefficient matrix (n x n)
    b : NDArray
        Right-hand side vector (n,)

    Returns
    -------
    NDArray
        Solution vector x (n,)

    Examples
    --------
    >>> import numpy as np
    >>> from numutils import solve_linear
    >>> A = np.array([[2, 1], [1, 3]])
    >>> b = np.array([1, 2])
    >>> x = solve_linear(A, b)
    >>> np.allclose(A @ x, b)
    True
    """
    return np.linalg.solve(A, b)

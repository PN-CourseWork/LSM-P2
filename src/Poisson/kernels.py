"""Jacobi iteration kernels."""

import numpy as np
import numba
from numba import njit, prange


@njit(parallel=True)
def _jacobi_step_numba(uold: np.ndarray, u: np.ndarray, f: np.ndarray, h: float, omega: float) -> float:
    """Numba JIT implementation of Jacobi iteration step."""
    c = 1.0 / 6.0
    h2 = h * h
    N = u.shape[0] - 2

    for i in prange(1, u.shape[0] - 1):
        for j in range(1, u.shape[1] - 1):
            for k in range(1, u.shape[2] - 1):
                u[i, j, k] = (
                    omega * c * (
                        uold[i - 1, j, k]
                        + uold[i + 1, j, k]
                        + uold[i, j - 1, k]
                        + uold[i, j + 1, k]
                        + uold[i, j, k - 1]
                        + uold[i, j, k + 1]
                        + h2 * f[i, j, k]
                    )
                    + (1.0 - omega) * uold[i, j, k]
                )

    diff_sum = 0.0
    for i in prange(u.shape[0]):
        for j in range(u.shape[1]):
            for k in range(u.shape[2]):
                diff = u[i, j, k] - uold[i, j, k]
                diff_sum += diff * diff

    return np.sqrt(diff_sum) / N**3


class NumPyKernel:
    """NumPy-based Jacobi kernel."""

    def __init__(self, h: float = None, omega: float = None):
        """Initialize NumPy kernel.

        Parameters
        ----------
        h : float, optional
            Grid spacing (can be set later via configure)
        omega : float, optional
            Relaxation parameter (can be set later via configure)
        """
        self.h = h
        self.omega = omega

    def configure(self, h: float, omega: float):
        """Configure kernel parameters.

        Parameters
        ----------
        h : float
            Grid spacing
        omega : float
            Relaxation parameter
        """
        self.h = h
        self.omega = omega

    def step(self, uold: np.ndarray, u: np.ndarray, f: np.ndarray) -> float:
        """Perform one Jacobi iteration step.

        Parameters
        ----------
        uold : np.ndarray
            Previous solution
        u : np.ndarray
            Current solution (will be updated in-place)
        f : np.ndarray
            Source term

        Returns
        -------
        float
            Iterative residual ||u - uold||_2 / N^3
        """
        c = 1.0 / 6.0
        h2 = self.h * self.h
        N = u.shape[0] - 2

        u[1:-1, 1:-1, 1:-1] = (
            self.omega * c * (
                uold[0:-2, 1:-1, 1:-1]
                + uold[2:, 1:-1, 1:-1]
                + uold[1:-1, 0:-2, 1:-1]
                + uold[1:-1, 2:, 1:-1]
                + uold[1:-1, 1:-1, 0:-2]
                + uold[1:-1, 1:-1, 2:]
                + h2 * f[1:-1, 1:-1, 1:-1]
            )
            + (1.0 - self.omega) * uold[1:-1, 1:-1, 1:-1]
        )

        return np.sqrt(np.sum((u - uold) ** 2)) / N**3

    def warmup(self, N: int = 10, omega: float = 0.75):
        """Warmup kernel (no-op for NumPy)."""
        pass


class NumbaKernel:
    """Numba JIT-compiled Jacobi kernel."""

    def __init__(self, h: float = None, omega: float = None, num_threads: int = None):
        """Initialize Numba kernel.

        Parameters
        ----------
        h : float, optional
            Grid spacing (can be set later via configure)
        omega : float, optional
            Relaxation parameter (can be set later via configure)
        num_threads : int, optional
            Number of threads for parallel execution. If None, uses Numba default.
        """
        self.h = h
        self.omega = omega
        self.num_threads = num_threads

        # Set thread count if specified
        if num_threads is not None:
            numba.set_num_threads(num_threads)

    def configure(self, h: float, omega: float):
        """Configure kernel parameters.

        Parameters
        ----------
        h : float
            Grid spacing
        omega : float
            Relaxation parameter
        """
        self.h = h
        self.omega = omega

    def step(self, uold: np.ndarray, u: np.ndarray, f: np.ndarray) -> float:
        """Perform one Jacobi iteration step.

        Parameters
        ----------
        uold : np.ndarray
            Previous solution
        u : np.ndarray
            Current solution (will be updated in-place)
        f : np.ndarray
            Source term

        Returns
        -------
        float
            Iterative residual ||u - uold||_2 / N^3
        """
        return _jacobi_step_numba(uold, u, f, self.h, self.omega)

    def warmup(self, N: int = 10, omega: float = 0.75):
        """Trigger JIT compilation with a small problem.

        Parameters
        ----------
        N : int, optional
            Small grid size for warmup (default: 10)
        omega : float, optional
            Relaxation parameter (default: 0.75)
        """
        h = 2.0 / (N - 1)
        u1 = np.zeros((N, N, N), dtype=np.float64)
        u2 = np.zeros((N, N, N), dtype=np.float64)
        f = np.random.randn(N, N, N)

        # Run 5 iterations to trigger compilation
        for _ in range(5):
            _jacobi_step_numba(u1, u2, f, h, omega)
            u1, u2 = u2, u1

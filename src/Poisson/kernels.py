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

    def __init__(self, N: int, omega: float, h: float = None):
        """Initialize NumPy kernel.

        Parameters
        ----------
        N : int
            Grid size (number of points in each dimension)
        omega : float
            Relaxation parameter
        h : float, optional
            Grid spacing. If None, assumes domain [-1, 1]³ and computes h = 2.0 / (N - 1)
        """
        self.N = N
        self.omega = omega
        self.h = h if h is not None else 2.0 / (N - 1)

    def step(self, uold: np.ndarray, u: np.ndarray, f: np.ndarray) -> float:
        """Perform one Jacobi iteration step.

        Parameters
        ----------
        uold : np.ndarray
            Previous solution (including ghost zones for MPI)
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

    def warmup(self, warmup_size: int = 10):
        """Warmup kernel (no-op for NumPy).

        Parameters
        ----------
        warmup_size : int, optional
            Small grid size for warmup (default: 10, unused for NumPy)
        """
        pass


class NumbaKernel:
    """Numba JIT-compiled Jacobi kernel."""

    def __init__(self, N: int, omega: float, num_threads: int = None, h: float = None):
        """Initialize Numba kernel.

        Parameters
        ----------
        N : int
            Grid size (number of points in each dimension)
        omega : float
            Relaxation parameter
        num_threads : int, optional
            Number of threads for parallel execution. If None, uses Numba default.
        h : float, optional
            Grid spacing. If None, assumes domain [-1, 1]³ and computes h = 2.0 / (N - 1)
        """
        self.N = N
        self.omega = omega
        self.h = h if h is not None else 2.0 / (N - 1)
        self.num_threads = num_threads

        # Set thread count if specified
        if num_threads is not None:
            numba.set_num_threads(num_threads)

    def step(self, uold: np.ndarray, u: np.ndarray, f: np.ndarray) -> float:
        """Perform one Jacobi iteration step.

        Parameters
        ----------
        uold : np.ndarray
            Previous solution (including ghost zones for MPI)
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

    def warmup(self, warmup_size: int = 10):
        """Trigger JIT compilation with a small problem.

        Parameters
        ----------
        warmup_size : int, optional
            Small grid size for warmup (default: 10)
        """
        h_warmup = 2.0 / (warmup_size - 1)
        u1 = np.zeros((warmup_size, warmup_size, warmup_size), dtype=np.float64)
        u2 = np.zeros((warmup_size, warmup_size, warmup_size), dtype=np.float64)
        f = np.random.randn(warmup_size, warmup_size, warmup_size)

        # Run 5 iterations to trigger compilation
        for _ in range(5):
            _jacobi_step_numba(u1, u2, f, h_warmup, self.omega)
            u1, u2 = u2, u1

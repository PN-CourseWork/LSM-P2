"""Jacobi iteration kernels.

Simple kernel implementations - tracking is handled by the solver.
"""

import numpy as np
import numba
from numba import njit, prange


@njit(parallel=True)
def _jacobi_step_numba(
    uold: np.ndarray, u: np.ndarray, f: np.ndarray, h: float, omega: float
) -> float:
    """Numba JIT implementation of Jacobi iteration step."""
    c = 1.0 / 6.0
    h2 = h * h

    for i in prange(1, u.shape[0] - 1):
        for j in range(1, u.shape[1] - 1):
            for k in range(1, u.shape[2] - 1):
                u[i, j, k] = (
                    omega
                    * c
                    * (
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

    return 0.0  # Residual computed separately in solver


class NumPyKernel:
    """NumPy-based Jacobi kernel."""

    def __init__(self, omega: float, specified_numba_threads: int = 1):
        self.omega = omega
        self.observed_numba_threads = None  # Not applicable for NumPy

    def step(self, uold: np.ndarray, u: np.ndarray, f: np.ndarray, h: float):
        """Perform one Jacobi iteration step."""
        c = 1.0 / 6.0
        h2 = h * h

        u[1:-1, 1:-1, 1:-1] = (
            self.omega
            * c
            * (
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

    def warmup(self, warmup_size: int = 10):
        """No-op for NumPy kernel."""
        pass


class NumbaKernel:
    """Numba JIT-compiled Jacobi kernel."""

    def __init__(self, omega: float, specified_numba_threads: int = 1):
        self.omega = omega

        # Set requested threads (may be clamped by NUMBA_NUM_THREADS env var)
        if specified_numba_threads is not None:
            numba.set_num_threads(specified_numba_threads)

        # Record what Numba actually reports
        self.observed_numba_threads = numba.get_num_threads()

    def step(self, uold: np.ndarray, u: np.ndarray, f: np.ndarray, h: float):
        """Perform one Jacobi iteration step."""
        _jacobi_step_numba(uold, u, f, h, self.omega)

    def warmup(self, warmup_size: int = 10):
        """Trigger JIT compilation with a small problem."""
        h = 2.0 / (warmup_size - 1)
        u1 = np.zeros((warmup_size, warmup_size, warmup_size), dtype=np.float64)
        u2 = np.zeros_like(u1)
        f = np.random.randn(warmup_size, warmup_size, warmup_size)
        for _ in range(5):
            _jacobi_step_numba(u1, u2, f, h, self.omega)
            u1, u2 = u2, u1

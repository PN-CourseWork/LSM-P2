"""Jacobi iteration kernels."""

import time
import numpy as np
import numba
from numba import njit, prange

from .datastructures import KernelParams, KernelMetrics, KernelSeries


@njit(parallel=True)
def _jacobi_step_numba(uold: np.ndarray, u: np.ndarray, f: np.ndarray, h: float, omega: float) -> float:
    """Numba JIT implementation of Jacobi iteration step."""
    c = 1.0 / 6.0
    h2 = h * h

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

    # Compute sum of squared differences over interior points only
    # Return unnormalized sum - normalization done globally in solver
    diff_sum = 0.0
    for i in prange(1, u.shape[0] - 1):
        for j in range(1, u.shape[1] - 1):
            for k in range(1, u.shape[2] - 1):
                diff = u[i, j, k] - uold[i, j, k]
                diff_sum += diff * diff

    return diff_sum


class _BaseKernel:
    """Base class for Jacobi kernels with common tracking logic."""

    def __init__(self, **kwargs):
        self.parameters = KernelParams(**kwargs)
        self.metrics = KernelMetrics()
        self.timeseries = KernelSeries()

    def _track(self, residual: float, compute_time: float):
        """Update metrics and timeseries."""
        self.metrics.iterations += 1
        self.metrics.final_residual = residual
        self.metrics.total_compute_time += compute_time
        self.timeseries.residuals.append(residual)
        self.timeseries.compute_times.append(compute_time)

    def warmup(self, warmup_size=10):
        """Warmup kernel (no-op by default)."""
        pass


class NumPyKernel(_BaseKernel):
    """NumPy-based Jacobi kernel."""

    def step(self, uold: np.ndarray, u: np.ndarray, f: np.ndarray) -> float:
        """Perform one Jacobi iteration step."""
        start = time.perf_counter()

        c = 1.0 / 6.0
        h2 = self.parameters.h * self.parameters.h

        u[1:-1, 1:-1, 1:-1] = (
            self.parameters.omega * c * (
                uold[0:-2, 1:-1, 1:-1]
                + uold[2:, 1:-1, 1:-1]
                + uold[1:-1, 0:-2, 1:-1]
                + uold[1:-1, 2:, 1:-1]
                + uold[1:-1, 1:-1, 0:-2]
                + uold[1:-1, 1:-1, 2:]
                + h2 * f[1:-1, 1:-1, 1:-1]
            )
            + (1.0 - self.parameters.omega) * uold[1:-1, 1:-1, 1:-1]
        )

        # Compute sum of squared differences over interior points only
        # Return unnormalized sum - normalization done globally in solver
        diff = u[1:-1, 1:-1, 1:-1] - uold[1:-1, 1:-1, 1:-1]
        diff_sum = np.sum(diff ** 2)
        self._track(diff_sum, time.perf_counter() - start)
        return diff_sum


class NumbaKernel(_BaseKernel):
    """Numba JIT-compiled Jacobi kernel."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.parameters.numba_threads is not None:
            numba.set_num_threads(self.parameters.numba_threads)

    def step(self, uold: np.ndarray, u: np.ndarray, f: np.ndarray) -> float:
        """Perform one Jacobi iteration step."""
        start = time.perf_counter()
        residual = _jacobi_step_numba(uold, u, f, self.parameters.h, self.parameters.omega)
        self._track(residual, time.perf_counter() - start)
        return residual

    def warmup(self, warmup_size=10):
        """Trigger JIT compilation with a small problem."""
        h = 2.0 / (warmup_size - 1)
        u1 = np.zeros((warmup_size, warmup_size, warmup_size), dtype=np.float64)
        u2 = np.zeros_like(u1)
        f = np.random.randn(warmup_size, warmup_size, warmup_size)
        for _ in range(5):
            _jacobi_step_numba(u1, u2, f, h, self.parameters.omega)
            u1, u2 = u2, u1

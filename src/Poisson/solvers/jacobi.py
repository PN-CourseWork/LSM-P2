"""Sequential Jacobi Solver."""

import time
import numpy as np

from .base import BaseSolver
from ..kernels import NumPyKernel, NumbaKernel
from ..problems import sinusoidal_source_term, sinusoidal_exact_solution


class JacobiSolver(BaseSolver):
    """Sequential Jacobi solver for kernel benchmarks.

    No MPI overhead - runs entirely on a single process.
    Useful for benchmarking kernel performance (NumPy vs Numba).
    """

    def __init__(
        self,
        N: int,
        use_numba: bool = False,
        specified_numba_threads: int = 1,
        **kwargs,
    ):
        super().__init__(N, **kwargs)

        self.use_numba = use_numba
        self.specified_numba_threads = specified_numba_threads

        # Select kernel
        self._init_kernel()

        # Allocate arrays
        self._init_arrays()

    def _init_kernel(self):
        """Initialize the Jacobi kernel."""
        if self.use_numba:
            self.kernel = NumbaKernel(
                N=self.N,
                omega=self.omega,
                specified_numba_threads=self.specified_numba_threads,
            )
        else:
            self.kernel = NumPyKernel(N=self.N, omega=self.omega)

    def _init_arrays(self):
        """Allocate solution arrays."""
        self.u = np.zeros((self.N, self.N, self.N), dtype=np.float64)
        self.u_old = np.zeros((self.N, self.N, self.N), dtype=np.float64)
        self.f = sinusoidal_source_term(self.N)

    def warmup(self, warmup_size: int = 10):
        """Warmup kernel (trigger Numba JIT)."""
        self.kernel.warmup(warmup_size=warmup_size)

    def solve(self):
        """Run Jacobi iteration."""
        self._reset_timeseries()

        n_interior = (self.N - 2) ** 3
        u, u_old = self.u, self.u_old

        t_start = self._get_time()

        for i in range(self.max_iter):
            # Sync halos (no-op for sequential)
            halo_time = self._sync_halos(u_old)

            # Compute step
            t0 = self._get_time()
            self.kernel.step(u_old, u, self.f)
            self._apply_boundary_conditions(u)
            compute_time = self._get_time() - t0

            self.timeseries.compute_times.append(compute_time)
            if halo_time > 0:
                self.timeseries.halo_times.append(halo_time)

            # Compute residual
            residual = self._compute_residual(u, u_old, n_interior)
            self.timeseries.residual_history.append(residual)

            if residual < self.tolerance:
                self.metrics.converged = True
                self.metrics.iterations = i + 1
                break

            # Swap buffers
            u, u_old = u_old, u
        else:
            self.metrics.iterations = self.max_iter
            self.metrics.converged = False

        wall_time = self._get_time() - t_start
        self._finalize(wall_time, u_old)

        return self.metrics

    def _reset_timeseries(self):
        """Clear timeseries data."""
        self.timeseries.residual_history.clear()
        self.timeseries.compute_times.clear()
        self.timeseries.halo_times.clear()

    def _get_time(self) -> float:
        """Get current time. Override for MPI timing."""
        return time.perf_counter()

    def _sync_halos(self, u: np.ndarray) -> float:
        """Sync halo regions. No-op for sequential."""
        return 0.0

    def _apply_boundary_conditions(self, u: np.ndarray):
        """Apply boundary conditions. No-op for sequential (zeros already)."""
        pass

    def _compute_residual(
        self, u: np.ndarray, u_old: np.ndarray, n_interior: int
    ) -> float:
        """Compute global residual norm."""
        diff = u[1:-1, 1:-1, 1:-1] - u_old[1:-1, 1:-1, 1:-1]
        return np.sqrt(np.sum(diff**2)) / n_interior

    def _finalize(self, wall_time: float, u_solution: np.ndarray):
        """Finalize metrics after solve."""
        self.u = u_solution
        self.metrics.final_residual = self.timeseries.residual_history[-1]
        self.metrics.total_compute_time = sum(self.timeseries.compute_times)
        self.metrics.observed_numba_threads = self.kernel.observed_numba_threads
        self._compute_metrics(wall_time, self.metrics.iterations)

    def compute_l2_error(self) -> float:
        """Compute L2 error against analytical solution."""
        u_exact = sinusoidal_exact_solution(self.N)
        diff = self.u[1:-1, 1:-1, 1:-1] - u_exact[1:-1, 1:-1, 1:-1]
        l2_error = np.sqrt(np.sum(diff**2) * self.h**3)
        self.metrics.final_error = l2_error
        return l2_error

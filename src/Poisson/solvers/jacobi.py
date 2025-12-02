"""Jacobi Solvers - Sequential and MPI."""

import time
import numpy as np

from .base import BaseSolver
from ..kernels import NumPyKernel, NumbaKernel
from ..problems import sinusoidal_source_term, sinusoidal_exact_solution


class JacobiSolver(BaseSolver):
    """Sequential Jacobi solver for kernel benchmarks.

    No MPI overhead - runs entirely on a single process.
    Useful for benchmarking kernel performance (NumPy vs Numba).

    Parameters
    ----------
    N : int
        Grid size (N x N x N).
    use_numba : bool
        Use Numba JIT kernel (default: False).
    numba_threads : int
        Number of Numba threads (default: 1).
    omega : float
        Relaxation factor (default: 0.8).
    max_iter : int
        Maximum iterations (default: 1000).
    tolerance : float
        Convergence tolerance (default: 1e-6).
    """

    def __init__(
        self,
        N: int,
        use_numba: bool = False,
        numba_threads: int = 1,
        **kwargs,
    ):
        super().__init__(N, **kwargs)

        self.use_numba = use_numba
        self.numba_threads = numba_threads

        # Select kernel
        self._init_kernel()

        # Allocate arrays
        self._init_arrays()

    def _init_kernel(self):
        """Initialize the Jacobi kernel."""
        if self.use_numba:
            self.kernel = NumbaKernel(
                N=self.N, omega=self.omega, numba_threads=self.numba_threads
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
                self.timeseries.halo_exchange_times.append(halo_time)

            # Compute residual
            residual = self._compute_residual(u, u_old, n_interior)
            self.timeseries.residual_history.append(residual)

            if residual < self.tolerance:
                self.results.converged = True
                self.results.iterations = i + 1
                break

            # Swap buffers
            u, u_old = u_old, u
        else:
            self.results.iterations = self.max_iter
            self.results.converged = False

        wall_time = self._get_time() - t_start
        self._finalize(wall_time, u_old)

        return self.results

    def _reset_timeseries(self):
        """Clear timeseries data."""
        self.timeseries.residual_history.clear()
        self.timeseries.compute_times.clear()
        self.timeseries.halo_exchange_times.clear()

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
        """Finalize results after solve."""
        self.u = u_solution
        self.results.final_residual = self.timeseries.residual_history[-1]
        self.results.total_compute_time = sum(self.timeseries.compute_times)
        self._compute_metrics(wall_time, self.results.iterations)

    def compute_l2_error(self) -> float:
        """Compute L2 error against analytical solution."""
        u_exact = sinusoidal_exact_solution(self.N)
        diff = self.u[1:-1, 1:-1, 1:-1] - u_exact[1:-1, 1:-1, 1:-1]
        l2_error = np.sqrt(np.sum(diff**2) * self.h**3)
        self.results.final_error = l2_error
        return l2_error

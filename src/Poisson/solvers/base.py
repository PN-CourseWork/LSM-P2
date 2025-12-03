"""Base class for solvers."""

import time
from abc import ABC, abstractmethod

import numpy as np

from ..datastructures import GlobalMetrics, LocalMetrics
from ..problems import sinusoidal_exact_solution


class BaseSolver(ABC):
    """Abstract base for all Poisson solvers."""

    # Bytes per lattice update: 7-point stencil = 6 neighbors + center + f = 8 mem-ops × 8 bytes
    BYTES_PER_POINT = 64

    def __init__(
        self,
        N: int,
        omega: float = 0.8,
        max_iter: int = 1000,
        tolerance: float = 1e-6,
    ):
        self.N = N
        self.omega = omega
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.h = 2.0 / (N - 1)

        # Metrics containers (match datastructures.py naming)
        self.metrics = GlobalMetrics()
        self.timeseries = LocalMetrics()

        # Timing accumulators
        self._time_compute = 0.0
        self._time_halo = 0.0

    @abstractmethod
    def solve(self) -> GlobalMetrics:
        """Execute the solver. Returns results."""
        pass

    def warmup(self, warmup_size: int = 10):
        """Warmup kernel (trigger Numba JIT if used)."""
        self.kernel.warmup(warmup_size=warmup_size)

    def compute_l2_error(self) -> float:
        """Compute L2 error against analytical solution.

        Uses hook methods for flexibility:
        - _get_solution_array(): Returns the solution array
        - _compute_l2_norm(): Computes norm (handles MPI allreduce if needed)
        """
        u = self._get_solution_array()
        u_exact = sinusoidal_exact_solution(self.N)
        l2_error = self._compute_l2_norm(u, u_exact)
        self.metrics.final_error = l2_error
        return l2_error

    def post_solve(self):
        """Compute all error metrics after solve.

        Call this after solve() to populate:
        - metrics.final_error (L2 error vs analytical)
        - metrics.final_alg_error (algebraic residual norm)
        - rank_topology (for MPI solvers)
        """
        self.compute_l2_error()
        self.compute_alg_residual()
        self.rank_topology = self._gather_topology()

    def _gather_topology(self):
        """Gather rank topology. Override for MPI solvers."""
        return None

    def _get_solution_array(self) -> np.ndarray:
        """Return the solution array."""
        return self.u

    def _get_source_array(self) -> np.ndarray:
        """Return the source term array."""
        return self.f

    def compute_alg_residual(
        self,
        u: np.ndarray = None,
        f: np.ndarray = None,
        h: float = None,
        r: np.ndarray = None,
    ) -> float:
        """Compute algebraic residual r = f - Au and return RMS norm.

        Parameters
        ----------
        u : array, optional
            Solution array. If None, uses _get_solution_array().
        f : array, optional
            Source term. If None, uses _get_source_array().
        h : float, optional
            Grid spacing. If None, uses self.h.
        r : array, optional
            Output array for residual. If None, allocates temporary array.

        Returns
        -------
        float
            RMS norm of the algebraic residual.

        Note: Caller is responsible for halo sync before and BC after if needed.
        """
        if u is None:
            u = self._get_solution_array()
        if f is None:
            f = self._get_source_array()
        if h is None:
            h = self.h
        if r is None:
            r = np.zeros_like(u)

        # Compute r = f - Au (interior only)
        h2 = h * h
        u_center = u[1:-1, 1:-1, 1:-1]
        u_neighbors = (
            u[0:-2, 1:-1, 1:-1]
            + u[2:, 1:-1, 1:-1]
            + u[1:-1, 0:-2, 1:-1]
            + u[1:-1, 2:, 1:-1]
            + u[1:-1, 1:-1, 0:-2]
            + u[1:-1, 1:-1, 2:]
        )
        laplacian = (u_neighbors - 6.0 * u_center) / h2
        r[1:-1, 1:-1, 1:-1] = f[1:-1, 1:-1, 1:-1] + laplacian

        # Compute RMS norm
        interior = r[1:-1, 1:-1, 1:-1]
        local_sum_sq = np.sum(interior**2)
        local_pts = float(interior.size)

        global_sum_sq = self._reduce_sum(local_sum_sq)
        global_pts = self._reduce_sum(local_pts)

        alg_residual = np.sqrt(global_sum_sq / global_pts)  # RMS norm
        self.metrics.final_alg_error = alg_residual
        return alg_residual

    def _get_time(self) -> float:
        """Get current time. Override for MPI timing."""
        return time.perf_counter()

    def _reduce_sum(self, local_sum: float) -> float:
        """Reduce sum across ranks. Override for MPI."""
        return local_sum

    def _is_root(self) -> bool:
        """True if this rank should log metrics. Override for MPI."""
        return True

    def _sync_halos(self, u: np.ndarray, lvl=None) -> float:
        """Sync halo regions. No-op for sequential. Override for MPI."""
        return 0.0

    def _apply_boundary_conditions(self, u: np.ndarray, lvl=None):
        """Apply boundary conditions. No-op for sequential. Override for MPI."""
        pass

    def _barrier(self):
        """Synchronize all ranks before timing. No-op for sequential."""
        pass

    def _compute_l2_norm(self, u: np.ndarray, u_exact: np.ndarray) -> float:
        """Compute L2 norm. Override for MPI allreduce."""
        diff = u[1:-1, 1:-1, 1:-1] - u_exact[1:-1, 1:-1, 1:-1]
        return np.sqrt(np.sum(diff**2) * self.h**3)

    def _reset(self):
        """Reset timers and timeseries."""
        self._time_compute = 0.0
        self._time_halo = 0.0
        self.timeseries.clear()

    def _finalize(self, wall_time: float):
        """Finalize metrics after solve."""
        if self._is_root():
            self.metrics.final_residual = self.timeseries.residual_history[-1]
            self.metrics.total_compute_time = self._time_compute
            self.metrics.total_halo_time = self._time_halo
        self.metrics.observed_numba_threads = self.kernel.observed_numba_threads
        self._compute_metrics(wall_time, self.metrics.iterations)

    def _compute_metrics(self, wall_time: float, iterations: int):
        """Compute performance metrics."""
        self.metrics.wall_time = wall_time
        self.metrics.iterations = iterations

        n_interior = (self.N - 2) ** 3
        if iterations > 0 and wall_time > 0:
            self.metrics.mlups = n_interior * iterations / (wall_time * 1e6)
            total_bytes = n_interior * iterations * self.BYTES_PER_POINT
            self.metrics.bandwidth_gb_s = total_bytes / (wall_time * 1e9)

"""Jacobi Solvers - Sequential and MPI."""

import time
import numpy as np
from mpi4py import MPI # Added import
from ..mpi.grid import DistributedGrid # Added import

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


class JacobiMPISolver(JacobiSolver):
    """Parallel Jacobi solver with MPI domain decomposition.

    Extends JacobiSolver with distributed grids and halo exchange.
    """

    def __init__(
        self,
        N: int,
        strategy: str = "sliced",
        communicator: str = "custom",
        **kwargs,
    ):
        # MPI setup before parent init
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.strategy = strategy
        self.communicator = communicator

        super().__init__(N, **kwargs)

        # Store config info
        self.local_shape = self.grid.local_shape
        self.halo_size_mb = self.grid.get_halo_size_bytes() / (1024 * 1024)

    def _init_arrays(self):
        """Allocate arrays for MPI run."""
        self.grid = DistributedGrid(
            self.N, self.comm, strategy=self.strategy, halo_exchange=self.communicator
        )
        self.u = self.grid.allocate()
        self.u_old = self.grid.allocate()
        self.f = self.grid.allocate()
        self.grid.fill_source_term(self.f)
        self.grid.apply_boundary_conditions(self.u)
        self.grid.apply_boundary_conditions(self.u_old)

    def _get_time(self) -> float:
        """Get current time using MPI.Wtime()."""
        return MPI.Wtime()

    def _sync_halos(self, u: np.ndarray) -> float:
        """Sync halo regions with neighbors."""
        t0 = MPI.Wtime()
        self.grid.sync_halos(u)
        halo_time = MPI.Wtime() - t0
        self.timeseries.halo_exchange_times.append(halo_time)
        return halo_time

    def _apply_boundary_conditions(self, u: np.ndarray):
        """Apply Dirichlet boundary conditions at physical boundaries."""
        self.grid.apply_boundary_conditions(u)

    def _compute_residual(
        self, u: np.ndarray, u_old: np.ndarray, n_interior: int
    ) -> float:
        """Compute global residual norm."""
        local_diff_sq_sum = np.sum(
            (u[1:-1, 1:-1, 1:-1] - u_old[1:-1, 1:-1, 1:-1]) ** 2
        )
        global_diff_sq_sum = self.comm.allreduce(local_diff_sq_sum, op=MPI.SUM)
        # RMS is sqrt(sum of squares / N)
        return np.sqrt(global_diff_sq_sum / self.grid.N_interior_global)

    def _finalize(self, wall_time: float, u_solution: np.ndarray):
        """Finalize results (rank 0 only for some metrics)."""
        if self.rank == 0:
            self.results.final_residual = self.timeseries.residual_history[-1]
            self.results.total_compute_time = sum(self.timeseries.compute_times)
            self.results.total_halo_time = sum(self.timeseries.halo_exchange_times)

        self._compute_metrics(wall_time, self.results.iterations)
        self.u = u_solution # Ensure the solution is stored

    def compute_l2_error(self) -> float:
        """Compute global L2 error using DistributedGrid."""
        l2_error = self.grid.compute_l2_error(self.u)
        self.results.final_error = l2_error
        return l2_error

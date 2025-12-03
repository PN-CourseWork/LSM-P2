"""Sequential Jacobi Solver."""

import numpy as np

from .base import BaseSolver
from ..kernels import NumPyKernel, NumbaKernel
from ..problems import sinusoidal_source_term


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
                omega=self.omega,
                specified_numba_threads=self.specified_numba_threads,
            )
        else:
            self.kernel = NumPyKernel(omega=self.omega)

    def _init_arrays(self):
        """Allocate solution arrays."""
        self.u = np.zeros((self.N, self.N, self.N), dtype=np.float64)
        self.u_old = np.zeros((self.N, self.N, self.N), dtype=np.float64)
        self.f = sinusoidal_source_term(self.N)

    def solve(self):
        """Run Jacobi iteration."""
        self._reset()

        n_interior = (self.N - 2) ** 3
        u, u_old = self.u, self.u_old

        self._barrier()  # Sync all ranks before timing
        t_start = self._get_time()

        for i in range(self.max_iter):
            # Sync halos (no-op for sequential)
            t0 = self._get_time()
            self._sync_halos(u_old)
            self._time_halo += self._get_time() - t0

            # Compute step
            t0 = self._get_time()
            self.kernel.step(u_old, u, self.f, self.h)
            self._apply_boundary_conditions(u)
            self._time_compute += self._get_time() - t0

            # Compute iterative residual
            residual = self._compute_iter_res(u, u_old, n_interior)
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

        # Save final solution (u_old has it after swap)
        self.u = u_old

        wall_time = self._get_time() - t_start
        self._finalize(wall_time)

        return self.metrics

    def _compute_iter_res(
        self, u: np.ndarray, u_old: np.ndarray, n_interior: int
    ) -> float:
        """Compute iterative residual ||u - u_old||."""
        diff = u[1:-1, 1:-1, 1:-1] - u_old[1:-1, 1:-1, 1:-1]
        local_sum = np.sum(diff**2)
        global_sum = self._reduce_sum(local_sum)
        return np.sqrt(global_sum) / n_interior

"""Sequential Jacobi solver."""

import time
import socket
from datetime import datetime
import numpy as np
from mpi4py import MPI
from .base import PoissonSolver
from .datastructures import RuntimeConfig, GlobalResults, PerRankResults


class SequentialJacobi(PoissonSolver):
    """Sequential Jacobi solver (single-node, no domain decomposition)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def solve(self, u1, u2, f, h, max_iter, tolerance=1e-8, u_true=None):
        """Solve using sequential Jacobi iteration."""
        N = u1.shape[0]
        converged = False
        compute_times = []
        t_start = time.perf_counter()

        # Main iteration loop
        for i in range(max_iter):
            if i % 2 == 0:
                uold, u = u1, u2
            else:
                u, uold = u1, u2

            # Jacobi step
            t_comp_start = time.perf_counter()
            residual = self._step(uold, u, f, h, self.config.omega)
            t_comp_end = time.perf_counter()
            compute_times.append(t_comp_end - t_comp_start)

            # Check convergence
            if residual < tolerance:
                converged = True
                if self.config.verbose:
                    print(f"Converged at iteration {i + 1} (residual: {residual:.2e})")
                break

        elapsed_time = time.perf_counter() - t_start

        if not converged and self.config.verbose:
            print(f"Did not converge after {max_iter} iterations (residual: {residual:.2e})")

        # Compute error
        final_error = None
        if u_true is not None:
            final_error = self.compute_error(u, u_true)
            if self.config.verbose:
                print(f"Final error vs true solution: {final_error:.2e}")

        # Build results
        runtime_config = RuntimeConfig(
            N=N,
            h=h,
            method="sequential_jacobi",
            omega=self.config.omega,
            tolerance=tolerance,
            max_iter=max_iter,
            use_numba=self.config.use_numba,
            num_threads=self.get_num_threads(self.config.use_numba),
            mpi_size=1,
            timestamp=datetime.now().isoformat(),
        )

        global_results = GlobalResults(
            iterations=i + 1,
            converged=converged,
            final_residual=residual,
            final_error=final_error or 0.0,
            wall_time=elapsed_time,
            compute_time=sum(compute_times),
            mpi_comm_time=0.0,
        )

        per_rank_results = PerRankResults(
            mpi_rank=0,
            hostname=socket.gethostname(),
            wall_time=elapsed_time,
            compute_time=sum(compute_times),
            mpi_comm_time=0.0,
        )

        return u, runtime_config, global_results, per_rank_results

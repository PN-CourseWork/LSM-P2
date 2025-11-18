"""Sequential Jacobi solver."""

import time
import socket
import numpy as np
from .base import PoissonSolver
from .datastructures import RuntimeConfig, GlobalResults, PerRankResults


class SequentialJacobi(PoissonSolver):
    """Sequential Jacobi solver (single-node, no domain decomposition)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config.method = "sequential"



    def method_solve(self):
        """Solve using sequential Jacobi iteration.

        Results are stored in solver instance variables and nothing is returned.
        """
        super().method_solve()


        # Clear runtime accumulation lists
        self.compute_times.clear()
        self.residual_history.clear()

        t_start = time.perf_counter()

        # Main iteration loop
        for i in range(self.config.max_iter):
            if i % 2 == 0:
                uold, u = u1, u2
            else:
                u, uold = u1, u2

            # Jacobi step
            t_comp_start = time.perf_counter()
            residual = self._step(uold, u, f, h, self.config.omega)
            t_comp_end = time.perf_counter()

            self.compute_times.append(t_comp_end - t_comp_start)
            self.residual_history.append(float(residual))

            # Check convergence
            if residual < self.config.tolerance:
                self.global_results.converged = True
                break

        elapsed_time = time.perf_counter() - t_start

        # Compute error
        final_error = float(np.linalg.norm(u - self.problem.u_exact))

               

        # Store all per-rank results and aggregate timings
        self.all_per_rank_results = [self.per_rank_results]
        timings = self._aggregate_timing_results(self.all_per_rank_results)

        self.global_results = GlobalResults(
            iterations=i + 1, residual_history=self.residual_history,
            converged=converged, final_error=final_error, **timings
        )

        # Store solution grid
        self.u = u

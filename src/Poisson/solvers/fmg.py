"""Full Multigrid (FMG) Solver - Sequential."""

import time
from typing import List

import numpy as np

from .base import BaseSolver
from ..datastructures import GridLevel, GlobalMetrics
from ..kernels import NumPyKernel, NumbaKernel
from ..problems import sinusoidal_source_term
from .multigrid_operators import restrict, prolong


class FMGSolver(BaseSolver):
    """Sequential Full Multigrid solver.

    Implements FMG algorithm without MPI.
    Useful for validation and single-node performance testing.

    Parameters
    ----------
    N : int
        Grid size (N x N x N). Must satisfy (N-1) divisible by 2^levels.
    n_smooth : int
        Pre/post smoothing iterations (default: 3).
    fmg_post_vcycles : int
        V-cycles after FMG phase (default: 1).
    use_numba : bool
        Use Numba JIT kernel (default: False).
    omega : float
        Relaxation factor (default: 2/3).
    max_iter : int
        Maximum smoothing iterations for coarsest grid solve (default: 1000).
    tolerance : float
        Convergence tolerance for final residual check (default: 1e-6).
    """

    def __init__(
        self,
        N: int,
        n_smooth: int = 3,
        fmg_post_vcycles: int = 1,
        use_numba: bool = False,
        omega: float = 2 / 3,
        specified_numba_threads: int = 1,
        **kwargs,
    ):
        super().__init__(N, omega=omega, **kwargs)

        self.n_smooth = n_smooth
        self.fmg_post_vcycles = fmg_post_vcycles
        self.use_numba = use_numba
        self.specified_numba_threads = specified_numba_threads
        self.min_coarse_size = 3

        # Infer number of levels
        self.n_levels = self._infer_levels(N)

        # Build grid hierarchy
        self.levels: List[GridLevel] = []
        self._build_hierarchy()

        # Timing accumulators
        self._time_compute = 0.0
        self._time_halo = 0.0

    def _infer_levels(self, N: int) -> int:
        """Determine number of grid levels."""
        levels = 1
        N_current = N

        while True:
            if (N_current - 1) % 2 != 0:
                raise ValueError(
                    f"Grid size N={N_current} not compatible with multigrid "
                    "(N-1 must be divisible by 2)."
                )

            N_next = (N_current - 1) // 2 + 1
            if N_next < self.min_coarse_size:
                break

            levels += 1
            N_current = N_next

            if N_current == 3:
                break

        return levels

    def _build_hierarchy(self):
        """Allocate arrays for all grid levels."""
        N_values = []
        N_current = self.N
        for _ in range(self.n_levels):
            N_values.append(N_current)
            N_current = (N_current - 1) // 2 + 1

        KernelClass = NumbaKernel if self.use_numba else NumPyKernel

        for level, N in enumerate(N_values):
            h = 2.0 / (N - 1)
            kernel = KernelClass(
                N=N,
                omega=self.omega,
                specified_numba_threads=self.specified_numba_threads,
            )

            lvl = GridLevel(
                level=level,
                N=N,
                h=h,
                u=np.zeros((N, N, N), dtype=np.float64),
                u_temp=np.zeros((N, N, N), dtype=np.float64),
                f=sinusoidal_source_term(N),
                r=np.zeros((N, N, N), dtype=np.float64),
                kernel=kernel,
            )
            self.levels.append(lvl)

    def _get_kernel(self):
        """Return the kernel for warmup (finest level)."""
        return self.levels[0].kernel

    def _get_time(self) -> float:
        """Get current time. Override for MPI timing."""
        return time.perf_counter()

    def _sync_halos(self, u, lvl) -> float:
        """Sync halo regions. No-op for sequential."""
        return 0.0

    def _apply_boundary_conditions(self, u, lvl):
        """Apply boundary conditions. No-op for sequential (arrays include boundaries)."""
        pass

    def solve(self) -> GlobalMetrics:
        """Full Multigrid: solve from coarsest to finest."""
        self._reset()

        t_start = self._get_time()

        # Clear all solution arrays
        for lvl in self.levels:
            lvl.u.fill(0.0)
            lvl.u_temp.fill(0.0)
            lvl.r.fill(0.0)

        # Solve on coarsest level
        self._coarse_solve(self.levels[-1])

        # Ascend hierarchy
        for level in reversed(range(self.n_levels - 1)):
            coarse = self.levels[level + 1]
            fine = self.levels[level]

            # Prolong coarse solution as initial guess
            self._sync_halos(coarse.u, coarse)
            fine.r.fill(0.0)
            prolong(coarse.u, fine.r)
            self._sync_halos(fine.r, fine)

            fine.u.fill(0.0)
            fine.u[1:-1, 1:-1, 1:-1] = fine.r[1:-1, 1:-1, 1:-1]
            self._apply_boundary_conditions(fine.u, fine)

            # Smooth
            for _ in range(self.n_smooth):
                self._smooth(fine)

            # V-cycle from this level
            residual = self._v_cycle(level)

        # Post V-cycles with residual tracking
        for _ in range(self.fmg_post_vcycles):
            residual = self._v_cycle(0)
            self.timeseries.residual_history.append(residual)

        self.metrics.converged = residual < self.tolerance
        self.metrics.iterations = self.n_levels + self.fmg_post_vcycles

        wall_time = self._get_time() - t_start
        self._finalize(wall_time)
        return self.metrics

    def _v_cycle(self, level: int) -> float:
        """Recursive V-cycle."""
        lvl = self.levels[level]

        # Pre-smoothing
        for _ in range(self.n_smooth):
            self._smooth(lvl)

        # Compute residual
        self._compute_residual(lvl)

        # Get residual norm on finest level
        residual = self._residual_norm(lvl.r, lvl) if level == 0 else 0.0

        # Base case: coarsest level
        if level == self.n_levels - 1:
            self._coarse_solve(lvl)
            return residual

        # Restriction
        next_lvl = self.levels[level + 1]
        self._sync_halos(lvl.r, lvl)
        restrict(lvl.r, next_lvl.f)
        self._sync_halos(next_lvl.f, next_lvl)
        next_lvl.u.fill(0.0)

        # Recurse
        self._v_cycle(level + 1)

        # Prolongation
        self._sync_halos(next_lvl.u, next_lvl)
        lvl.r.fill(0.0)
        prolong(next_lvl.u, lvl.r)
        self._sync_halos(lvl.r, lvl)

        lvl.u[1:-1, 1:-1, 1:-1] += lvl.r[1:-1, 1:-1, 1:-1]
        self._apply_boundary_conditions(lvl.u, lvl)

        # Post-smoothing
        for _ in range(self.n_smooth):
            self._smooth(lvl)

        return residual

    def _smooth(self, lvl: GridLevel):
        """One Jacobi smoothing step."""
        self._sync_halos(lvl.u, lvl)
        t0 = self._get_time()
        lvl.kernel.step(lvl.u, lvl.u_temp, lvl.f)
        self._apply_boundary_conditions(lvl.u_temp, lvl)
        self._time_compute += self._get_time() - t0
        lvl.u, lvl.u_temp = lvl.u_temp, lvl.u

    def _compute_residual(self, lvl: GridLevel):
        """Compute r = f - Au."""
        u, f, r = lvl.u, lvl.f, lvl.r
        h2 = lvl.h * lvl.h

        self._sync_halos(u, lvl)

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

        # Zero residual at physical boundaries
        self._apply_boundary_conditions(r, lvl)

    def _coarse_solve(self, lvl: GridLevel):
        """Solve on coarsest grid using max_iter smoothing iterations."""
        for _ in range(self.max_iter):
            self._smooth(lvl)

    def _residual_norm(self, r: np.ndarray, lvl: GridLevel = None) -> float:
        """Compute RMS residual norm. Override for MPI allreduce."""
        interior = r[1:-1, 1:-1, 1:-1]
        n_pts = interior.size
        return np.sqrt(np.sum(interior**2)) / n_pts

    def _reset(self):
        """Reset timers and timeseries."""
        self._time_compute = 0.0
        self._time_halo = 0.0
        self.timeseries.clear()

    def _finalize(self, wall_time: float):
        """Finalize metrics."""
        self.metrics.final_residual = (
            self.timeseries.residual_history[-1]
            if self.timeseries.residual_history
            else 0.0
        )
        self.metrics.total_compute_time = self._time_compute
        self.metrics.total_halo_time = self._time_halo
        # Get observed numba threads from finest level kernel
        self.metrics.observed_numba_threads = self.levels[
            0
        ].kernel.observed_numba_threads
        self._compute_metrics(wall_time, self.metrics.iterations)

    def _get_solution_array(self) -> np.ndarray:
        """Return the solution array (finest level)."""
        return self.levels[0].u

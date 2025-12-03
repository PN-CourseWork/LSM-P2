"""MPI-parallel Full Multigrid Solver (extends FMGSolver)."""

import numpy as np
from mpi4py import MPI

from .fmg import FMGSolver
from ..datastructures import GridLevel
from ..kernels import NumPyKernel, NumbaKernel
from ..mpi.grid import DistributedGrid


class FMGMPISolver(FMGSolver):
    """Parallel Full Multigrid solver with MPI domain decomposition.

    Extends FMGSolver with distributed grids and halo exchange.
    The algorithm (solve, fmg_solve, _v_cycle) is inherited from FMGSolver.

    Parameters
    ----------
    N : int
        Global grid size (N x N x N).
    strategy : str
        Decomposition: 'sliced' or 'cubic' (default: 'sliced').
    communicator : str
        Halo exchange: 'numpy' or 'custom' (default: 'custom').
    n_smooth : int
        Pre/post smoothing iterations (default: 3).
    fmg_post_vcycles : int
        V-cycles after FMG phase (default: 1).
    use_numba : bool
        Use Numba JIT kernel (default: False).
    omega : float
        Relaxation factor (default: 2/3).
    max_iter : int
        Maximum V-cycles (default: 100).
    tolerance : float
        Convergence tolerance (default: 1e-6).
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

        # Parent init builds hierarchy (calls _build_hierarchy)
        super().__init__(N, **kwargs)

        # Store config info from finest level
        fine = self.levels[0]
        self.local_shape = fine.grid.local_shape
        self.halo_size_mb = fine.grid.get_halo_size_bytes() / (1024 * 1024)

    def _infer_levels(self, N: int) -> int:
        """Determine number of levels considering MPI constraints."""
        # Minimum local interior for restriction/prolongation
        min_local = 3

        if self.size > 1:
            if self.strategy == "cubic":
                ranks_per_dim = int(round(self.size ** (1 / 3)))
                min_N_for_ranks = max(
                    (min_local + 2) * ranks_per_dim, self.min_coarse_size
                )
            else:
                min_N_for_ranks = self.size + 2
        else:
            min_N_for_ranks = 3

        levels = 1
        N_current = N

        while True:
            if (N_current - 1) % 2 != 0:
                raise ValueError(
                    f"Grid size N={N_current} not compatible with multigrid "
                    "(N-1 must be divisible by 2)."
                )

            N_next = (N_current - 1) // 2 + 1

            # Stop if next level would be too small for MPI
            if self.size > 1 and N_next < min_N_for_ranks:
                break

            if N_next < self.min_coarse_size:
                break

            levels += 1
            N_current = N_next

            if N_current == 3:
                break

        return levels

    def _build_hierarchy(self):
        """Build grid hierarchy with distributed grids."""
        N_values = []
        N_current = self.N
        for _ in range(self.n_levels):
            N_values.append(N_current)
            N_current = (N_current - 1) // 2 + 1

        KernelClass = NumbaKernel if self.use_numba else NumPyKernel

        for level, N in enumerate(N_values):
            h = 2.0 / (N - 1)

            # Create distributed grid
            grid = DistributedGrid(
                N, self.comm, strategy=self.strategy, halo_exchange=self.communicator
            )

            # Allocate arrays via grid
            u = grid.allocate()
            u_temp = grid.allocate()
            f = grid.allocate()
            r = grid.allocate()

            # Fill source term and apply BCs
            grid.fill_source_term(f)
            grid.apply_boundary_conditions(u)
            grid.apply_boundary_conditions(u_temp)

            kernel = KernelClass(N=N, omega=self.omega)

            lvl = GridLevel(
                level=level,
                N=N,
                h=h,
                u=u,
                u_temp=u_temp,
                f=f,
                r=r,
                kernel=kernel,
                grid=grid,
            )
            self.levels.append(lvl)

    # ========================================================================
    # Hook method overrides for MPI
    # ========================================================================

    def _get_time(self) -> float:
        """Get current time using MPI.Wtime()."""
        return MPI.Wtime()

    def _sync_halos(self, u: np.ndarray, lvl: GridLevel) -> float:
        """Sync halo regions with neighbors."""
        if lvl.grid is not None:
            t0 = MPI.Wtime()
            lvl.grid.sync_halos(u)
            halo_time = MPI.Wtime() - t0
            self._time_halo += halo_time
            return halo_time
        return 0.0

    def _apply_boundary_conditions(self, u: np.ndarray, lvl: GridLevel):
        """Apply Dirichlet boundary conditions at physical boundaries."""
        if lvl.grid is not None:
            lvl.grid.apply_boundary_conditions(u)

    def _residual_norm(self, r: np.ndarray, lvl: GridLevel = None) -> float:
        """Compute global RMS residual norm via allreduce."""
        interior = r[1:-1, 1:-1, 1:-1]
        local_sum_sq = np.sum(interior**2)
        local_pts = float(interior.size)

        local_data = np.array([local_sum_sq, local_pts])
        global_data = np.empty(2)
        self.comm.Allreduce(local_data, global_data, op=MPI.SUM)

        return np.sqrt(global_data[0]) / global_data[1]

    def _finalize(self, wall_time: float):
        """Finalize results (rank 0 only for some metrics)."""
        if self.rank == 0:
            self.results.final_residual = (
                self.timeseries.residual_history[-1]
                if self.timeseries.residual_history
                else 0.0
            )
            self.results.total_compute_time = self._time_compute
            self.results.total_halo_time = self._time_halo

        self._compute_metrics(wall_time, self.results.iterations)

    def compute_l2_error(self) -> float:
        """Compute L2 error against analytical solution (parallel)."""
        fine = self.levels[0]
        l2_error = fine.grid.compute_l2_error(fine.u)
        self.results.final_error = l2_error
        return l2_error

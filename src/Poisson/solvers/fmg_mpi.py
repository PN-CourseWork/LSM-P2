"""MPI-parallel Full Multigrid Solver (extends FMGSolver)."""

import numpy as np
from mpi4py import MPI

from .fmg import FMGSolver
from .mpi_mixin import MPISolverMixin
from ..datastructures import GridLevel
from ..mpi.grid import DistributedGrid


class FMGMPISolver(MPISolverMixin, FMGSolver):
    """Parallel Full Multigrid solver with MPI domain decomposition.

    Extends FMGSolver with distributed grids and halo exchange.
    The algorithm (solve, _v_cycle) is inherited from FMGSolver.

    Parameters
    ----------
    N : int
        Global grid size (N x N x N).
    strategy : str
        Decomposition: 'sliced' or 'cubic' (default: 'sliced').
    communicator : str
        Halo exchange: 'numpy' or 'custom' (default: 'custom').
    """

    def __init__(
        self,
        N: int,
        strategy: str = "sliced",
        communicator: str = "custom",
        **kwargs,
    ):
        # MPI setup before parent init
        self._init_mpi()

        self.strategy = strategy
        self.communicator = communicator

        # Parent init builds hierarchy (calls _build_hierarchy)
        super().__init__(N, **kwargs)

    def _infer_levels(self, N: int) -> int:
        """Determine number of levels considering MPI constraints."""
        min_local = 2  # Minimum local interior for restriction/prolongation

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

            lvl = GridLevel(
                level=level,
                N=N,
                h=h,
                u=u,
                u_temp=u_temp,
                f=f,
                r=r,
                kernel=None,  # Single kernel at solver level
                grid=grid,
            )
            self.levels.append(lvl)

    # ========================================================================
    # Hook method overrides for MPI (grid-level specific)
    # ========================================================================

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

    def compute_l2_error(self) -> float:
        """Compute L2 error against analytical solution (parallel)."""
        fine = self.levels[0]
        l2_error = fine.grid.compute_l2_error(fine.u)
        self.metrics.final_error = l2_error
        return l2_error

    def _gather_topology(self):
        """Gather rank topology from all MPI ranks."""
        fine = self.levels[0]
        rank_info = fine.grid.get_rank_info()
        all_ranks = self.comm.gather(rank_info, root=0)
        return all_ranks if self.rank == 0 else None

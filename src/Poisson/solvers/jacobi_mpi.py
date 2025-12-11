"""MPI-parallel Jacobi Solver (extends JacobiSolver)."""

import numpy as np
from mpi4py import MPI

from .jacobi import JacobiSolver
from .mpi_mixin import MPISolverMixin
from ..mpi.grid import DistributedGrid


class JacobiMPISolver(MPISolverMixin, JacobiSolver):
    """Parallel Jacobi solver with MPI domain decomposition.

    Extends JacobiSolver with MPI communication for distributed solving.
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

        # Create distributed grid
        self.grid = DistributedGrid(
            N, self.comm, strategy=strategy, halo_exchange=communicator
        )

        # Parent init (calls _init_kernel and _init_arrays)
        super().__init__(N, **kwargs)

    def _init_arrays(self):
        """Allocate local arrays using distributed grid."""
        self.u = self.grid.allocate()
        self.u_old = self.grid.allocate()
        self.f = self.grid.allocate()

        # Fill source term and apply BCs
        self.grid.fill_source_term(self.f)
        self.grid.apply_boundary_conditions(self.u)
        self.grid.apply_boundary_conditions(self.u_old)

    def _sync_halos(self, u: np.ndarray, lvl=None) -> float:
        """Sync halo regions with neighbors."""
        t0 = MPI.Wtime()
        self.grid.sync_halos(u)
        return MPI.Wtime() - t0

    def _apply_boundary_conditions(self, u: np.ndarray, lvl=None):
        """Apply Dirichlet boundary conditions at physical boundaries."""
        self.grid.apply_boundary_conditions(u)

    def compute_l2_error(self) -> float:
        """Compute L2 error using grid's parallel implementation."""
        l2_error = self.grid.compute_l2_error(self.u)
        self.metrics.final_error = l2_error
        return l2_error

    def _gather_topology(self):
        """Gather rank topology from all MPI ranks."""
        rank_info = self.grid.get_rank_info()
        all_ranks = self.comm.gather(rank_info, root=0)
        return all_ranks if self.rank == 0 else None

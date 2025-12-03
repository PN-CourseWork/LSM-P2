"""MPI-parallel Jacobi Solver (extends JacobiSolver)."""

import numpy as np
from mpi4py import MPI

from .jacobi import JacobiSolver
from ..mpi.grid import DistributedGrid


class JacobiMPISolver(JacobiSolver):
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
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.strategy = strategy
        self.communicator = communicator

        # Create distributed grid
        self.grid = DistributedGrid(
            N, self.comm, strategy=strategy, halo_exchange=communicator
        )

        # Store config info
        self.local_shape = self.grid.local_shape
        self.halo_size_mb = self.grid.get_halo_size_bytes() / (1024 * 1024)

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

    def _get_time(self) -> float:
        """Get current time using MPI.Wtime()."""
        return MPI.Wtime()

    def _sync_halos(self, u: np.ndarray) -> float:
        """Sync halo regions with neighbors."""
        t0 = MPI.Wtime()
        self.grid.sync_halos(u)
        return MPI.Wtime() - t0

    def _apply_boundary_conditions(self, u: np.ndarray):
        """Apply Dirichlet boundary conditions at physical boundaries."""
        self.grid.apply_boundary_conditions(u)

    def _compute_residual(
        self, u: np.ndarray, u_old: np.ndarray, n_interior: int
    ) -> float:
        """Compute global residual norm via MPI allreduce."""
        diff = u[1:-1, 1:-1, 1:-1] - u_old[1:-1, 1:-1, 1:-1]
        local_sum = np.sum(diff**2)

        global_sum = np.zeros(1)
        self.comm.Allreduce(np.array([local_sum]), global_sum, op=MPI.SUM)

        return np.sqrt(global_sum[0]) / n_interior

    def _finalize(self, wall_time: float, u_solution: np.ndarray):
        """Finalize metrics after solve (rank 0 only for some metrics)."""
        self.u = u_solution

        if self.rank == 0:
            self.metrics.final_residual = self.timeseries.residual_history[-1]
            self.metrics.total_compute_time = sum(self.timeseries.compute_times)
            self.metrics.total_halo_time = sum(self.timeseries.halo_times)

        self._compute_metrics(wall_time, self.metrics.iterations)

    def _get_solution_array(self) -> np.ndarray:
        """Return the solution array."""
        return self.u

    def compute_l2_error(self) -> float:
        """Compute L2 error against analytical solution (parallel).

        Overrides base to use grid's parallel implementation.
        """
        l2_error = self.grid.compute_l2_error(self.u)
        self.metrics.final_error = l2_error
        return l2_error

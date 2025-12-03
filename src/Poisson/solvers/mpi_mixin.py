"""MPI mixin providing common parallel solver functionality."""

import numpy as np
from mpi4py import MPI


class MPISolverMixin:
    """Mixin providing common MPI functionality for parallel solvers.

    Provides shared implementations for:
    - MPI initialization (comm, rank, size)
    - Timing via MPI.Wtime()
    - Global reductions via MPI.Allreduce()
    - Root rank checking

    Usage:
        class MyMPISolver(MPISolverMixin, MySolver):
            def __init__(self, ...):
                self._init_mpi()
                ...
    """

    def _init_mpi(self):
        """Initialize MPI attributes. Call early in subclass __init__."""
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def _get_time(self) -> float:
        """Get current time using MPI.Wtime()."""
        return MPI.Wtime()

    def _reduce_sum(self, local_sum: float) -> float:
        """Reduce sum via MPI Allreduce."""
        global_sum = np.zeros(1)
        self.comm.Allreduce(np.array([local_sum]), global_sum, op=MPI.SUM)
        return global_sum[0]

    def _is_root(self) -> bool:
        """Only rank 0 logs metrics."""
        return self.rank == 0

    def _barrier(self):
        """Synchronize all ranks before timing."""
        self.comm.Barrier()

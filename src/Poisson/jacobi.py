"""Unified Jacobi solver for Poisson equation.

This solver handles both sequential and distributed (MPI) execution with a
single interface. Sequential execution is the default, and distributed execution
is enabled by providing decomposition and communicator strategies.
"""

import socket
import numpy as np
from mpi4py import MPI
from .base import PoissonSolver
from .datastructures import LocalResults, LocalFields
from .strategies import (
    SlicedDecomposition,
    CubicDecomposition,
    CustomMPICommunicator,
    NumpyCommunicator,
)


class JacobiPoisson(PoissonSolver):
    """Unified Jacobi solver for sequential and distributed execution.

    This solver provides a single interface for both sequential and MPI-parallel
    execution. Sequential execution is the default. For distributed execution,
    specify decomposition and communicator strategies.

    Parameters
    ----------
    decomposition : str or DecompositionStrategy, optional
        Domain decomposition strategy: "sliced", "cubic", or strategy instance.
        If None (default), runs in sequential mode.
    communicator : str or CommunicatorStrategy, optional
        Ghost exchange strategy: "custom", "numpy", or strategy instance.
        Required if decomposition is specified.
    **kwargs
        Additional arguments passed to PoissonSolver base class
        (N, omega, use_numba, max_iter, tolerance, etc.)

    Examples
    --------
    Sequential execution:
    >>> solver = JacobiPoisson(N=100, omega=0.75)
    >>> solver.solve()

    Distributed execution (must run with mpiexec):
    >>> solver = JacobiPoisson(decomposition="cubic", communicator="custom",
    ...                         N=100, omega=0.75)
    >>> solver.solve()
    """

    def __init__(self, decomposition=None, communicator=None, **kwargs):
        super().__init__(**kwargs)

        # MPI setup (always needed for rank checking)
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Determine execution mode
        if decomposition is None:
            # Sequential mode
            self.is_distributed = False
            kernel_name = "Numba" if self.config.use_numba else "NumPy"
            self.config.method = f"Sequential ({kernel_name})"
            self.decomposition = None
            self.communicator = None
        else:
            # Distributed mode
            self.is_distributed = True

            if communicator is None:
                raise ValueError("communicator must be specified when using decomposition")

            # Instantiate decomposition strategy
            if isinstance(decomposition, str):
                decomposition = decomposition.lower()
                if decomposition == "sliced":
                    self.decomposition = SlicedDecomposition()
                elif decomposition == "cubic":
                    self.decomposition = CubicDecomposition()
                else:
                    raise ValueError(f"Unknown decomposition strategy: {decomposition}")
            else:
                self.decomposition = decomposition

            # Instantiate communicator strategy
            if isinstance(communicator, str):
                communicator = communicator.lower()
                if communicator == "custom":
                    self.communicator = CustomMPICommunicator()
                elif communicator == "numpy":
                    self.communicator = NumpyCommunicator()
                else:
                    raise ValueError(f"Unknown communicator strategy: {communicator}")
            else:
                self.communicator = communicator

            # Set method name
            decomp_name = self.decomposition.__class__.__name__.replace("Decomposition", "")
            comm_name = self.communicator.__class__.__name__.replace("Communicator", "")
            self.config.method = f"MPI_{decomp_name}_{comm_name}"

        # Initialize local fields
        self.local_fields = LocalFields()

    def method_solve(self):
        """Solve using Jacobi iteration.

        For sequential mode (decomposition=None), rank 0 owns the entire domain.
        For distributed mode, domain is decomposed across all ranks.
        """
        # Only rank 0 runs in sequential mode
        if not self.is_distributed and self.rank != 0:
            return

        # Broadcast N to all ranks (in distributed mode)
        N = self.config.N
        if self.is_distributed:
            N = self.comm.bcast(N, root=0)
        h = 2.0 / (N - 1)

        # Initialize local arrays
        if self.is_distributed:
            # Decomposed domain initialization
            if isinstance(self.decomposition, CubicDecomposition):
                u1_local, u2_local, f_local = self.decomposition.initialize_local_arrays_distributed(
                    N, self.rank, self.comm
                )
            elif isinstance(self.decomposition, SlicedDecomposition):
                u1_local, u2_local, f_local = self.decomposition.initialize_local_arrays_distributed(
                    N, self.rank, self.size
                )
            else:
                raise ValueError(f"Unknown decomposition strategy: {type(self.decomposition)}")
        else:
            # Sequential: rank 0 owns entire domain
            u1_local = self.global_fields.u1
            u2_local = self.global_fields.u2
            f_local = self.global_fields.f

        # Store in local_fields
        self.local_fields.u1 = u1_local
        self.local_fields.u2 = u2_local

        # Main iteration loop
        for i in range(self.config.max_iter):
            if i % 2 == 0:
                uold_local, u_local = u1_local, u2_local
            else:
                u_local, uold_local = u1_local, u2_local

            # Exchange ghost zones (only in distributed mode)
            if self.is_distributed:
                t_comm_start = MPI.Wtime()
                self._exchange_ghosts(uold_local)
                t_comm_end = MPI.Wtime()
                self.global_timeseries.halo_exchange_times.append(t_comm_end - t_comm_start)

            # Jacobi step on local domain
            t_comp_start = MPI.Wtime()
            local_residual = self._step(uold_local, u_local, f_local, h, self.config.omega)
            t_comp_end = MPI.Wtime()
            self.global_timeseries.compute_times.append(t_comp_end - t_comp_start)

            # Compute global residual
            if self.is_distributed:
                # Reduction across all ranks
                t_comm_start = MPI.Wtime()
                global_residual = self.comm.allreduce(local_residual**2, op=MPI.SUM)
                global_residual = np.sqrt(global_residual)
                t_comm_end = MPI.Wtime()
                self.global_timeseries.mpi_comm_times.append(t_comm_end - t_comm_start)
            else:
                # Sequential: local residual is global residual
                global_residual = local_residual

            # Store residual history on rank 0
            if self.rank == 0:
                self.global_timeseries.residual_history.append(float(global_residual))

            # Check convergence
            if global_residual < self.config.tolerance:
                if self.rank == 0:
                    self.global_results.converged = True
                    self.global_results.iterations = i + 1
                break
        else:
            if self.rank == 0:
                self.global_results.iterations = self.config.max_iter

        # Gather solution to rank 0 (only in distributed mode)
        if self.is_distributed:
            u_global = self._gather_solution(u_local, N)
        else:
            # Sequential: u_local is already u_global
            u_global = u_local

        # Compute error on rank 0
        if self.rank == 0:
            # Get exact solution
            if self.is_distributed:
                from .datastructures import sinusoidal_exact_solution
                u_exact = sinusoidal_exact_solution(N)
            else:
                u_exact = self.global_fields.u_exact

            error_diff = u_global - u_exact
            self.global_results.final_error = float(np.sqrt(h**3 * np.sum(error_diff**2)))

            # Store final solution in global_fields (if they exist)
            if hasattr(self, 'global_fields'):
                if self.global_results.iterations % 2 == 0:
                    self.global_fields.u1[:] = u_global
                else:
                    self.global_fields.u2[:] = u_global

        # Build per-rank results
        self.local_results.mpi_rank = self.rank
        self.local_results.hostname = socket.gethostname()
        self.local_results.compute_time = sum(self.global_timeseries.compute_times)
        if self.is_distributed:
            self.local_results.mpi_comm_time = sum(self.global_timeseries.mpi_comm_times)
            self.local_results.halo_exchange_time = sum(self.global_timeseries.halo_exchange_times)

    def _exchange_ghosts(self, u_local):
        """Exchange ghost zones using the communicator strategy."""
        if isinstance(self.decomposition, SlicedDecomposition):
            self.communicator.exchange_ghosts_sliced(
                u_local, self.decomposition, self.rank, self.size, self.comm
            )
        elif isinstance(self.decomposition, CubicDecomposition):
            self.communicator.exchange_ghosts_cubic(
                u_local, self.decomposition, self.rank, self.comm
            )
        else:
            raise ValueError(f"Unknown decomposition strategy: {type(self.decomposition)}")

    def _gather_solution(self, u_local, N):
        """Gather local solutions to global array on rank 0.

        Delegates to the decomposition strategy's gather_solution method.
        """
        return self.decomposition.gather_solution(u_local, N, self.rank, self.comm)

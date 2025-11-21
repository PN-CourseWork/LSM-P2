"""Unified Jacobi solver for Poisson equation.

This solver handles both sequential and distributed (MPI) execution with a
single interface. Sequential execution is the default, and distributed execution
is enabled by providing decomposition and communicator strategies.
"""

import socket
import numpy as np
from mpi4py import MPI
from .base import PoissonSolver
from .datastructures import LocalResults, LocalFields, sinusoidal_exact_solution
from .strategies import (
    NoDecomposition,
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

        # Instantiate decomposition strategy (NoDecomposition for sequential)
        if decomposition is None:
            # Sequential mode: entire domain on rank 0
            self.decomposition = NoDecomposition()
            # Use NumpyCommunicator (will handle empty exchange specs)
            self.communicator = NumpyCommunicator()
            kernel_name = "Numba" if self.config.use_numba else "NumPy"
            self.config.method = f"Sequential ({kernel_name})"
        else:
            # Distributed mode
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

        # Initialize local arrays using decomposition strategy
        N = self.config.N
        u1_local, u2_local, f_local = self.decomposition.initialize_local_arrays_distributed(
            N, self.rank, self.comm
        )

        # Store in local_fields (None for non-root ranks in sequential mode)
        self.local_fields.u1 = u1_local
        self.local_fields.u2 = u2_local
        self.local_fields.f = f_local

    def method_solve(self):
        """Solve using Jacobi iteration.

        Uses decomposition strategy to determine domain ownership.
        For NoDecomposition, rank 0 owns entire domain and others do nothing.
        For other decompositions, domain is distributed across all ranks.
        """
        # NoDecomposition returns None for non-root ranks - they should exit
        if self.local_fields.u1 is None:
            return

        # Get problem size and local arrays
        N = self.config.N
        h = 2.0 / (N - 1)
        u1_local = self.local_fields.u1
        u2_local = self.local_fields.u2
        f_local = self.local_fields.f

        # Main iteration loop
        for i in range(self.config.max_iter):
            if i % 2 == 0:
                uold_local, u_local = u1_local, u2_local
            else:
                u_local, uold_local = u1_local, u2_local

            # Exchange ghost zones (decomposition provides exchange spec)
            t_comm_start = MPI.Wtime()
            self.communicator.exchange_ghosts(uold_local, self.decomposition, self.rank, self.comm)
            t_comm_end = MPI.Wtime()
            if t_comm_end - t_comm_start > 0:
                self.global_timeseries.halo_exchange_times.append(t_comm_end - t_comm_start)

            # Jacobi step on local domain
            t_comp_start = MPI.Wtime()
            local_residual = self._step(uold_local, u_local, f_local, h, self.config.omega)
            t_comp_end = MPI.Wtime()
            self.global_timeseries.compute_times.append(t_comp_end - t_comp_start)

            # Compute global residual (reduction across active ranks)
            t_comm_start = MPI.Wtime()
            global_residual = self.comm.allreduce(local_residual**2, op=MPI.SUM)
            global_residual = np.sqrt(global_residual)
            t_comm_end = MPI.Wtime()
            if t_comm_end - t_comm_start > 0:
                self.global_timeseries.mpi_comm_times.append(t_comm_end - t_comm_start)

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

        # Gather solution to rank 0 using decomposition strategy
        u_global = self._gather_solution(u_local, N)

        # Store final solution in global_fields on rank 0
        if self.rank == 0 and hasattr(self, 'global_fields'):
            if self.global_results.iterations % 2 == 0:
                self.global_fields.u1[:] = u_global
            else:
                self.global_fields.u2[:] = u_global

        # Build per-rank results
        self.local_results.mpi_rank = self.rank
        self.local_results.hostname = socket.gethostname()
        self.local_results.compute_time = sum(self.global_timeseries.compute_times)
        self.local_results.mpi_comm_time = sum(self.global_timeseries.mpi_comm_times)
        self.local_results.halo_exchange_time = sum(self.global_timeseries.halo_exchange_times)

    def summary(self, exact_solution=None):
        """Compute summary statistics and error against exact solution.

        Parameters
        ----------
        exact_solution : callable or np.ndarray, optional
            Either a function that takes N and returns the exact solution,
            or the exact solution array itself. If None, uses the default
            sinusoidal exact solution.
        """
        if self.rank != 0:
            return

        N = self.config.N
        h = 2.0 / (N - 1)

        # Get the current solution from global_fields
        if self.global_results.iterations % 2 == 0:
            u_computed = self.global_fields.u1
        else:
            u_computed = self.global_fields.u2

        # Get exact solution
        if exact_solution is None:
            u_exact = sinusoidal_exact_solution(N)
        elif callable(exact_solution):
            u_exact = exact_solution(N)
        else:
            u_exact = exact_solution

        # Compute L2 error
        error_diff = u_computed - u_exact
        self.global_results.final_error = float(np.sqrt(h**3 * np.sum(error_diff**2)))

    def _gather_solution(self, u_local, N):
        """Gather local solutions to global array on rank 0.

        Uses decomposition strategy to extract interior and determine placement,
        then performs generic gather and reconstruction.

        Parameters
        ----------
        u_local : np.ndarray
            Local solution array with ghost zones
        N : int
            Global grid size

        Returns
        -------
        u_global : np.ndarray or None
            Global solution on rank 0, None on other ranks
        """
        # Extract interior points using decomposition strategy
        local_interior = self.decomposition.extract_interior(u_local)

        # Gather all interior arrays to rank 0
        all_interiors = self.comm.gather(local_interior, root=0)

        if self.rank == 0:
            u_global = np.zeros((N, N, N))

            # Place each rank's interior data using decomposition mapping
            for rank_id, interior_data in enumerate(all_interiors):
                placement = self.decomposition.get_interior_placement(rank_id, N, self.comm)
                u_global[placement] = interior_data

            return u_global
        else:
            return None

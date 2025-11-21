"""Unified Jacobi solver for Poisson equation.

This solver handles both sequential and distributed (MPI) execution with a
single interface. Sequential execution is the default, and distributed execution
is enabled by providing decomposition and communicator strategies.
"""

import socket
import numpy as np
from mpi4py import MPI
from .base import PoissonSolver
from .datastructures import sinusoidal_exact_solution
from .strategies import (
    NoDecomposition,
    SlicedDecomposition,
    CubicDecomposition,
    CustomMPICommunicator,
    NumpyCommunicator,
)

# Strategy registries
DECOMPOSITION_REGISTRY = {
    "sliced": SlicedDecomposition,
    "cubic": CubicDecomposition,
}

COMMUNICATOR_REGISTRY = {
    "custom": CustomMPICommunicator,
    "numpy": NumpyCommunicator,
}


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

        # MPI setup
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Setup strategies
        if decomposition is None:
            # Sequential mode: entire domain on rank 0
            self.decomposition = NoDecomposition()
            self.communicator = NumpyCommunicator()
            kernel_name = "Numba" if self.config.use_numba else "NumPy"
            self.config.method = f"Sequential ({kernel_name})"
        else:
            # Distributed mode - instantiate decomposition
            self.decomposition = self._create_strategy(
                decomposition, DECOMPOSITION_REGISTRY, "decomposition"
            )

            # Default communicator to "numpy" if not specified
            if communicator is None:
                communicator = "numpy"

            self.communicator = self._create_strategy(
                communicator, COMMUNICATOR_REGISTRY, "communicator"
            )

            # Set method name
            decomp_name = self.decomposition.__class__.__name__.replace("Decomposition", "")
            comm_name = self.communicator.__class__.__name__.replace("Communicator", "")
            self.config.method = f"MPI_{decomp_name}_{comm_name}"

        # Initialize local arrays using decomposition strategy
        N = self.config.N
        self.u1_local, self.u2_local, self.f_local = self.decomposition.initialize_local_arrays_distributed(
            N, self.rank, self.comm
        )

    def _create_strategy(self, strategy, registry, strategy_type):
        """Create a strategy instance from string or return existing instance.

        Parameters
        ----------
        strategy : str or object
            Strategy name or instance
        registry : dict
            Registry mapping names to classes
        strategy_type : str
            Type name for error messages

        Returns
        -------
        object
            Strategy instance
        """
        if isinstance(strategy, str):
            strategy_lower = strategy.lower()
            if strategy_lower not in registry:
                raise ValueError(
                    f"Unknown {strategy_type} strategy: {strategy}. "
                    f"Available: {list(registry.keys())}"
                )
            return registry[strategy_lower]()
        else:
            return strategy

    def method_solve(self):
        """Solve using Jacobi iteration.

        Uses decomposition strategy to determine domain ownership.
        For NoDecomposition, rank 0 owns entire domain and others do nothing.
        For other decompositions, domain is distributed across all ranks.
        """
        # NoDecomposition returns None for non-root ranks - they should exit
        if self.u1_local is None:
            return

        # Get problem size
        N = self.config.N
        h = 2.0 / (N - 1)

        # Initialize buffer pointers for ping-pong
        uold_local = self.u1_local
        u_local = self.u2_local

        # Main iteration loop
        for i in range(self.config.max_iter):
            # Exchange ghost zones
            self._time_operation(
                lambda: self.communicator.exchange_ghosts(uold_local, self.decomposition, self.rank, self.comm),
                self.global_timeseries.halo_exchange_times
            )

            # Jacobi step on local domain
            local_residual = self._time_operation(
                lambda: self._step(uold_local, u_local, self.f_local, h, self.config.omega),
                self.global_timeseries.compute_times
            )

            # Compute global residual
            global_residual = self._time_operation(
                lambda: np.sqrt(self.comm.allreduce(local_residual**2, op=MPI.SUM)),
                self.global_timeseries.mpi_comm_times
            )

            # Store residual history on rank 0
            if self.rank == 0:
                self.global_timeseries.residual_history.append(float(global_residual))

            # Check convergence
            if global_residual < self.config.tolerance:
                self._record_convergence(i + 1, converged=True)
                break

            # Swap buffers for next iteration
            uold_local, u_local = u_local, uold_local
        else:
            # Max iterations reached without convergence
            self._record_convergence(self.config.max_iter, converged=False)

        # Gather solution to rank 0
        if self.rank == 0:
            # Extract interior points using decomposition strategy
            local_interior = self.decomposition.extract_interior(u_local)
            all_interiors = self.comm.gather(local_interior, root=0)

            u_global = np.zeros((N, N, N))
            for rank_id, interior_data in enumerate(all_interiors):
                placement = self.decomposition.get_interior_placement(rank_id, N, self.comm)
                u_global[placement] = interior_data

            # Store final solution in global_fields
            if hasattr(self, 'global_fields'):
                self.global_fields.u1[:] = u_global
        else:
            # Non-root ranks just participate in gather
            local_interior = self.decomposition.extract_interior(u_local)
            self.comm.gather(local_interior, root=0)

        # Build per-rank results
        self._build_local_results()

    def _time_operation(self, operation, time_list):
        """Time an operation and append to time list.

        Parameters
        ----------
        operation : callable
            Operation to time
        time_list : list
            List to append timing to

        Returns
        -------
        result
            Result of operation
        """
        t_start = MPI.Wtime()
        result = operation()
        t_end = MPI.Wtime()
        time_list.append(t_end - t_start)
        return result

    def _record_convergence(self, iterations, converged):
        """Record convergence status on rank 0.

        Parameters
        ----------
        iterations : int
            Number of iterations performed
        converged : bool
            Whether the solver converged
        """
        if self.rank == 0:
            self.global_results.iterations = iterations
            self.global_results.converged = converged

    def _build_local_results(self):
        """Build per-rank results."""
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

        # Get the computed solution from global_fields
        u_computed = self.global_fields.u1

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

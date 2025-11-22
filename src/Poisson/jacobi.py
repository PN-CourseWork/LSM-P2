"""Unified Jacobi solver for Poisson equation.

This solver handles both sequential and distributed (MPI) execution with a
single interface. Sequential execution is the default, and distributed execution
is enabled by providing decomposition and communicator strategies.
"""

import socket
import numpy as np
from mpi4py import MPI
from numba import get_num_threads

from .kernels import jacobi_step_numpy, jacobi_step_numba
from .datastructures import (
    GlobalConfig,
    GlobalFields,
    GlobalResults,
    LocalResults,
    TimeSeriesLocal,
    TimeSeriesGlobal,
    sinusoidal_exact_solution,
)
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


# ============================================================================
# Module-level utilities
# ============================================================================

def _create_strategy(strategy, registry, strategy_type):
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


def _time_operation(operation, time_list):
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


# ============================================================================
# Main solver class
# ============================================================================

class JacobiPoisson:
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
        Additional arguments for solver configuration
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
        # MPI setup
        comm = MPI.COMM_WORLD
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        # Global configuration
        self.config = GlobalConfig(**kwargs)
        self.config.num_threads = get_num_threads()
        self.config.mpi_size = self.size

        # Local results and timeseries (all ranks)
        self.local_results = LocalResults()
        self.global_timeseries = TimeSeriesLocal()

        # Global results and fields (rank 0 only)
        if self.rank == 0:
            self.global_results = GlobalResults()
            self.global_fields = GlobalFields(N=self.config.N)
            self.global_timeseries = TimeSeriesGlobal()

        # Kernel selection
        self._step = jacobi_step_numba if self.config.use_numba else jacobi_step_numpy

        # Setup strategies
        if decomposition is None:
            # Sequential mode: entire domain on rank 0
            self.decomposition = NoDecomposition()
            self.communicator = NumpyCommunicator()
            kernel_name = "Numba" if self.config.use_numba else "NumPy"
            self.config.method = f"Sequential ({kernel_name})"
        else:
            # Distributed mode
            self.decomposition = _create_strategy(
                decomposition, DECOMPOSITION_REGISTRY, "decomposition"
            )

            # Default communicator to "numpy" if not specified
            if communicator is None:
                communicator = "numpy"

            self.communicator = _create_strategy(
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

    # ========================================================================
    # Main solve interface
    # ========================================================================

    def solve(self):
        """Main solve routine that wraps the Jacobi iteration."""
        # Start wall timings
        time_start = MPI.Wtime() if self.rank == 0 else None

        # Run Jacobi iteration
        self._method_solve()

        # Post-processing
        self._post_solve(time_start)

    def _method_solve(self):
        """Solve using Jacobi iteration.

        Orchestrates the high-level solve process: iteration loop,
        solution gathering, and results collection.
        """
        # NoDecomposition returns None for non-root ranks - they should exit
        if self.u1_local is None:
            return

        # Run main iteration loop
        u_final = self._run_iterations()

        # Gather solution to rank 0
        self._gather_solution(u_final)

        # Build per-rank results
        self._build_local_results()

    def _run_iterations(self):
        """Execute main Jacobi iteration loop.

        Returns
        -------
        np.ndarray
            Final solution array (local domain)
        """
        # Get problem parameters
        N = self.config.N
        h = 2.0 / (N - 1)

        # Initialize buffer pointers for ping-pong
        uold_local = self.u1_local
        u_local = self.u2_local

        # Main iteration loop
        for i in range(self.config.max_iter):
            # Perform one Jacobi iteration
            global_residual = self._perform_iteration(uold_local, u_local, h)

            # Check convergence
            if global_residual < self.config.tolerance:
                self._record_convergence(i + 1, converged=True)
                return u_local

            # Swap buffers for next iteration
            uold_local, u_local = u_local, uold_local
        else:
            # Max iterations reached without convergence
            self._record_convergence(self.config.max_iter, converged=False)
            return u_local

    def _perform_iteration(self, uold_local, u_local, h):
        """Perform a single Jacobi iteration.

        Parameters
        ----------
        uold_local : np.ndarray
            Previous solution (local domain with ghosts)
        u_local : np.ndarray
            New solution array (local domain with ghosts)
        h : float
            Grid spacing

        Returns
        -------
        float
            Global residual for this iteration
        """
        # Exchange ghost zones
        _time_operation(
            lambda: self.communicator.exchange_ghosts(uold_local, self.decomposition, self.rank, self.comm),
            self.global_timeseries.halo_exchange_times
        )

        # Jacobi step on local domain
        local_residual = _time_operation(
            lambda: self._step(uold_local, u_local, self.f_local, h, self.config.omega),
            self.global_timeseries.compute_times
        )

        # Compute global residual
        global_residual = _time_operation(
            lambda: np.sqrt(self.comm.allreduce(local_residual**2, op=MPI.SUM)),
            self.global_timeseries.mpi_comm_times
        )

        # Store residual history on rank 0
        if self.rank == 0:
            self.global_timeseries.residual_history.append(float(global_residual))

        return global_residual

    def _gather_solution(self, u_local):
        """Gather local solutions to global array on rank 0.

        Parameters
        ----------
        u_local : np.ndarray
            Local solution (with ghost zones)
        """
        N = self.config.N

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

    def _post_solve(self, start_time):
        """Aggregate timing results after solve.

        Parameters
        ----------
        start_time : float or None
            Start time from MPI.Wtime() on rank 0, None on other ranks
        """
        # Calculate wall time and aggregate timings (rank 0 only)
        if self.rank == 0:
            wall_time = MPI.Wtime() - start_time

            # Aggregate local timings to global results
            self.global_results.wall_time = wall_time
            self.global_results.compute_time = sum(self.global_timeseries.compute_times)
            self.global_results.mpi_comm_time = sum(self.global_timeseries.mpi_comm_times)
            self.global_results.halo_exchange_time = sum(self.global_timeseries.halo_exchange_times)

    # ========================================================================
    # Helper methods
    # ========================================================================

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

    # ========================================================================
    # Public utility methods
    # ========================================================================

    def warmup(self, N=10):
        """Warmup the solver (trigger JIT compilation for Numba).

        Parameters
        ----------
        N : int, optional
            Small grid size for warmup (default: 10)
        """
        h = 2.0 / (N - 1)
        u1 = np.zeros((N, N, N))
        u2 = np.zeros((N, N, N))
        f = np.random.randn(N, N, N)

        for _ in range(5):
            self._step(u1, u2, f, h, self.config.omega)
            u1, u2 = u2, u1

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

    def _dataclass_to_df(self, obj):
        """Convert any dataclass instance to single-row DataFrame.

        Parameters
        ----------
        obj : dataclass instance
            Any dataclass object to convert

        Returns
        -------
        pd.DataFrame or None
            Single-row DataFrame on rank 0, None on other ranks
        """
        if self.rank != 0:
            return None

        from dataclasses import asdict
        import pandas as pd

        return pd.DataFrame([asdict(obj)])

    def save_results(self, path, include_timeseries=False):
        """Save results to parquet file.

        Parameters
        ----------
        path : str or Path
            Output file path (should end in .parquet)
        include_timeseries : bool, optional
            Include detailed timeseries data (default: False)

        Notes
        -----
        Only rank 0 performs the save operation.
        """
        if self.rank != 0:
            return

        import pandas as pd

        # Combine config and results into single row
        df = pd.concat([
            self._dataclass_to_df(self.config),
            self._dataclass_to_df(self.global_results)
        ], axis=1)

        if include_timeseries:
            df_timeseries = self._dataclass_to_df(self.global_timeseries)
            df = pd.concat([df, df_timeseries], axis=1)

        df.to_parquet(path, index=False)

    def log_to_mlflow(self, experiment_name):
        """Log results to MLflow.

        Parameters
        ----------
        experiment_name : str
            Name of the MLflow experiment

        Notes
        -----
        Only rank 0 performs the logging operation.
        Logs config as params and results as metrics.
        """
        if self.rank != 0:
            return

        import mlflow
        from dataclasses import asdict

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run():
            # Log configuration as parameters
            mlflow.log_params(asdict(self.config))

            # Log results as metrics
            mlflow.log_metrics(asdict(self.global_results))

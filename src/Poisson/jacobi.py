"""Unified Jacobi solver for Poisson equation.

This solver handles both sequential and distributed (MPI) execution with a
single interface. Sequential execution is the default, and distributed execution
is enabled by providing decomposition and communicator strategies.
"""

import numpy as np
from dataclasses import asdict
from mpi4py import MPI
from numba import get_num_threads
import mlflow
import h5py

from .kernels import NumPyKernel, NumbaKernel
from .datastructures import (
    GlobalParams,
    GlobalMetrics,
    LocalSeries,
)
from .communicators import NumpyCommunicator
from .decomposition import NoDecomposition


# ============================================================================
# Module-level utilities
# ============================================================================


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
    execution. For single-rank execution, sequential mode is automatic. For
    multi-rank execution, decomposition and communicator must be provided.

    Parameters
    ----------
    decomposition : DecompositionStrategy, optional
        Domain decomposition strategy object (e.g., SlicedDecomposition(),
        CubicDecomposition()). Required for multi-rank execution.
    communicator : CommunicatorStrategy, optional
        Ghost exchange communicator object (e.g., NumpyCommunicator(),
        CustomMPICommunicator()). If None, defaults to NumpyCommunicator().
    **kwargs
        Solver configuration: N, omega, use_numba, max_iter, tolerance, etc.

    Examples
    --------
    Sequential execution (single rank):
    >>> solver = JacobiPoisson(N=100)
    >>> solver.solve()

    Distributed execution (run with mpiexec -n 4):
    >>> from Poisson import SlicedDecomposition, NumpyCommunicator
    >>> solver = JacobiPoisson(
    ...     N=100,
    ...     decomposition=SlicedDecomposition(),
    ...     communicator=NumpyCommunicator()
    ... )
    >>> solver.solve()
    """

    def __init__(self, decomposition=None, communicator=None, **kwargs):
        # MPI setup
        comm = MPI.COMM_WORLD
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        # Configuration (all ranks)
        self.config = GlobalParams(**kwargs)
        self.config.numba_threads = get_num_threads()
        self.config.mpi_size = self.size

        # Timeseries (all ranks)
        self.timeseries = LocalSeries()

        # Results (rank 0 only)
        if self.rank == 0:
            self.results = GlobalMetrics()

        # Kernel selection
        N = self.config.N
        if self.config.use_numba:
            self.kernel = NumbaKernel(N=N, omega=self.config.omega, numba_threads=self.config.numba_threads)
        else:
            self.kernel = NumPyKernel(N=N, omega=self.config.omega)
        self._step = self.kernel.step

        # Setup strategies based on MPI size
        if self.size == 1:
            # Single rank: use sequential execution but preserve strategy names for metadata
            if decomposition is not None:
                # Store actual strategy names provided by user (for validation/testing)
                self.config.decomposition = decomposition.__class__.__name__.lower().replace('decomposition', '')
                self.config.communicator = (
                    communicator.__class__.__name__.lower().replace('communicator', '').replace('mpi', '')
                    if communicator is not None else 'numpy'
                )
            else:
                # True sequential mode - no strategies provided
                self.config.decomposition = "none"
                self.config.communicator = "none"

            # Always use NoDecomposition internally for single rank
            self.decomposition = NoDecomposition()
            self.communicator = communicator if communicator is not None else NumpyCommunicator()
        else:
            # Multi-rank: require decomposition strategy
            if decomposition is None:
                raise ValueError(
                    "Decomposition strategy required for multi-rank execution. "
                    "Example: decomposition=SlicedDecomposition()"
                )

            self.decomposition = decomposition
            self.communicator = communicator if communicator is not None else NumpyCommunicator()

            # Store strategy names in config
            self.config.decomposition = decomposition.__class__.__name__.lower().replace('decomposition', '')
            self.config.communicator = self.communicator.__class__.__name__.lower().replace('communicator', '').replace('mpi', '')

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

    def _run_iterations(self):
        """Execute main Jacobi iteration loop.

        Returns
        -------
        np.ndarray
            Final solution array (local domain)
        """
        # Initialize buffer pointers for ping-pong
        uold_local = self.u1_local
        u_local = self.u2_local

        # Main iteration loop
        for i in range(self.config.max_iter):
            # Perform one Jacobi iteration
            global_residual = self._perform_iteration(uold_local, u_local)

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

    def _perform_iteration(self, uold_local, u_local):
        """Perform a single Jacobi iteration.

        Parameters
        ----------
        uold_local : np.ndarray
            Previous solution (local domain with ghosts)
        u_local : np.ndarray
            New solution array (local domain with ghosts)

        Returns
        -------
        float
            Global residual for this iteration
        """
        # Exchange ghost zones
        _time_operation(
            lambda: self.communicator.exchange_ghosts(uold_local, self.decomposition, self.rank, self.comm),
            self.timeseries.halo_exchange_times
        )

        # Jacobi step on local domain - returns sum of squared differences
        local_diff_sum = _time_operation(
            lambda: self._step(uold_local, u_local, self.f_local),
            self.timeseries.compute_times
        )

        # Apply Dirichlet BCs at physical boundaries (for cubic decomposition)
        self.decomposition.apply_boundary_conditions(u_local, self.rank)

        # Compute global residual with proper normalization
        # Sum squared differences across all ranks, then normalize by global interior points
        N = self.config.N
        global_n_interior = (N - 2) ** 3  # Total interior points in global domain
        global_residual = _time_operation(
            lambda: np.sqrt(self.comm.allreduce(local_diff_sum, op=MPI.SUM)) / global_n_interior,
            self.timeseries.mpi_comm_times
        )

        # Store residual history on rank 0
        if self.rank == 0:
            self.timeseries.residual_history.append(float(global_residual))

        return global_residual

    def _gather_solution(self, u_local):
        """Gather local solutions to global array on rank 0.

        Parameters
        ----------
        u_local : np.ndarray
            Local solution (with ghost zones)

        Notes
        -----
        Stores the gathered global solution in self.u_global (rank 0 only).
        This is used for computing error in summary() and writing to HDF5.
        """
        N = self.config.N

        if self.rank == 0:
            # Extract interior points using decomposition strategy
            local_interior = self.decomposition.extract_interior(u_local)
            all_interiors = self.comm.gather(local_interior, root=0)

            self.u_global = np.zeros((N, N, N))
            for rank_id, interior_data in enumerate(all_interiors):
                placement = self.decomposition.get_interior_placement(rank_id, N, self.comm)
                self.u_global[placement] = interior_data
        else:
            # Non-root ranks just participate in gather
            local_interior = self.decomposition.extract_interior(u_local)
            self.comm.gather(local_interior, root=0)

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
            self.results.iterations = iterations
            self.results.converged = converged

    # ========================================================================
    # Public utility methods
    # ========================================================================

    def warmup(self, warmup_size=10):
        """Warmup the solver (trigger JIT compilation for Numba).

        Parameters
        ----------
        warmup_size : int, optional
            Small grid size for warmup (default: 10)
        """
        self.kernel.warmup(warmup_size=warmup_size)

    def save_hdf5(self, path):
        """Save complete simulation state to HDF5.

        Parameters
        ----------
        path : str or Path
            Output HDF5 file path

        Notes
        -----
        Gathers distributed solution to rank 0, which writes the file.
        Note: Parallel HDF5 I/O is available but has known issues on macOS.

        File structure:
        - /config: Runtime configuration
        - /fields/u: Global solution array
        - /results: Convergence information
        - /timings/rank_0/: Rank 0 timing data
        """

        N = self.config.N

        # Gather solution to rank 0
        if not hasattr(self, 'u_global'):
            self._gather_solution(self.u2_local if hasattr(self, 'u2_local') else self.u1_local)

        # Only rank 0 writes
        if self.rank != 0:
            return

        with h5py.File(path, 'w') as f:
            # Write config
            config_grp = f.create_group('config')
            for key, value in asdict(self.config).items():
                config_grp.attrs[key] = value

            # Write solution array
            fields_grp = f.create_group('fields')
            u_dset = fields_grp.create_dataset('u', (N, N, N), dtype='f8')
            u_dset[:] = self.u_global

            # Write results
            results_grp = f.create_group('results')
            for key, value in asdict(self.results).items():
                results_grp.attrs[key] = value

            # Write timing data (rank 0 only)
            rank_grp = f.create_group(f'timings/rank_{self.rank}')
            rank_grp.create_dataset('compute_times', data=self.timeseries.compute_times)
            rank_grp.create_dataset('mpi_comm_times', data=self.timeseries.mpi_comm_times)
            rank_grp.create_dataset('halo_exchange_times', data=self.timeseries.halo_exchange_times)
            rank_grp.create_dataset('residual_history', data=self.timeseries.residual_history)

    # ========================================================================
    # MLflow logging
    # ========================================================================

    def mlflow_start(self, experiment_name):
        """Start MLflow logging for this solver run.

        Parameters
        ----------
        experiment_name : str
            Name of the MLflow experiment

        Notes
        -----
        Only rank 0 performs MLflow operations. Logs all configuration
        parameters at the start of the run.

        Requires MLflow 2.15.0 or above for storing artifacts in a volume.
        """
        if self.rank != 0:
            return

        mlflow.login()

        # Databricks requires absolute workspace paths for experiment names
        if not experiment_name.startswith("/"):
            # Use shared workspace location (no user directory needed)
            experiment_name = f"/Shared/{experiment_name}"

        if mlflow.get_experiment_by_name(experiment_name) is None:
            mlflow.create_experiment(name=experiment_name)

        mlflow.set_experiment(experiment_name)

        mlflow.start_run()
        mlflow.log_params(asdict(self.config))

    def mlflow_end(self):
        """End MLflow logging and record final metrics.

        Notes
        -----
        Only rank 0 performs MLflow operations. Logs:
        - Global metrics (iterations, converged, final_error)
        - Residual history as step-by-step metrics
        - Per-rank timing data as a table
        """
        if self.rank != 0:
            return


        # Log global results
        #results_dict = asdict(self.results)
        #mlflow.log_metrics({k: v for k, v in results_dict.items() if v is not None})

        # Log residual history as step-by-step metrics
            #for step, residual in enumerate(self.timeseries.residual_history):
        #mlflow.log_metric("residual", residual, step=step)

        # Log timing summary
        mlflow.log_metrics({
            "total_compute_time": sum(self.timeseries.compute_times),
            "total_halo_time": sum(self.timeseries.halo_exchange_times),
            "total_mpi_comm_time": sum(self.timeseries.mpi_comm_times),
        })

        mlflow.end_run()

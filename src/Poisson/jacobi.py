"""Unified Jacobi solver for Poisson equation.

This solver handles both sequential and distributed (MPI) execution with a
single interface. Sequential execution is the default, and distributed execution
is enabled by providing decomposition and communicator strategies.
"""

import numpy as np
from mpi4py import MPI
from numba import get_num_threads

from .kernels import jacobi_step_numpy, jacobi_step_numba
from .datastructures import (
    Config,
    Results,
    Timeseries,
    sinusoidal_exact_solution,
)
from .strategies import (
    NoDecomposition,
    NumpyCommunicator,
)


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
        self.config = Config(**kwargs)
        self.config.num_threads = get_num_threads()
        self.config.mpi_size = self.size

        # Timeseries (all ranks)
        self.timeseries = Timeseries()

        # Results (rank 0 only)
        if self.rank == 0:
            self.results = Results()

        # Kernel selection
        self._step = jacobi_step_numba if self.config.use_numba else jacobi_step_numpy

        # Setup strategies based on MPI size
        if self.size == 1:
            # Single rank: always use sequential mode
            self.decomposition = NoDecomposition()
            self.communicator = NumpyCommunicator()
            self.config.decomposition = "none"
            self.config.communicator = "none"
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
            self.timeseries.halo_exchange_times
        )

        # Jacobi step on local domain
        local_residual = _time_operation(
            lambda: self._step(uold_local, u_local, self.f_local, h, self.config.omega),
            self.timeseries.compute_times
        )

        # Compute global residual
        global_residual = _time_operation(
            lambda: np.sqrt(self.comm.allreduce(local_residual**2, op=MPI.SUM)),
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

    def _post_solve(self, start_time):
        """Record final timing after solve.

        Parameters
        ----------
        start_time : float or None
            Start time from MPI.Wtime() on rank 0, None on other ranks

        Notes
        -----
        Timing aggregation is now handled by PostProcessor from HDF5 files.
        This method is kept minimal - convergence info is already in self.results.
        """
        # Nothing to do here - results are already stored in self.results
        # Timing data is in self.timeseries (per-rank)
        pass

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

        # Get the computed solution (gathered in _gather_solution)
        u_computed = self.u_global

        # Get exact solution
        if exact_solution is None:
            u_exact = sinusoidal_exact_solution(N)
        elif callable(exact_solution):
            u_exact = exact_solution(N)
        else:
            u_exact = exact_solution

        # Compute L2 error (interior points only)
        error_diff = u_computed[1:-1, 1:-1, 1:-1] - u_exact[1:-1, 1:-1, 1:-1]
        self.results.final_error = float(np.sqrt(h**3 * np.sum(error_diff**2)))

    def save_hdf5(self, path):
        """Save complete simulation state to HDF5 using parallel I/O.

        Parameters
        ----------
        path : str or Path
            Output HDF5 file path

        Notes
        -----
        This method uses parallel HDF5 I/O where each rank writes its data
        concurrently when running with multiple MPI ranks. For sequential
        execution (single rank), it uses standard HDF5 I/O.

        The file structure is:
        - /config: Runtime configuration (written by rank 0)
        - /fields/u: Global solution array (collective write by all ranks)
        - /results: Convergence information (written by rank 0)
        - /timings/rank_N/: Per-rank timing data (each rank writes its own)

        Requires h5py built with MPI support when using multiple ranks.
        If MPI support is not available, falls back to sequential write on rank 0.
        """
        import h5py
        from dataclasses import asdict

        N = self.config.N
        use_parallel_io = False

        # Determine whether to use parallel I/O
        if self.size > 1:
            try:
                # Try to use parallel MPI driver
                file_kwargs = {'driver': 'mpio', 'comm': self.comm}
                f = h5py.File(path, 'w', **file_kwargs)
                use_parallel_io = True
            except (ValueError, TypeError):
                # Fall back to sequential write if MPI not supported
                if self.rank == 0:
                    import warnings
                    warnings.warn(
                        "h5py was built without MPI support. Falling back to "
                        "sequential HDF5 writes (rank 0 only). For better "
                        "scalability, rebuild h5py with MPI support.",
                        UserWarning
                    )
                    # Gather solution to rank 0 if not already done
                    if not hasattr(self, 'u_global'):
                        self._gather_solution(self.u2_local if hasattr(self, 'u2_local') else self.u1_local)

                    f = h5py.File(path, 'w')
                else:
                    # Non-root ranks exit in fallback mode
                    return
        else:
            f = h5py.File(path, 'w')

        # Write data to HDF5 file
        with f:
            # Rank 0 writes config
            if self.rank == 0:
                config_grp = f.create_group('config')
                for key, value in asdict(self.config).items():
                    config_grp.attrs[key] = value

            # Write solution array
            fields_grp = f.create_group('fields')
            u_dset = fields_grp.create_dataset('u', (N, N, N), dtype='f8')

            if use_parallel_io:
                # Parallel I/O: each rank writes its interior slice
                local_interior = self.decomposition.extract_interior(
                    self.u2_local if hasattr(self, 'u2_local') else self.u1_local
                )
                placement = self.decomposition.get_interior_placement(self.rank, N, self.comm)
                u_dset[placement] = local_interior
            else:
                # Sequential write: rank 0 writes complete solution
                if self.rank == 0:
                    if hasattr(self, 'u_global'):
                        u_dset[:] = self.u_global
                    else:
                        # For sequential execution, write from local array
                        u_dset[:] = self.u2_local if hasattr(self, 'u2_local') else self.u1_local

            # Rank 0 writes results
            if self.rank == 0:
                results_grp = f.create_group('results')
                for key, value in asdict(self.results).items():
                    results_grp.attrs[key] = value

            # Write timing data
            if use_parallel_io:
                # Parallel I/O: each rank writes its own timing data
                rank_grp = f.create_group(f'timings/rank_{self.rank}')
                rank_grp.create_dataset('compute_times', data=self.timeseries.compute_times, compression='gzip')
                rank_grp.create_dataset('mpi_comm_times', data=self.timeseries.mpi_comm_times, compression='gzip')
                rank_grp.create_dataset('halo_exchange_times', data=self.timeseries.halo_exchange_times, compression='gzip')

                # Rank 0 additionally writes residual history
                if self.rank == 0:
                    rank_grp.create_dataset('residual_history', data=self.timeseries.residual_history, compression='gzip')
            else:
                # Sequential write: only rank 0 writes timing data
                if self.rank == 0:
                    rank_grp = f.create_group(f'timings/rank_0')
                    rank_grp.create_dataset('compute_times', data=self.timeseries.compute_times, compression='gzip')
                    rank_grp.create_dataset('mpi_comm_times', data=self.timeseries.mpi_comm_times, compression='gzip')
                    rank_grp.create_dataset('halo_exchange_times', data=self.timeseries.halo_exchange_times, compression='gzip')
                    rank_grp.create_dataset('residual_history', data=self.timeseries.residual_history, compression='gzip')

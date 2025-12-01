"""Unified Jacobi solver for Poisson equation.

This solver handles both sequential and distributed (MPI) execution with a
single interface. Sequential execution is the default, and distributed execution
is enabled by providing decomposition and communicator strategies.
"""

import time

import numpy as np
from dataclasses import asdict
from mpi4py import MPI
from numba import get_num_threads


from .kernels import NumPyKernel, NumbaKernel
from .datastructures import GlobalParams, GlobalMetrics, LocalSeries
from .mpi.communicators import NumpyHaloExchange
from .mpi.decomposition import NoDecomposition


def _get_strategy_name(obj):
    """Extract clean strategy name from object class."""
    if obj is None:
        return "numpy"
    name = obj.__class__.__name__.lower()
    return (
        name.replace("decomposition", "")
        .replace("communicator", "")
        .replace("haloexchange", "")
        .replace("mpi", "")
    )


class JacobiPoisson:
    """Unified Jacobi solver for sequential and distributed execution.

    Parameters
    ----------
    decomposition : DecompositionStrategy, optional
        Domain decomposition strategy. Required for multi-rank execution.
    communicator : CommunicatorStrategy, optional
        Halo exchange communicator. Defaults to NumpyHaloExchange().
    **kwargs
        Solver configuration: N, omega, use_numba, max_iter, tolerance, etc.
    """

    def __init__(self, decomposition=None, communicator=None, **kwargs):
        # MPI setup
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Configuration
        self.config = GlobalParams(**kwargs)
        self.config.numba_threads = get_num_threads()
        self.config.mpi_size = self.size

        # Data structures
        self.timeseries = LocalSeries()
        if self.rank == 0:
            self.results = GlobalMetrics()

        # Kernel selection
        KernelClass = NumbaKernel if self.config.use_numba else NumPyKernel
        self.kernel = KernelClass(
            N=self.config.N,
            omega=self.config.omega,
            numba_threads=self.config.numba_threads if self.config.use_numba else None,
        )

        # Strategy setup
        self._setup_strategies(decomposition, communicator)

        # Initialize arrays
        self.u1_local, self.u2_local, self.f_local = (
            self.decomposition.initialize_local_arrays_distributed(
                self.config.N, self.rank, self.comm
            )
        )

    def _setup_strategies(self, decomposition, communicator):
        """Configure decomposition and communicator strategies."""
        if self.size == 1:
            # Single rank: store names but use NoDecomposition internally
            self.config.decomposition = (
                getattr(decomposition, "strategy", "none") if decomposition else "none"
            )
            self.config.communicator = (
                _get_strategy_name(communicator) if communicator else "none"
            )
            self.decomposition = NoDecomposition()
            self.communicator = communicator or NumpyHaloExchange()
        else:
            if decomposition is None:
                raise ValueError(
                    "Decomposition strategy required for multi-rank execution"
                )
            self.decomposition = decomposition
            self.communicator = communicator or NumpyHaloExchange()
            self.config.decomposition = getattr(decomposition, "strategy", "unknown")
            self.config.communicator = _get_strategy_name(self.communicator)

    # ========================================================================
    # Solve interface
    # ========================================================================

    def solve(self):
        """Run Jacobi iteration to solve the Poisson equation."""
        if self.u1_local is None:  # Non-root in sequential mode
            return

        u_final = self._iterate()

        # Aggregate timing data on rank 0
        if self.rank == 0:
            self.results.total_compute_time = sum(self.timeseries.compute_times)
            self.results.total_halo_time = sum(self.timeseries.halo_exchange_times)

        self._gather_solution(u_final)

    def _iterate(self):
        """Execute Jacobi iteration loop with non-blocking residual reduction.

        Uses MPI_Iallreduce to overlap the global residual computation with
        the next iteration's halo exchange, hiding communication latency.
        """
        uold, u = self.u1_local, self.u2_local
        n_interior = (self.config.N - 2) ** 3

        # Buffers for non-blocking allreduce
        local_res_buf = np.zeros(1)
        global_res_buf = np.zeros(1)
        pending_request = None

        for i in range(self.config.max_iter):
            # === Halo exchange (overlaps with previous iteration's Iallreduce) ===
            t0 = MPI.Wtime()
            self.communicator.exchange_halos(uold, self.decomposition, self.rank, self.comm)
            halo_time = MPI.Wtime() - t0

            # === Wait for previous iteration's residual reduction ===
            if pending_request is not None:
                pending_request.Wait()

                # Check convergence from previous iteration
                global_residual = np.sqrt(global_res_buf[0]) / n_interior
                if self.rank == 0:
                    self.timeseries.residual_history.append(float(global_residual))

                if global_residual < self.config.tolerance:
                    self.timeseries.halo_exchange_times.append(halo_time)
                    self._record_convergence(i, converged=True)
                    return uold  # Previous u is the converged solution

            self.timeseries.halo_exchange_times.append(halo_time)

            # === Compute Jacobi update ===
            t0 = MPI.Wtime()
            self.kernel.step(uold, u, self.f_local)
            self.decomposition.apply_boundary_conditions(u, self.rank)

            # Compute local residual
            diff = u[1:-1, 1:-1, 1:-1] - uold[1:-1, 1:-1, 1:-1]
            local_res_buf[0] = np.sum(diff**2)
            self.timeseries.compute_times.append(MPI.Wtime() - t0)

            # === Start non-blocking global reduction (overlaps with next halo exchange) ===
            pending_request = self.comm.Iallreduce(local_res_buf, global_res_buf, op=MPI.SUM)

            # Swap buffers for next iteration
            uold, u = u, uold

        # Final wait for last iteration's reduction
        if pending_request is not None:
            pending_request.Wait()
            global_residual = np.sqrt(global_res_buf[0]) / n_interior
            if self.rank == 0:
                self.timeseries.residual_history.append(float(global_residual))

            if global_residual < self.config.tolerance:
                self._record_convergence(self.config.max_iter, converged=True)
                return uold

        self._record_convergence(self.config.max_iter, converged=False)
        return uold

    def _gather_solution(self, u_local):
        """Gather local solutions to rank 0."""
        local_interior = self.decomposition.extract_interior(u_local)

        if self.rank == 0:
            all_interiors = self.comm.gather(local_interior, root=0)
            self.u_global = np.zeros((self.config.N,) * 3)
            for rank_id, data in enumerate(all_interiors):
                placement = self.decomposition.get_interior_placement(
                    rank_id, self.config.N, self.comm
                )
                self.u_global[placement] = data
        else:
            self.comm.gather(local_interior, root=0)

    def _record_convergence(self, iterations, converged):
        """Record convergence on rank 0."""
        if self.rank == 0:
            self.results.iterations = iterations
            self.results.converged = converged

    # ========================================================================
    # Validation
    # ========================================================================

    def compute_l2_error(self):
        """Compute L2 error against analytical solution (parallel).

        Each rank computes its local contribution, then MPI reduces.
        Result stored in self.results.final_error on rank 0.

        Returns
        -------
        float or None
            L2 error on rank 0, None on other ranks
        """
        N = self.config.N
        h = 2.0 / (N - 1)

        # Get final solution
        u_local = self.u2_local if self.u2_local is not None else self.u1_local

        # Compute exact solution for local domain
        info = self.decomposition.get_rank_info(self.rank)
        gs = info.global_start
        local_shape = info.local_shape

        if (
            hasattr(self.decomposition, "strategy")
            and self.decomposition.strategy == "cubic"
        ):
            # Cubic: all dims decomposed
            nz, ny, nx = local_shape
            z_idx = np.arange(gs[0], gs[0] + nz)
            y_idx = np.arange(gs[1], gs[1] + ny)
            x_idx = np.arange(gs[2], gs[2] + nx)

            zs = -1.0 + z_idx * h
            ys = -1.0 + y_idx * h
            xs = -1.0 + x_idx * h

            Z = zs.reshape((nz, 1, 1))
            Y = ys.reshape((1, ny, 1))
            X = xs.reshape((1, 1, nx))

            u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)
            u_numerical = u_local[1:-1, 1:-1, 1:-1]
        else:
            # Sliced: only z decomposed, interior is [1:-1, 1:-1, 1:-1] in global
            nz = local_shape[0]
            z_idx = np.arange(gs[0], gs[0] + nz)

            zs = -1.0 + z_idx * h
            ys = np.linspace(-1, 1, N)[1:-1]  # Interior only
            xs = np.linspace(-1, 1, N)[1:-1]

            Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
            u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)
            u_numerical = u_local[1:-1, 1:-1, 1:-1]

        # Compute local squared error
        local_sq_error = np.sum((u_numerical - u_exact) ** 2)

        # Global reduction
        global_sq_error = self.comm.allreduce(local_sq_error, op=MPI.SUM)
        l2_error = float(np.sqrt(h**3 * global_sq_error))

        if self.rank == 0:
            self.results.final_error = l2_error
            return l2_error
        return None

    # ========================================================================
    # Utilities
    # ========================================================================

    def warmup(self, warmup_size=10):
        """Warmup kernel (trigger Numba JIT)."""
        self.kernel.warmup(warmup_size=warmup_size)

    def save_hdf5(self, path):
        """Save config, results, and timeseries to HDF5 (rank 0 only)."""
        if self.rank != 0:
            return

        import pandas as pd
        import warnings

        row = {**asdict(self.config), **asdict(self.results)}
        df_results = pd.DataFrame([row])

        # Convert string columns to avoid PyTables pickle warning
        for col in df_results.select_dtypes(include=['object']).columns:
            df_results[col] = df_results[col].astype(str)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            df_results.to_hdf(path, key="results", mode="w", format="table")

            # Save timeseries data for per-iteration analysis
            ts_dict = asdict(self.timeseries)
            # Filter out None/empty lists and ensure equal lengths
            ts_data = {k: v for k, v in ts_dict.items() if v}
            if ts_data:
                max_len = max(len(v) for v in ts_data.values())
                # Pad shorter lists with NaN
                for k, v in ts_data.items():
                    if len(v) < max_len:
                        ts_data[k] = v + [float('nan')] * (max_len - len(v))
                pd.DataFrame(ts_data).to_hdf(path, key="timeseries", mode="a", format="table")


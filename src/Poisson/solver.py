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
            self.results.total_mpi_comm_time = sum(self.timeseries.mpi_comm_times)

        self._gather_solution(u_final)

    def _iterate(self):
        """Execute Jacobi iteration loop."""
        uold, u = self.u1_local, self.u2_local

        for i in range(self.config.max_iter):
            residual = self._step(uold, u)

            if residual < self.config.tolerance:
                self._record_convergence(i + 1, converged=True)
                return u

            uold, u = u, uold

        self._record_convergence(self.config.max_iter, converged=False)
        return u

    def _step(self, uold, u):
        """Perform one Jacobi step with timing."""
        # Halo exchange
        t0 = MPI.Wtime()
        self.communicator.exchange_halos(uold, self.decomposition, self.rank, self.comm)
        self.timeseries.halo_exchange_times.append(MPI.Wtime() - t0)

        # Compute update
        t0 = MPI.Wtime()
        self.kernel.step(uold, u, self.f_local)

        # Boundary conditions (before residual computation)
        self.decomposition.apply_boundary_conditions(u, self.rank)

        # Compute residual after BCs (so boundary cells don't contribute)
        diff = u[1:-1, 1:-1, 1:-1] - uold[1:-1, 1:-1, 1:-1]
        local_diff_sum = np.sum(diff**2)
        self.timeseries.compute_times.append(MPI.Wtime() - t0)

        # Global residual
        t0 = MPI.Wtime()
        n_interior = (self.config.N - 2) ** 3
        global_residual = (
            np.sqrt(self.comm.allreduce(local_diff_sum, op=MPI.SUM)) / n_interior
        )
        self.timeseries.mpi_comm_times.append(MPI.Wtime() - t0)

        if self.rank == 0:
            self.timeseries.residual_history.append(float(global_residual))

        return global_residual

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
        """Save config and results to HDF5 (rank 0 only)."""
        if self.rank != 0:
            return

        import pandas as pd

        row = {**asdict(self.config), **asdict(self.results)}
        pd.DataFrame([row]).to_hdf(path, key="results", mode="w")


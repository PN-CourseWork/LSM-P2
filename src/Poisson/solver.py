"""Unified Jacobi solver for Poisson equation."""

import numpy as np
from mpi4py import MPI
from numba import get_num_threads

from .base import BasePoissonSolver
from .kernels import NumPyKernel, NumbaKernel
from .datastructures import GlobalParams
from .mpi.grid import DistributedGrid


class JacobiPoisson(BasePoissonSolver):
    """Unified Jacobi solver for sequential and distributed execution.

    Parameters
    ----------
    N : int
        Grid size (N x N x N)
    strategy : str
        Decomposition strategy: 'sliced', 'cubic', or 'auto'
    communicator : str
        Halo exchange method: 'numpy' (buffer-based) or 'custom' (MPI datatypes)
    **kwargs
        Solver configuration: omega, use_numba, max_iter, tolerance, etc.
    """

    def __init__(self, N: int, strategy: str = 'auto', communicator: str = 'numpy', **kwargs):
        super().__init__(N, **kwargs)

        # Configuration
        self.config = GlobalParams(N=N, **kwargs)
        self.config.numba_threads = get_num_threads()
        self.config.mpi_size = self.size
        self.config.decomposition = strategy if self.size > 1 else "none"
        self.config.communicator = communicator

        # Kernel selection
        KernelClass = NumbaKernel if self.config.use_numba else NumPyKernel
        self.kernel = KernelClass(
            N=self.config.N,
            omega=self.config.omega,
            numba_threads=self.config.numba_threads if self.config.use_numba else None,
        )

        # Create distributed grid (handles decomposition + halo exchange)
        self.grid = DistributedGrid(N, self.comm, strategy=strategy, halo_exchange=communicator)

        # Initialize arrays
        self.u1_local = self.grid.allocate()
        self.u2_local = self.grid.allocate()
        self.f_local = self.grid.allocate()

        # Pointer to the array holding the current best solution
        self.u_solution = self.u1_local

        # Fill source term and apply boundary conditions
        self.grid.fill_source_term(self.f_local)
        self.grid.apply_boundary_conditions(self.u1_local)
        self.grid.apply_boundary_conditions(self.u2_local)

    # ========================================================================
    # Solve interface
    # ========================================================================

    def _get_solution_array(self):
        return self.u_solution

    def solve(self):
        """Run Jacobi iteration to solve the Poisson equation."""
        if self.u1_local is None:  # Non-root in sequential mode
            return

        self.u_solution = self._iterate()

        # Aggregate timing data on rank 0
        if self.rank == 0:
            self.results.total_compute_time = sum(self.timeseries.compute_times)
            self.results.total_halo_time = sum(self.timeseries.halo_exchange_times)

        self._gather_solution(self.u_solution)

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
            self.grid.sync_halos(uold)
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
            self.grid.apply_boundary_conditions(u)

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
        # Extract interior (excluding halos)
        local_interior = u_local[1:-1, 1:-1, 1:-1].copy()

        if self.rank == 0:
            all_interiors = self.comm.gather(local_interior, root=0)
            self.u_global = np.zeros((self.config.N,) * 3)

            # Place each rank's interior in the global array
            for rank_id, data in enumerate(all_interiors):
                # Get geometry for this rank
                # Note: We need to recreate the grid to get other ranks' geometry
                # For now, just place based on known decomposition
                # This is a simplified approach - works for both sliced and cubic
                gs = self.grid.global_start if rank_id == self.rank else None
                gs = self.comm.bcast(self.grid.global_start, root=rank_id)
                ge = self.comm.bcast(self.grid.global_end, root=rank_id)
                self.u_global[gs[0]:ge[0], gs[1]:ge[1], gs[2]:ge[2]] = data
        else:
            self.comm.gather(local_interior, root=0)
            # Participate in broadcasts
            for rank_id in range(self.size):
                self.comm.bcast(self.grid.global_start if rank_id == self.rank else None, root=rank_id)
                self.comm.bcast(self.grid.global_end if rank_id == self.rank else None, root=rank_id)

    def _record_convergence(self, iterations, converged):
        """Record convergence on rank 0."""
        if self.rank == 0:
            self.results.iterations = iterations
            self.results.converged = converged




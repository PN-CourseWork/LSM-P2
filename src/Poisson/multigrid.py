"""Multigrid Solver for 3D Poisson Equation."""

import time
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from mpi4py import MPI

from .kernels import NumPyKernel, NumbaKernel
from .datastructures import GlobalParams, GlobalMetrics, LocalSeries
from .mpi.communicators import NumpyHaloExchange
from .mpi.decomposition import DomainDecomposition, NoDecomposition
from .multigrid_operators import restrict, prolong


@dataclass
class GridLevel:
    """Represents one level in the Multigrid hierarchy."""
    level_index: int
    N: int
    h: float
    
    # Arrays (Local to rank)
    u: np.ndarray  # Solution / Error correction
    f: np.ndarray  # RHS / Restricted residual
    r: np.ndarray  # Residual
    
    # Solver components
    kernel: object
    communicator: object
    decomposition: object
    
    # Buffers for smoothing
    u_temp: np.ndarray 


class MultigridPoisson:
    """Multigrid V-Cycle Solver."""

    def __init__(self, levels: int = 3, pre_smooth: int = 2, post_smooth: int = 2, decomposition_strategy: str = 'cubic', **kwargs):
        """
        Initialize Multigrid Solver.
        
        Parameters
        ----------
        levels : int
            Number of grid levels (depth of V-cycle).
        pre_smooth : int
            Number of pre-smoothing steps.
        post_smooth : int
            Number of post-smoothing steps.
        decomposition_strategy : str
            MPI decomposition strategy ('cubic' or 'sliced').
        kwargs : dict
            Configuration passed to GlobalParams (N, omega, etc.)
        """
        self.config = GlobalParams(**kwargs)
        self.levels = levels
        self.pre_smooth = pre_smooth
        self.post_smooth = post_smooth
        self.decomposition_strategy = decomposition_strategy
        
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        self.results = GlobalMetrics() # Add this line
        self.timeseries = LocalSeries() # Add this line for top-level timing if needed for compatibility
        
        # Initialize Grid Hierarchy
        self.grid_levels: List[GridLevel] = []
        self._setup_hierarchy()

    def _setup_hierarchy(self):
        """Allocate arrays and setup solvers for all levels."""
        N_current = self.config.N
        
        if self.size > 1:
            from .mpi.decomposition import DomainDecomposition
        else:
            base_decomposition_cls = NoDecomposition

        for l in range(self.levels):
            h = 2.0 / (N_current - 1)
            
            if self.size > 1:
                decomp = DomainDecomposition(N=N_current, size=self.size, strategy=self.decomposition_strategy) 
            else:
                decomp = NoDecomposition()
                
            u1, u2, f = decomp.initialize_local_arrays_distributed(N_current, self.rank, self.comm)
            
            r = np.zeros_like(u1)
            
            KernelClass = NumbaKernel if self.config.use_numba else NumPyKernel # Re-enable NumbaKernel selection
            kernel = KernelClass(
                N=N_current,
                omega=self.config.omega,
                numba_threads=None
            )
            
            communicator = NumpyHaloExchange()
            
            level_obj = GridLevel(
                level_index=l,
                N=N_current,
                h=h,
                u=u1,
                u_temp=u2,
                f=f,
                r=r,
                kernel=kernel,
                communicator=communicator,
                decomposition=decomp
            )
            self.grid_levels.append(level_obj)
            
            if (N_current - 1) % 2 != 0:
                raise ValueError(f"Grid size N={N_current} is not compatible with Multigrid (N-1 must be divisible by 2).")
            
            N_current = (N_current - 1) // 2 + 1

    def solve(self):
        """Execute V-Cycles until convergence."""
        t_start_solve = MPI.Wtime()
        
        fine_lvl = self.grid_levels[0]
        
        # Initial residual calculation
        if self.rank == 0:
            print("Computing initial residual...")
        
        self._compute_residual(fine_lvl)
        initial_residual = self._get_global_residual_norm(fine_lvl.r)
        
        if self.rank == 0:
            print(f"Initial Residual: {initial_residual}")

        residual = initial_residual
        for iter in range(self.config.max_iter):
            residual = self.v_cycle(0)
            
            if self.rank == 0:
                print(f"V-Cycle {iter}: Residual = {residual}")
                
            if residual < self.config.tolerance:
                if self.rank == 0:
                    print("Converged.")
                self.results.converged = True
                self.results.iterations = iter + 1
                break
        else:
            if self.rank == 0:
                print("Max iterations reached without convergence.")
            self.results.converged = False
            self.results.iterations = self.config.max_iter

        self.results.wall_time = MPI.Wtime() - t_start_solve
        
    def _get_global_residual_norm(self, r_array: np.ndarray) -> float:
        """Helper to compute global RMS norm of residual (matching JacobiPoisson)."""
        local_res_sq = np.sum(r_array[1:-1, 1:-1, 1:-1]**2)
        global_res_sq_sum = self.comm.allreduce(local_res_sq, op=MPI.SUM)
        
        N_fine_interior_points = (self.grid_levels[0].N - 2) ** 3
        # JacobiPoisson divides by N_interior_points (not 1.5 power).
        # This gives RMS error relative to number of points.
        norm = np.sqrt(global_res_sq_sum) / N_fine_interior_points
        return norm

    def v_cycle(self, level_idx: int) -> float:
        """Recursive V-Cycle Step."""
        lvl = self.grid_levels[level_idx]
        
        # 1. Pre-smoothing
        for _ in range(self.pre_smooth):
            self._smooth(lvl)
            
        # 2. Compute Residual r = f - Au
        self._compute_residual(lvl)
        
        # Return residual norm on finest level for convergence check
        if level_idx == 0:
            current_residual_norm = self._get_global_residual_norm(lvl.r)
        else:
            current_residual_norm = 0.0 # Not needed for coarse levels
        
        # Base Case: Coarsest Level
        if level_idx == self.levels - 1:
            for _ in range(self.pre_smooth + self.post_smooth + 5):
                self._smooth(lvl)
            return current_residual_norm

        # 3. Restriction: f_coarse = R(r_fine)
        next_lvl = self.grid_levels[level_idx + 1]
        
        # DEBUG: Disable halo exchange for restrict
        # lvl.communicator.exchange_halos(lvl.r, lvl.decomposition, self.rank, self.comm)
        
        restrict(lvl.r, next_lvl.f)
        
        next_lvl.u[:] = 0.0
        
        # 4. Recursion
        self.v_cycle(level_idx + 1)
        
        # 5. Prolongation: u_fine += P(u_coarse_correction)
        
        # DEBUG: Disable halo exchange for prolong
        # next_lvl.communicator.exchange_halos(next_lvl.u, next_lvl.decomposition, self.rank, self.comm)
        
        lvl.r[:] = 0.0
        prolong(next_lvl.u, lvl.r)
        
        lvl.u[1:-1, 1:-1, 1:-1] += lvl.r[1:-1, 1:-1, 1:-1]
        
        # 6. Post-smoothing
        for _ in range(self.post_smooth):
            self._smooth(lvl)
            
        return current_residual_norm

    def _smooth(self, lvl: GridLevel):
        """Perform one Jacobi smoothing step."""
        # DEBUG: Temporarily disable halo exchange
        # lvl.communicator.exchange_halos(lvl.u, lvl.decomposition, self.rank, self.comm)
        
        lvl.kernel.step(lvl.u, lvl.u_temp, lvl.f)
        lvl.decomposition.apply_boundary_conditions(lvl.u_temp, self.rank)
        lvl.u, lvl.u_temp = lvl.u_temp, lvl.u

    def _compute_residual(self, lvl: GridLevel):
        """Compute residual r = f - Au."""
        u = lvl.u
        f = lvl.f
        r = lvl.r
        h2 = lvl.h * lvl.h
        
        # DEBUG: Temporarily disable halo exchange
        # lvl.communicator.exchange_halos(u, lvl.decomposition, self.rank, self.comm)
        
        u_center = u[1:-1, 1:-1, 1:-1]
        u_neighbors = (
            u[0:-2, 1:-1, 1:-1] + u[2:, 1:-1, 1:-1] +
            u[1:-1, 0:-2, 1:-1] + u[1:-1, 2:, 1:-1] +
            u[1:-1, 1:-1, 0:-2] + u[1:-1, 1:-1, 2:]
        )
        
        laplacian_u = (u_neighbors - 6.0 * u_center) / h2
        r[1:-1, 1:-1, 1:-1] = f[1:-1, 1:-1, 1:-1] + laplacian_u
        
    # ========================================================================
    # Validation
    # ========================================================================

    def compute_l2_error(self):
        """Compute L2 error against analytical solution (parallel).

        Each rank computes its local contribution, then MPI reduces.
        Result stored in self.results.final_error on rank 0.
        """
        # This will be similar to JacobiPoisson's compute_l2_error.
        # It needs the final solution `lvl.u` from the finest level.
        
        N = self.config.N
        h = 2.0 / (N - 1)

        # Get final solution from finest level
        fine_lvl = self.grid_levels[0]
        u_local = fine_lvl.u

        # Compute exact solution for local domain
        info = fine_lvl.decomposition.get_rank_info(self.rank)
        gs = info.global_start
        local_shape = info.local_shape

        # Logic copied from JacobiPoisson
        if (
            hasattr(fine_lvl.decomposition, "strategy")
            and fine_lvl.decomposition.strategy == "cubic"
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
        else: # Sliced
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
    # Save results to HDF5 (copied from JacobiPoisson)
    # ========================================================================

    def save_hdf5(self, path):
        """Save config and results to HDF5 (rank 0 only)."""
        if self.rank != 0:
            return

        import pandas as pd
        from dataclasses import asdict

        # Ensure all metrics are present, even if not explicitly set
        row = {
            **asdict(self.config),
            **asdict(self.results)
        }
        pd.DataFrame([row]).to_hdf(path, key="results", mode="w")
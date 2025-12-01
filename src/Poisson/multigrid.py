"""Multigrid Solver for 3D Poisson Equation."""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from mpi4py import MPI

from .solver import JacobiPoisson
from .kernels import NumPyKernel, NumbaKernel
from .datastructures import GlobalParams, GlobalMetrics, LocalSeries
from .mpi.grid import DistributedGrid
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

    # Distributed grid (handles decomposition + halo exchange)
    grid: DistributedGrid

    # Solver kernel
    kernel: object

    # Buffers for smoothing
    u_temp: np.ndarray


class MultigridPoisson(JacobiPoisson):
    """Multigrid V-Cycle Solver extending JacobiPoisson.

    Uses the unified DistributedGrid class for both sliced and cubic
    decomposition, ensuring consistent interior decomposition for
    multigrid operators.
    """

    def __init__(
        self,
        levels: Optional[int] = None,
        min_coarse_size: int = 3,
        n_smooth: int = 5,
        fmg_post_cycles: int = 1,
        decomposition_strategy: str = 'sliced',
        communicator: str = 'numpy',
        **kwargs,
    ):
        """
        Initialize Multigrid Solver.

        Parameters
        ----------
        levels : int
            Number of grid levels (depth of V-cycle).
        n_smooth : int
            Number of smoothing steps (pre and post).
        fmg_post_cycles : int
            Post-FMG V-cycles (default 1). One V-cycle after FMG phase
            polishes the solution to discretization error.
        decomposition_strategy : str
            MPI decomposition strategy ('sliced' or 'cubic'). Both use
            consistent interior decomposition via DistributedGrid.
        communicator : str
            Halo exchange method: 'numpy' (buffer-based) or 'custom' (MPI datatypes)
        kwargs : dict
            Configuration passed to GlobalParams (N, omega, etc.)
        """
        # MPI setup (needed before _infer_levels)
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Store multigrid-specific parameters before hierarchy setup
        self.decomposition_strategy = decomposition_strategy
        self.communicator = communicator
        self.min_coarse_size = max(min_coarse_size, 3)
        self.n_smooth = n_smooth
        self.fmg_post_cycles = max(0, fmg_post_cycles)

        if self.size > 1 and self.decomposition_strategy not in ("sliced", "cubic"):
            raise ValueError("Multigrid supports 'sliced' or 'cubic' decomposition for MPI.")

        # Config setup (needed for _infer_levels)
        self.config = GlobalParams(**kwargs)
        self.config.mpi_size = self.size
        self.config.decomposition = self.decomposition_strategy if self.size > 1 else "none"
        self.config.communicator = communicator

        # Determine number of levels automatically if not provided
        self.levels = levels if levels is not None else self._infer_levels(self.config.N)

        # Timing accumulators for multigrid operations
        self._time_compute = 0.0
        self._time_halo = 0.0
        self._time_mpi = 0.0

        # Results and timeseries (on all ranks for simplicity)
        self.results = GlobalMetrics()
        self.timeseries = LocalSeries()

        # Initialize Grid Hierarchy
        self.grid_levels: List[GridLevel] = []
        self._setup_hierarchy()

        # Set up pointers so inherited methods work
        fine_lvl = self.grid_levels[0]
        self.u1_local = fine_lvl.u
        self.u2_local = fine_lvl.u_temp
        self.f_local = fine_lvl.f
        self.kernel = fine_lvl.kernel

    def _infer_levels(self, N: int) -> int:
        """
        Infer the number of grid levels by coarsening until the grid is too small.

        Returns the total number of levels including the finest and coarsest.
        """
        if N < 3:
            raise ValueError("Grid size must be at least 3.")

        # Compute minimum grid size based on decomposition strategy
        # Each rank needs at least min_local interior points per dimension
        min_local = 3  # Minimum 3x3x3 local interior for restriction/prolongation

        if self.size > 1:
            if self.decomposition_strategy == "cubic":
                # For cubic decomposition, P ranks split into ~P^(1/3) per dimension
                # Local size = N / P^(1/3), need local size >= min_local + 2 (for halos)
                ranks_per_dim = int(round(self.size ** (1/3)))
                min_N_for_ranks = max((min_local + 2) * ranks_per_dim, self.min_coarse_size)
            else:
                # For sliced, all ranks share one axis
                min_N_for_ranks = self.size + 2
        else:
            min_N_for_ranks = 3

        if N < min_N_for_ranks:
            raise ValueError(
                f"Grid N={N} too small for {self.size} ranks with {self.decomposition_strategy} decomposition."
            )

        levels = 1  # count the finest grid
        N_current = N

        while True:
            if (N_current - 1) % 2 != 0:
                raise ValueError(
                    f"Grid size N={N_current} is not compatible with Multigrid (N-1 must be divisible by 2)."
                )

            N_next = (N_current - 1) // 2 + 1

            # Stop if next level would be too small for the current MPI size
            if self.size > 1 and N_next < min_N_for_ranks:
                break

            if N_next < self.min_coarse_size:
                break

            levels += 1
            N_current = N_next

            if N_current == 3:
                break

        return levels

    def _setup_hierarchy(self):
        """Allocate arrays and setup solvers for all levels.

        Uses DistributedGrid which consistently decomposes interior points
        for both sliced and cubic strategies, ensuring multigrid operators
        work correctly.
        """
        # Compute all N values from finest to coarsest
        N_values = []
        N_current = self.config.N
        for _ in range(self.levels):
            N_values.append(N_current)
            if (N_current - 1) % 2 != 0:
                raise ValueError(
                    f"Grid size N={N_current} is not compatible with "
                    "Multigrid (N-1 must be divisible by 2)."
                )
            N_current = (N_current - 1) // 2 + 1

        for l, N in enumerate(N_values):
            h = 2.0 / (N - 1)

            # Create distributed grid (unified for sliced/cubic)
            grid = DistributedGrid(
                N, self.comm,
                strategy=self.decomposition_strategy,
                halo_exchange=self.communicator
            )

            # Allocate arrays using grid's allocator
            u1 = grid.allocate()
            u2 = grid.allocate()
            f = grid.allocate()
            r = grid.allocate()

            # Fill source term for this level
            grid.fill_source_term(f)

            # Apply boundary conditions
            grid.apply_boundary_conditions(u1)
            grid.apply_boundary_conditions(u2)

            # Create kernel for smoothing
            KernelClass = NumbaKernel if self.config.use_numba else NumPyKernel
            kernel = KernelClass(
                N=N,
                omega=self.config.omega,
                numba_threads=None
            )

            level_obj = GridLevel(
                level_index=l,
                N=N,
                h=h,
                u=u1,
                u_temp=u2,
                f=f,
                r=r,
                grid=grid,
                kernel=kernel,
            )
            self.grid_levels.append(level_obj)

    def solve(self):
        """Execute V-Cycles until convergence."""
        # Reset timers and timeseries
        self._time_compute = 0.0
        self._time_halo = 0.0
        self._time_mpi = 0.0
        self.timeseries.compute_times.clear()
        self.timeseries.halo_exchange_times.clear()
        self.timeseries.level_indices.clear()
        self.timeseries.residual_history.clear()

        t_start_solve = MPI.Wtime()
        
        fine_lvl = self.grid_levels[0]
        
        # Initial residual calculation
        if self.rank == 0:
            print("Computing initial residual...")
        
        self._compute_residual(fine_lvl)
        initial_residual = self._get_global_residual_norm(fine_lvl.r, fine_lvl)
        
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
        self.results.total_compute_time = self._time_compute
        self.results.total_halo_time = self._time_halo
        self.results.total_mpi_comm_time = self._time_mpi
        
    def _get_global_residual_norm(self, r_array: np.ndarray, lvl: "GridLevel" = None) -> float:
        """Compute global RMS norm of residual.

        Uses allreduce to gather sum of squared residuals across all ranks.
        With DistributedGrid, interior slice [1:-1, 1:-1, 1:-1] always contains
        only interior points (boundaries are in halos).
        """
        # Interior slice is consistent for all decomposition strategies
        local_res_sq = np.sum(r_array[1:-1, 1:-1, 1:-1] ** 2)
        local_interior_pts = float(np.prod(np.array(r_array.shape) - 2))

        t0 = MPI.Wtime()
        local_data = np.array([local_res_sq, local_interior_pts])
        global_data = np.empty(2)
        self.comm.Allreduce(local_data, global_data, op=MPI.SUM)
        self._time_mpi += MPI.Wtime() - t0

        global_res_sq_sum = global_data[0]
        global_interior_pts = global_data[1]
        norm = np.sqrt(global_res_sq_sum) / global_interior_pts
        return norm

    def v_cycle(self, level_idx: int) -> float:
        """Recursive V-Cycle Step."""
        lvl = self.grid_levels[level_idx]

        # 1. Pre-smoothing
        for _ in range(self.n_smooth):
            self._smooth(lvl)

        # 2. Compute Residual r = f - Au
        self._compute_residual(lvl)

        # Return residual norm on finest level for convergence check
        if level_idx == 0:
            current_residual_norm = self._get_global_residual_norm(lvl.r, lvl)
        else:
            current_residual_norm = 0.0

        # Base Case: Coarsest Level
        if level_idx == self.levels - 1:
            return self._coarse_solve(lvl, max_iters=20)

        # 3. Restriction: f_coarse = R(r_fine)
        next_lvl = self.grid_levels[level_idx + 1]

        # Sync residual halos before restriction
        t0 = MPI.Wtime()
        lvl.grid.sync_halos(lvl.r)
        self._time_halo += MPI.Wtime() - t0

        t0 = MPI.Wtime()
        restrict(lvl.r, next_lvl.f)
        self._time_compute += MPI.Wtime() - t0

        # Sync coarse RHS halos after restriction
        t0 = MPI.Wtime()
        next_lvl.grid.sync_halos(next_lvl.f)
        self._time_halo += MPI.Wtime() - t0

        next_lvl.u[:] = 0.0

        # 4. Recursion
        self.v_cycle(level_idx + 1)

        # 5. Prolongation: u_fine += P(u_coarse_correction)
        t0 = MPI.Wtime()
        next_lvl.grid.sync_halos(next_lvl.u)
        self._time_halo += MPI.Wtime() - t0

        lvl.r[:] = 0.0
        t0 = MPI.Wtime()
        prolong(next_lvl.u, lvl.r)
        self._time_compute += MPI.Wtime() - t0

        # Sync prolongated correction halos
        t0 = MPI.Wtime()
        lvl.grid.sync_halos(lvl.r)
        self._time_halo += MPI.Wtime() - t0

        lvl.u[1:-1, 1:-1, 1:-1] += lvl.r[1:-1, 1:-1, 1:-1]

        # Enforce Dirichlet BC at physical boundaries
        lvl.grid.apply_boundary_conditions(lvl.u)

        # 6. Post-smoothing
        for _ in range(self.n_smooth):
            self._smooth(lvl)

        return current_residual_norm

    def _coarse_solve(self, lvl: GridLevel, max_iters: int = 500) -> float:
        """Solve on the coarsest grid accurately.

        For FMG to work properly, the coarsest grid must be solved
        to high accuracy since it provides the initial guess for all
        finer levels. With MPI decomposition, the "coarse" grid may
        still be fairly large (e.g., N=17 with 4 ranks), so we need
        more iterations than the theoretical minimum.
        """
        for _ in range(max_iters):
            self._smooth(lvl)
        self._compute_residual(lvl)
        return self._get_global_residual_norm(lvl.r, lvl)

    def fmg_solve(self, cycles: int = 1, debug: bool = False):
        """Full Multigrid (FMG) cycle starting from the coarsest grid."""
        # Reset timers and timeseries
        self._time_compute = 0.0
        self._time_halo = 0.0
        self._time_mpi = 0.0
        self.timeseries.compute_times.clear()
        self.timeseries.halo_exchange_times.clear()
        self.timeseries.level_indices.clear()
        self.timeseries.residual_history.clear()

        # Clear solution arrays (keep f - each level has source at its resolution)
        for lvl in self.grid_levels:
            lvl.u.fill(0.0)
            lvl.u_temp.fill(0.0)
            lvl.r.fill(0.0)

        if debug and self.rank == 0:
            print(f"[FMG] levels={self.levels}, fmg_post_cycles={self.fmg_post_cycles}")

        t_start = MPI.Wtime()
        residual = None

        for c in range(cycles):
            if debug and self.rank == 0:
                print(f"[FMG] Cycle {c+1}/{cycles}: Coarse solve at level {self.levels-1}")

            # Solve on coarsest level
            coarse_res = self._coarse_solve(self.grid_levels[-1])
            residual = coarse_res

            # Ascend hierarchy: prolongate and refine with a V-cycle at each level
            for l in reversed(range(self.levels - 1)):
                if debug and self.rank == 0:
                    print(f"[FMG] Ascending: prolong {l+1}->{l}, smooth, V-cycle({l})")

                coarse = self.grid_levels[l + 1]
                fine = self.grid_levels[l]

                # Sync coarse halos before prolongation
                t0 = MPI.Wtime()
                coarse.grid.sync_halos(coarse.u)
                self._time_halo += MPI.Wtime() - t0

                # Prolong coarse solution as initial guess on fine
                fine.r.fill(0.0)
                t0 = MPI.Wtime()
                prolong(coarse.u, fine.r)
                self._time_compute += MPI.Wtime() - t0

                # Sync prolongated solution halos
                t0 = MPI.Wtime()
                fine.grid.sync_halos(fine.r)
                self._time_halo += MPI.Wtime() - t0

                fine.u.fill(0.0)
                fine.u[1:-1, 1:-1, 1:-1] = fine.r[1:-1, 1:-1, 1:-1]

                # Enforce Dirichlet BC at physical boundaries
                fine.grid.apply_boundary_conditions(fine.u)

                # Smooth the interpolated guess
                for _ in range(self.n_smooth):
                    self._smooth(fine)

                # Refine with one V-cycle from this level
                residual = self.v_cycle(l)

        # Final residual check on finest level
        fine_lvl = self.grid_levels[0]
        self._compute_residual(fine_lvl)
        residual = self._get_global_residual_norm(fine_lvl.r, fine_lvl)

        if debug and self.rank == 0:
            print(f"[FMG] After FMG phase: residual={residual:.2e}")

        # Optional finishing V-cycles on finest level
        post_iters = 0
        while residual >= self.config.tolerance and post_iters < min(self.fmg_post_cycles, self.config.max_iter):
            if debug and self.rank == 0:
                print(f"[FMG] Post V-cycle {post_iters+1}/{self.fmg_post_cycles}: residual={residual:.2e}")
            residual = self.v_cycle(0)
            post_iters += 1

        if debug and self.rank == 0:
            print(f"[FMG] Done: {post_iters} post V-cycles, final residual={residual:.2e}")

        # Finalize metrics
        self.results.wall_time = MPI.Wtime() - t_start
        self.results.total_compute_time = self._time_compute
        self.results.total_halo_time = self._time_halo
        self.results.total_mpi_comm_time = self._time_mpi
        self.results.converged = residual < self.config.tolerance
        self.results.iterations = cycles * self.levels + post_iters

        return residual

    def _smooth(self, lvl: GridLevel):
        """Perform one Jacobi smoothing step."""
        t0 = MPI.Wtime()
        lvl.grid.sync_halos(lvl.u)
        halo_time = MPI.Wtime() - t0
        self._time_halo += halo_time

        t0 = MPI.Wtime()
        lvl.kernel.step(lvl.u, lvl.u_temp, lvl.f)
        lvl.grid.apply_boundary_conditions(lvl.u_temp)
        compute_time = MPI.Wtime() - t0
        self._time_compute += compute_time
        lvl.u, lvl.u_temp = lvl.u_temp, lvl.u

        # Track per-operation timing
        self.timeseries.compute_times.append(compute_time)
        self.timeseries.halo_exchange_times.append(halo_time)
        self.timeseries.level_indices.append(lvl.level_index)

    def _compute_residual(self, lvl: GridLevel):
        """Compute residual r = f - Au."""
        u = lvl.u
        f = lvl.f
        r = lvl.r
        h2 = lvl.h * lvl.h

        t0 = MPI.Wtime()
        lvl.grid.sync_halos(u)
        self._time_halo += MPI.Wtime() - t0

        u_center = u[1:-1, 1:-1, 1:-1]
        u_neighbors = (
            u[0:-2, 1:-1, 1:-1] + u[2:, 1:-1, 1:-1] +
            u[1:-1, 0:-2, 1:-1] + u[1:-1, 2:, 1:-1] +
            u[1:-1, 1:-1, 0:-2] + u[1:-1, 1:-1, 2:]
        )

        t0 = MPI.Wtime()
        laplacian_u = (u_neighbors - 6.0 * u_center) / h2
        r[1:-1, 1:-1, 1:-1] = f[1:-1, 1:-1, 1:-1] + laplacian_u

        # Zero out residual at physical boundaries
        lvl.grid.apply_boundary_conditions(r)

        self._time_compute += MPI.Wtime() - t0
        
    # ========================================================================
    # Validation
    # ========================================================================

    def compute_l2_error(self):
        """Compute L2 error against analytical solution (parallel).

        Uses DistributedGrid's compute_l2_error method which handles
        both sliced and cubic decomposition consistently.
        """
        fine_lvl = self.grid_levels[0]
        l2_error = fine_lvl.grid.compute_l2_error(fine_lvl.u)

        self.results.final_error = l2_error
        return l2_error

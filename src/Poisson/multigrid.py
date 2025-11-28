"""Multigrid Solver for 3D Poisson Equation."""

import time
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from mpi4py import MPI

from .kernels import NumPyKernel, NumbaKernel
from .datastructures import GlobalParams, GlobalMetrics, LocalSeries
from .mpi.communicators import NumpyHaloExchange, CustomHaloExchange
from .mpi.decomposition import DomainDecomposition, NoDecomposition
from .multigrid_operators import restrict, prolong, restrict_halo, prolong_halo


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

    def __init__(
        self,
        levels: Optional[int] = None,
        min_coarse_size: int = 3,
        n_smooth: int = 3,
        fmg_post_cycles: int = 50,
        decomposition_strategy: str = 'sliced',
        communicator: Optional[object] = None,
        **kwargs,
    ):
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
        # MPI setup
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.decomposition_strategy = decomposition_strategy
        self.config.mpi_size = self.size

        self.min_coarse_size = max(min_coarse_size, 3)

        # Determine number of levels automatically if not provided
        self.levels = levels if levels is not None else self._infer_levels(self.config.N)
        self.n_smooth = n_smooth
        self.fmg_post_cycles = max(0, fmg_post_cycles)

        if self.size > 1 and self.decomposition_strategy not in ("sliced", "cubic"):
            raise ValueError("Multigrid supports 'sliced' or 'cubic' decomposition for MPI.")

        # Communicator selection (numpy or custom)
        if isinstance(communicator, str):
            if communicator == "custom":
                self.communicator = CustomHaloExchange()
            elif communicator in (None, "numpy"):
                self.communicator = NumpyHaloExchange()
            else:
                raise ValueError(f"Unknown communicator: {communicator}")
        elif communicator is None:
            self.communicator = NumpyHaloExchange()
        else:
            self.communicator = communicator

        # Record metadata for outputs
        self.config.decomposition = self.decomposition_strategy if self.size > 1 else "none"
        self.config.communicator = self.communicator.__class__.__name__.lower()

        # Timing accumulators
        self._time_compute = 0.0
        self._time_halo = 0.0
        self._time_mpi = 0.0

        self.results = GlobalMetrics() # Add this line
        self.timeseries = LocalSeries() # Add this line for top-level timing if needed for compatibility
        
        # Initialize Grid Hierarchy
        self.grid_levels: List[GridLevel] = []
        self._setup_hierarchy()

    def _infer_levels(self, N: int) -> int:
        """
        Infer the number of grid levels by coarsening until the grid is too small.

        Returns the total number of levels including the finest and coarsest.
        """
        if N < 3:
            raise ValueError("Grid size must be at least 3.")

        # Compute minimum grid size based on decomposition strategy
        if self.size > 1:
            if self.decomposition_strategy == "cubic":
                # For cubic decomposition, we need larger coarse grids because
                # the restriction/prolongation accumulates errors at small grids
                # when not all dimensions are decomposed equally (e.g., 2x2x1).
                # Empirically, coarsest N>=33 is stable for multigrid.
                min_N_for_ranks = max(33, self.min_coarse_size)
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
        """Allocate arrays and setup solvers for all levels."""
        if self.size > 1:
            from .mpi.decomposition import DomainDecomposition

        # First, compute all N values from finest to coarsest
        N_values = []
        N_current = self.config.N
        for _ in range(self.levels):
            N_values.append(N_current)
            if (N_current - 1) % 2 != 0:
                raise ValueError(f"Grid size N={N_current} is not compatible with Multigrid (N-1 must be divisible by 2).")
            N_current = (N_current - 1) // 2 + 1

        # For cubic decomposition, compute aligned decompositions to ensure
        # restriction/prolongation work correctly across all levels
        if self.size > 1 and self.decomposition_strategy == "cubic":
            decompositions = self._compute_aligned_cubic_decompositions(N_values)
        else:
            decompositions = None

        for l, N in enumerate(N_values):
            h = 2.0 / (N - 1)

            if self.size > 1:
                if decompositions is not None:
                    decomp = decompositions[l]
                else:
                    decomp = DomainDecomposition(N=N, size=self.size, strategy=self.decomposition_strategy)
            else:
                decomp = NoDecomposition()

            u1, u2, f = decomp.initialize_local_arrays_distributed(N, self.rank, self.comm)

            r = np.zeros_like(u1)

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
                kernel=kernel,
                communicator=self.communicator,
                decomposition=decomp
            )
            self.grid_levels.append(level_obj)

    def _compute_aligned_cubic_decompositions(self, N_values):
        """Compute cubic decompositions with aligned boundaries for multigrid.

        For multigrid to work correctly, decomposition boundaries at fine levels
        must map exactly to coarse level boundaries via 2:1 coarsening.
        This means fine boundaries must be at EVEN global indices.

        We compute boundaries at the coarsest level first, then derive finer
        level boundaries by multiplying by 2.
        """
        from .mpi.decomposition import DomainDecomposition
        from mpi4py import MPI

        # Get process grid dimensions
        dims = MPI.Compute_dims(self.size, 3)
        px, py, pz = dims

        # Compute boundaries at coarsest level
        N_coarsest = N_values[-1]

        def compute_aligned_splits(N, parts, coarsen_factor):
            """Compute split boundaries that are divisible by coarsen_factor."""
            base = N // parts
            rem = N % parts

            # Standard split
            counts = [base + (1 if i < rem else 0) for i in range(parts)]

            # Adjust to make boundaries align with coarsening
            # Boundaries are cumsum of counts
            boundaries = [0]
            for c in counts:
                boundaries.append(boundaries[-1] + c)

            # For multigrid alignment, boundaries should be divisible by coarsen_factor
            # This is automatically satisfied at coarsest level
            # For finer levels, we multiply by 2

            return counts, boundaries[:-1]  # starts

        # Compute coarsest level splits
        coarsest_splits = {}
        for axis, (count, name) in enumerate([(pz, 'z'), (py, 'y'), (px, 'x')]):
            if count > 1:
                base = N_coarsest // count
                rem = N_coarsest % count
                counts = [base + (1 if i < rem else 0) for i in range(count)]
                starts = [sum(counts[:i]) for i in range(count)]
                coarsest_splits[axis] = (counts, starts)
            else:
                coarsest_splits[axis] = ([N_coarsest], [0])

        # Build decompositions for all levels (from finest to coarsest)
        decompositions = []

        for level_idx, N in enumerate(N_values):
            # Compute the coarsening factor from this level to coarsest
            coarsen_factor = 2 ** (len(N_values) - 1 - level_idx)

            # Scale coarsest boundaries to this level
            level_splits = {}
            for axis in range(3):
                coarse_counts, coarse_starts = coarsest_splits[axis]
                # Scale starts by coarsening factor
                fine_starts = [s * coarsen_factor for s in coarse_starts]
                # Compute counts based on starts and N
                fine_counts = []
                for i, start in enumerate(fine_starts):
                    if i + 1 < len(fine_starts):
                        end = fine_starts[i + 1]
                    else:
                        end = N
                    fine_counts.append(end - start)
                level_splits[axis] = (fine_counts, fine_starts)

            # Create a DomainDecomposition with explicit splits
            decomp = DomainDecomposition(N=N, size=self.size, strategy='cubic')

            # Override the computed splits with our aligned splits
            decomp._split_info = {
                'nz': level_splits[0],
                'ny': level_splits[1],
                'nx': level_splits[2],
            }

            # Recompute rank info with aligned splits
            decomp._recompute_rank_info_from_splits()

            decompositions.append(decomp)

        return decompositions

    def solve(self):
        """Execute V-Cycles until convergence."""
        # Reset timers
        self._time_compute = 0.0
        self._time_halo = 0.0
        self._time_mpi = 0.0

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
        """Helper to compute global RMS norm of residual (matching JacobiPoisson).

        Uses allreduce to gather both the sum of squared residuals AND the total
        interior point count across all ranks, ensuring all ranks compute the
        same norm value.

        For cubic decomposition, properly excludes global boundary cells.
        """
        if lvl is not None and self.decomposition_strategy == "cubic":
            # For cubic, we need to exclude global boundary cells from residual
            decomp = lvl.decomposition
            info = decomp.get_rank_info(self.rank)
            gs = info.global_start
            ge = info.global_end
            N = decomp.N

            # Compute slice that excludes halos AND global boundaries
            # Local index 1 corresponds to global_start, local index -2 corresponds to global_end - 1
            slices = []
            for dim in range(3):
                # Start: skip halo (always), plus skip if at global boundary
                start = 2 if gs[dim] == 0 else 1
                # End: skip halo (always), plus skip if at global boundary
                end = -2 if ge[dim] == N else -1
                # Handle edge case where end would be 0 or positive
                if end == 0:
                    end = None
                slices.append(slice(start, end))

            interior = r_array[tuple(slices)]
            local_res_sq = np.sum(interior ** 2)
            local_interior_pts = float(interior.size)
        else:
            # For sliced or single-rank, the simple slice works
            local_res_sq = np.sum(r_array[1:-1, 1:-1, 1:-1] ** 2)
            local_interior_pts = float(np.prod(np.array(r_array.shape) - 2))

        t0 = MPI.Wtime()
        # Reduce both values in a single allreduce for efficiency
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
            current_residual_norm = 0.0 # Not needed for coarse levels
        
        # Base Case: Coarsest Level - solve with Jacobi until tolerance
        # Use limited iterations on coarse grid (10-20 sweeps is typical for V-cycle)
        if level_idx == self.levels - 1:
            return self._coarse_solve(lvl, max_iters=20)

        # 3. Restriction: f_coarse = R(r_fine)
        next_lvl = self.grid_levels[level_idx + 1]

        # Ensure residual halos are up to date before restriction
        t0 = MPI.Wtime()
        lvl.communicator.exchange_halos(lvl.r, lvl.decomposition, self.rank, self.comm, level=level_idx)
        self._time_halo += MPI.Wtime() - t0

        t0 = MPI.Wtime()
        # Use halo-aware operators for cubic decomposition (all dims have halos)
        if self.decomposition_strategy == "cubic":
            restrict_halo(lvl.r, next_lvl.f)
        else:
            restrict(lvl.r, next_lvl.f)
        self._time_compute += MPI.Wtime() - t0

        # Exchange halos on coarse RHS after restriction (needed for cubic decomposition)
        t0 = MPI.Wtime()
        next_lvl.communicator.exchange_halos(next_lvl.f, next_lvl.decomposition, self.rank, self.comm, level=level_idx + 1)
        self._time_halo += MPI.Wtime() - t0

        next_lvl.u[:] = 0.0
        
        # 4. Recursion
        self.v_cycle(level_idx + 1)
        
        # 5. Prolongation: u_fine += P(u_coarse_correction)

        # Refresh coarse halos before prolongation
        t0 = MPI.Wtime()
        next_lvl.communicator.exchange_halos(next_lvl.u, next_lvl.decomposition, self.rank, self.comm, level=level_idx + 1)
        self._time_halo += MPI.Wtime() - t0
        
        lvl.r[:] = 0.0
        t0 = MPI.Wtime()
        # Use halo-aware operators for cubic decomposition (all dims have halos)
        if self.decomposition_strategy == "cubic":
            prolong_halo(next_lvl.u, lvl.r)
        else:
            prolong(next_lvl.u, lvl.r)
        self._time_compute += MPI.Wtime() - t0

        # Exchange halos on prolongated correction (needed for cubic decomposition)
        t0 = MPI.Wtime()
        lvl.communicator.exchange_halos(lvl.r, lvl.decomposition, self.rank, self.comm, level=level_idx)
        self._time_halo += MPI.Wtime() - t0

        lvl.u[1:-1, 1:-1, 1:-1] += lvl.r[1:-1, 1:-1, 1:-1]

        # For cubic decomposition, enforce Dirichlet BC at global boundaries
        if self.decomposition_strategy == "cubic":
            lvl.decomposition.apply_boundary_conditions(lvl.u, self.rank)

        # 6. Post-smoothing
        for _ in range(self.n_smooth):
            self._smooth(lvl)
            
        return current_residual_norm

    def _coarse_solve(self, lvl: GridLevel, max_iters: int = 2000) -> float:
        """Solve on the coarsest grid with Jacobi until tolerance or max iterations."""
        current_residual_norm = float("inf")
        iterations = 0
        while iterations < max_iters:
            self._smooth(lvl)
            self._compute_residual(lvl)
            current_residual_norm = self._get_global_residual_norm(lvl.r, lvl)
            iterations += 1
            if current_residual_norm < self.config.tolerance:
                break
        return current_residual_norm

    def fmg_solve(self, cycles: int = 1):
        """Full Multigrid (FMG) cycle starting from the coarsest grid."""
        # Reset timers
        self._time_compute = 0.0
        self._time_halo = 0.0
        self._time_mpi = 0.0

        # Clear solution arrays (keep f - each level has source at its resolution)
        for lvl in self.grid_levels:
            lvl.u.fill(0.0)
            lvl.u_temp.fill(0.0)
            lvl.r.fill(0.0)

        t_start = MPI.Wtime()
        residual = None
        coarse_ok = True

        for _ in range(cycles):
            # Solve on coarsest level
            coarse_res = self._coarse_solve(self.grid_levels[-1])
            residual = coarse_res
            coarse_ok = coarse_ok and coarse_res < self.config.tolerance

            # Ascend hierarchy: prolongate and refine with a V-cycle at each level
            for l in reversed(range(self.levels - 1)):
                coarse = self.grid_levels[l + 1]
                fine = self.grid_levels[l]

                # Halo sync before prolongation
                t0 = MPI.Wtime()
                coarse.communicator.exchange_halos(coarse.u, coarse.decomposition, self.rank, self.comm, level=l + 1)
                self._time_halo += MPI.Wtime() - t0

                # Prolong coarse solution as initial guess on fine
                fine.r.fill(0.0)
                t0 = MPI.Wtime()
                # Use halo-aware operators for cubic decomposition (all dims have halos)
                if self.decomposition_strategy == "cubic":
                    prolong_halo(coarse.u, fine.r)
                else:
                    prolong(coarse.u, fine.r)
                self._time_compute += MPI.Wtime() - t0

                # Exchange halos on prolongated solution (needed for cubic decomposition)
                t0 = MPI.Wtime()
                fine.communicator.exchange_halos(fine.r, fine.decomposition, self.rank, self.comm, level=l)
                self._time_halo += MPI.Wtime() - t0

                fine.u.fill(0.0)
                fine.u[1:-1, 1:-1, 1:-1] = fine.r[1:-1, 1:-1, 1:-1]

                # For cubic decomposition, enforce Dirichlet BC at global boundaries
                if self.decomposition_strategy == "cubic":
                    fine.decomposition.apply_boundary_conditions(fine.u, self.rank)

                # Smooth the interpolated guess
                for _ in range(self.n_smooth):
                    self._smooth(fine)

                # Refine with one V-cycle from this level
                residual = self.v_cycle(l)

        # Final residual check on finest level
        fine_lvl = self.grid_levels[0]
        self._compute_residual(fine_lvl)
        residual = self._get_global_residual_norm(fine_lvl.r, fine_lvl)

        # Optional finishing V-cycles on finest level to enforce tolerance
        post_iters = 0
        while residual >= self.config.tolerance and post_iters < min(self.fmg_post_cycles, self.config.max_iter):
            residual = self.v_cycle(0)
            post_iters += 1

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
        lvl.communicator.exchange_halos(lvl.u, lvl.decomposition, self.rank, self.comm, level=lvl.level_index)
        self._time_halo += MPI.Wtime() - t0
        
        t0 = MPI.Wtime()
        lvl.kernel.step(lvl.u, lvl.u_temp, lvl.f)
        lvl.decomposition.apply_boundary_conditions(lvl.u_temp, self.rank)
        self._time_compute += MPI.Wtime() - t0
        lvl.u, lvl.u_temp = lvl.u_temp, lvl.u

    def _compute_residual(self, lvl: GridLevel):
        """Compute residual r = f - Au."""
        u = lvl.u
        f = lvl.f
        r = lvl.r
        h2 = lvl.h * lvl.h

        t0 = MPI.Wtime()
        lvl.communicator.exchange_halos(u, lvl.decomposition, self.rank, self.comm, level=lvl.level_index)
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

        # For cubic decomposition, zero out residual at global boundaries
        # (those points have u=0 by Dirichlet BC, so residual should be 0)
        if self.decomposition_strategy == "cubic":
            lvl.decomposition.apply_boundary_conditions(r, self.rank)

        self._time_compute += MPI.Wtime() - t0
        
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
        t0 = MPI.Wtime()
        global_sq_error = self.comm.allreduce(local_sq_error, op=MPI.SUM)
        self._time_mpi += MPI.Wtime() - t0
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

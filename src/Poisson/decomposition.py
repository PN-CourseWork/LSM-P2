"""Domain decomposition for distributed parallel computation.

Provides a clean abstraction for partitioning 3D grids across multiple ranks.
Pure geometric/mathematical decomposition with no MPI dependencies.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class RankInfo:
    """Decomposition information for a single rank."""
    rank: int

    # Local domain (interior points only)
    local_shape: tuple[int, int, int]  # (nx, ny, nz) interior
    global_start: tuple[int, int, int]  # (i, j, k) in global grid
    global_end: tuple[int, int, int]    # (i, j, k) exclusive

    # Ghost zones
    ghosted_shape: tuple[int, int, int]  # Shape including ghosts

    # Neighbors (None if at boundary)
    neighbors: dict  # Key: direction ('x_lower', etc), Value: rank or None

    # Communication metadata
    n_neighbors: int
    ghost_cells_total: int


class DomainDecomposition:
    """Domain decomposition for distributed computation.

    Computes how a 3D grid should be partitioned across multiple ranks.
    Pure geometric logic with no MPI dependencies.

    Parameters
    ----------
    N : int
        Global grid size (including boundaries)
    size : int
        Number of ranks
    strategy : str
        Decomposition strategy: 'sliced' (1D along Z) or 'cubic' (3D)

    Examples
    --------
    # Compute decomposition for 8 ranks
    >>> decomp = DomainDecomposition(N=100, size=8, strategy='cubic')
    >>> for rank in range(8):
    ...     info = decomp.get_rank_info(rank)
    ...     print(f"Rank {rank}: {info.local_shape}")
    """

    def __init__(self, N, size, strategy='sliced'):
        self.N = N
        self.size = size
        self.strategy = strategy

        # Decompose domain
        if strategy == 'sliced':
            self._decompose_sliced()
        elif strategy == 'cubic':
            self._decompose_cubic()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    # =========================================================================
    # Query Interface
    # =========================================================================

    def get_rank_info(self, rank):
        """Get decomposition info for a specific rank.

        Parameters
        ----------
        rank : int
            Rank to query

        Returns
        -------
        RankInfo
            Complete decomposition information for this rank
        """
        return self._rank_info[rank]

    def get_all_rank_info(self):
        """Get decomposition info for all ranks.

        Returns
        -------
        list of RankInfo
            Information for all ranks
        """
        return self._rank_info

    # =========================================================================
    # Internal Decomposition Logic
    # =========================================================================

    def _decompose_sliced(self):
        """Decompose domain with 1D slicing along Z-axis."""
        interior_N = self.N - 2

        # Compute decomposition for all ranks
        self._rank_info = []

        for rank in range(self.size):
            # Compute local size in Z direction
            base_size, remainder = divmod(interior_N, self.size)
            local_nz = base_size + (1 if rank < remainder else 0)

            # Compute Z start/end indices
            if rank < remainder:
                z_start = rank * (base_size + 1) + 1
            else:
                z_start = remainder * (base_size + 1) + (rank - remainder) * base_size + 1
            z_end = z_start + local_nz

            # Local shape (interior)
            local_shape = (local_nz, self.N, self.N)

            # Ghosted shape
            ghosted_shape = (local_nz + 2, self.N, self.N)

            # Global extent
            global_start = (z_start, 0, 0)
            global_end = (z_end, self.N, self.N)

            # Neighbors (only Z direction for sliced)
            neighbors = {}
            neighbors['z_lower'] = rank - 1 if rank > 0 else None
            neighbors['z_upper'] = rank + 1 if rank < self.size - 1 else None

            n_neighbors = sum(1 for n in neighbors.values() if n is not None)

            # Ghost cells: 2 full XY planes per neighbor
            ghost_cells_total = n_neighbors * self.N * self.N

            info = RankInfo(
                rank=rank,
                local_shape=local_shape,
                global_start=global_start,
                global_end=global_end,
                ghosted_shape=ghosted_shape,
                neighbors=neighbors,
                n_neighbors=n_neighbors,
                ghost_cells_total=ghost_cells_total
            )
            self._rank_info.append(info)

    def _decompose_cubic(self):
        """Decompose domain with 3D Cartesian grid.

        Matches legacy cubic.py: splits FULL N (including boundaries) across ranks.
        Array layout is (Z, Y, X) to match C-ordering.
        """
        from mpi4py import MPI

        # Use MPI's optimal factorization
        self.dims = MPI.Compute_dims(self.size, 3)
        px, py, pz = self.dims

        # Split FULL N (not N-2) - matches legacy approach
        def split_sizes(n, parts):
            base = n // parts
            rem = n % parts
            counts = [base + (1 if i < rem else 0) for i in range(parts)]
            starts = [sum(counts[:i]) for i in range(parts)]
            return counts, starts

        N = self.N
        nx_counts, nx_starts = split_sizes(N, px)
        ny_counts, ny_starts = split_sizes(N, py)
        nz_counts, nz_starts = split_sizes(N, pz)

        # Store for later use
        self._split_info = {
            'nx': (nx_counts, nx_starts),
            'ny': (ny_counts, ny_starts),
            'nz': (nz_counts, nz_starts),
        }

        self._rank_info = []

        for rank in range(self.size):
            # Compute 3D coords
            iz = rank % pz
            iy = (rank // pz) % py
            ix = rank // (py * pz)

            # Local sizes (of full domain portion, not interior)
            local_nz = nz_counts[iz]
            local_ny = ny_counts[iy]
            local_nx = nx_counts[ix]

            # Global starts (0-based in full N grid)
            z0 = nz_starts[iz]
            y0 = ny_starts[iy]
            x0 = nx_starts[ix]

            # local_shape is portion of full domain (Z, Y, X)
            local_shape = (local_nz, local_ny, local_nx)
            # Ghosted: +2 in each dimension for halo exchange
            ghosted_shape = (local_nz + 2, local_ny + 2, local_nx + 2)

            # Global start/end (0-based in full N grid)
            global_start = (z0, y0, x0)
            global_end = (z0 + local_nz, y0 + local_ny, x0 + local_nx)

            # Neighbors
            neighbors = {}
            neighbors['x_lower'] = self._cart_neighbor(ix-1, iy, iz, px, py, pz)
            neighbors['x_upper'] = self._cart_neighbor(ix+1, iy, iz, px, py, pz)
            neighbors['y_lower'] = self._cart_neighbor(ix, iy-1, iz, px, py, pz)
            neighbors['y_upper'] = self._cart_neighbor(ix, iy+1, iz, px, py, pz)
            neighbors['z_lower'] = self._cart_neighbor(ix, iy, iz-1, px, py, pz)
            neighbors['z_upper'] = self._cart_neighbor(ix, iy, iz+1, px, py, pz)

            n_neighbors = sum(1 for n in neighbors.values() if n is not None)

            nz, ny, nx = local_shape
            ghost_cells = 0
            if neighbors['x_lower'] is not None or neighbors['x_upper'] is not None:
                ghost_cells += 2 * nz * ny
            if neighbors['y_lower'] is not None or neighbors['y_upper'] is not None:
                ghost_cells += 2 * nz * nx
            if neighbors['z_lower'] is not None or neighbors['z_upper'] is not None:
                ghost_cells += 2 * ny * nx

            info = RankInfo(
                rank=rank,
                local_shape=local_shape,
                global_start=global_start,
                global_end=global_end,
                ghosted_shape=ghosted_shape,
                neighbors=neighbors,
                n_neighbors=n_neighbors,
                ghost_cells_total=ghost_cells
            )
            self._rank_info.append(info)

    def _cart_neighbor(self, ix, iy, iz, px, py, pz):
        """Get neighbor rank or None if out of bounds."""
        if ix < 0 or ix >= px or iy < 0 or iy >= py or iz < 0 or iz >= pz:
            return None
        return ix * (py * pz) + iy * pz + iz

    def _factorize_3d(self, n):
        """Simple 3D factorization (as cubic as possible)."""
        candidates = np.arange(1, int(n**0.5) + 1)
        divisors = np.concatenate([candidates[n % candidates == 0],
                                   n // candidates[n % candidates == 0]])
        divisors = np.unique(divisors)

        best = (n, 1, 1)
        best_score = float('inf')

        for i in divisors:
            remaining = n // i
            valid_j = divisors[divisors <= remaining]
            for j in valid_j[remaining % valid_j == 0]:
                k = remaining // j
                score = (i - j)**2 + (j - k)**2 + (k - i)**2
                if score < best_score:
                    best = (int(i), int(j), int(k))
                    best_score = score

        return best

    # =========================================================================
    # Solver Interface Methods
    # =========================================================================

    def initialize_local_arrays_distributed(self, N, rank, comm):
        """Initialize local arrays with ghost zones for this rank.

        Parameters
        ----------
        N : int
            Global grid size
        rank : int
            MPI rank
        comm : MPI.Comm
            MPI communicator

        Returns
        -------
        tuple
            (u1, u2, f) local arrays with ghost zones
        """
        info = self.get_rank_info(rank)
        shape = info.ghosted_shape

        u1 = np.zeros(shape, dtype=np.float64)
        u2 = np.zeros(shape, dtype=np.float64)
        f_local = np.zeros(shape, dtype=np.float64)

        # Build local source term using physical coordinates
        h = 2.0 / (N - 1)

        if self.strategy == 'sliced':
            # Sliced: (nz, N, N) - only z decomposed
            gs = info.global_start
            nz_local = info.local_shape[0]

            # Global z indices for this rank's interior
            z_indices = np.arange(gs[0], gs[0] + nz_local)
            zs = -1.0 + z_indices * h
            ys = np.linspace(-1, 1, N)
            xs = np.linspace(-1, 1, N)

            Z, Y, X = np.meshgrid(zs, ys, xs, indexing='ij')
            f_local[1:-1, :, :] = 3 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)
        else:
            # Cubic: (nz, ny, nx) - all dims decomposed
            # global_start is (z0, y0, x0) - 0-based indices in full N grid
            gs = info.global_start
            nz, ny, nx = info.local_shape

            # Global indices for this rank's local portion
            z_indices = np.arange(gs[0], gs[0] + nz)
            y_indices = np.arange(gs[1], gs[1] + ny)
            x_indices = np.arange(gs[2], gs[2] + nx)

            # Physical coordinates: x = -1 + i*h where i is 0..N-1
            zs = -1.0 + z_indices * h
            ys = -1.0 + y_indices * h
            xs = -1.0 + x_indices * h

            # Build meshgrid with (Z, Y, X) ordering
            Zl = zs.reshape((nz, 1, 1))
            Yl = ys.reshape((1, ny, 1))
            Xl = xs.reshape((1, 1, nx))

            f_local[1:-1, 1:-1, 1:-1] = 3 * np.pi**2 * np.sin(np.pi * Xl) * np.sin(np.pi * Yl) * np.sin(np.pi * Zl)

        return u1, u2, f_local

    def extract_interior(self, u_local):
        """Extract interior points from local array (excluding ghosts)."""
        if self.strategy == 'sliced':
            return u_local[1:-1, :, :].copy()
        else:
            return u_local[1:-1, 1:-1, 1:-1].copy()

    def get_interior_placement(self, rank_id, N, comm):
        """Get slice for placing rank's interior in global array.

        For cubic, global array is (N, N, N) with (Z, Y, X) ordering.
        """
        info = self.get_rank_info(rank_id)

        if self.strategy == 'sliced':
            gs = info.global_start
            ge = info.global_end
            return (slice(gs[0], ge[0]), slice(None), slice(None))
        else:
            # Cubic: global_start is already 0-based in full N grid
            gs = info.global_start
            ge = info.global_end
            return (slice(gs[0], ge[0]), slice(gs[1], ge[1]), slice(gs[2], ge[2]))

    def get_neighbors(self, rank):
        """Get neighbor dict for a rank (for communicator interface)."""
        return self.get_rank_info(rank).neighbors

    def apply_boundary_conditions(self, u_local, rank):
        """Apply Dirichlet BCs to local array cells at physical boundaries.

        For cubic decomposition, some local cells may correspond to global
        boundary indices (0 or N-1) which should be kept at 0.

        Parameters
        ----------
        u_local : np.ndarray
            Local array with ghost zones (modified in-place)
        rank : int
            MPI rank
        """
        if self.strategy != 'cubic':
            return  # Sliced doesn't need this - boundaries are handled differently

        info = self.get_rank_info(rank)
        gs = info.global_start
        ge = info.global_end
        N = self.N

        # Check each dimension for physical boundaries
        # The local interior is at indices [1:-1, 1:-1, 1:-1] in the ghosted array
        # Local index 1 corresponds to global_start
        # Local index -2 corresponds to global_end - 1

        # Z boundaries (axis 0)
        if gs[0] == 0:  # Has z=0 boundary
            u_local[1, :, :] = 0.0
        if ge[0] == N:  # Has z=N-1 boundary
            u_local[-2, :, :] = 0.0

        # Y boundaries (axis 1)
        if gs[1] == 0:  # Has y=0 boundary
            u_local[:, 1, :] = 0.0
        if ge[1] == N:  # Has y=N-1 boundary
            u_local[:, -2, :] = 0.0

        # X boundaries (axis 2)
        if gs[2] == 0:  # Has x=0 boundary
            u_local[:, :, 1] = 0.0
        if ge[2] == N:  # Has x=N-1 boundary
            u_local[:, :, -2] = 0.0


class NoDecomposition:
    """Stub decomposition for single-rank (sequential) execution."""

    def initialize_local_arrays_distributed(self, N, rank, comm):
        """Initialize arrays for single-rank execution."""
        from .problems import sinusoidal_source_term

        if rank == 0:
            u1 = np.zeros((N, N, N), dtype=np.float64)
            u2 = np.zeros((N, N, N), dtype=np.float64)
            f = sinusoidal_source_term(N)
            return u1, u2, f
        else:
            return None, None, None

    def extract_interior(self, u_local):
        """Extract interior (no-op for single rank, just copy)."""
        return u_local.copy()

    def get_interior_placement(self, rank_id, N, comm):
        """Get slice for full array (single rank owns everything)."""
        return (slice(None), slice(None), slice(None))

    def get_neighbors(self, rank):
        """No neighbors for single-rank execution."""
        return {}

    def apply_boundary_conditions(self, u_local, rank):
        """No-op for single-rank execution (boundaries handled by kernel)."""
        pass

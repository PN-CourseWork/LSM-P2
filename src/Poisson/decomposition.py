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
        """Decompose domain with 3D Cartesian grid."""
        # Compute 3D processor grid dimensions
        self.dims = self._factorize_3d(self.size)

        interior_N = self.N - 2

        # Compute decomposition for all ranks
        self._rank_info = []

        for rank in range(self.size):
            # Get 3D coordinates for this rank
            coords = self._rank_to_coords(rank, self.dims)

            # Decompose each dimension
            local_shape = []
            global_start = []
            global_end = []

            for dim_idx, (dim_size, coord) in enumerate(zip(self.dims, coords)):
                base_size, remainder = divmod(interior_N, dim_size)
                local_size = base_size + (1 if coord < remainder else 0)

                if coord < remainder:
                    start = coord * (base_size + 1) + 1
                else:
                    start = remainder * (base_size + 1) + (coord - remainder) * base_size + 1

                end = start + local_size

                local_shape.append(local_size)
                global_start.append(start)
                global_end.append(end)

            local_shape = tuple(local_shape)
            global_start = tuple(global_start)
            global_end = tuple(global_end)

            # Ghosted shape (add 2 to each dimension)
            ghosted_shape = tuple(n + 2 for n in local_shape)

            # Compute neighbors (6 faces)
            neighbors = self._get_cubic_neighbors(rank, coords, self.dims)

            n_neighbors = sum(1 for n in neighbors.values() if n is not None)

            # Ghost cells: count all ghost faces
            nx, ny, nz = local_shape
            ghost_cells = 0
            if neighbors['x_lower'] is not None or neighbors['x_upper'] is not None:
                ghost_cells += 2 * ny * nz
            if neighbors['y_lower'] is not None or neighbors['y_upper'] is not None:
                ghost_cells += 2 * nx * nz
            if neighbors['z_lower'] is not None or neighbors['z_upper'] is not None:
                ghost_cells += 2 * nx * ny

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

    def _rank_to_coords(self, rank, dims):
        """Convert linear rank to 3D coordinates."""
        px, py, pz = dims
        k = rank % pz
        j = (rank // pz) % py
        i = rank // (py * pz)
        return (i, j, k)

    def _coords_to_rank(self, coords, dims):
        """Convert 3D coordinates to linear rank."""
        i, j, k = coords
        px, py, pz = dims
        return i * (py * pz) + j * pz + k

    def _get_cubic_neighbors(self, rank, coords, dims):
        """Compute 6-face neighbors for cubic decomposition."""
        i, j, k = coords
        px, py, pz = dims

        neighbors = {}

        # X direction
        neighbors['x_lower'] = self._coords_to_rank((i-1, j, k), dims) if i > 0 else None
        neighbors['x_upper'] = self._coords_to_rank((i+1, j, k), dims) if i < px-1 else None

        # Y direction
        neighbors['y_lower'] = self._coords_to_rank((i, j-1, k), dims) if j > 0 else None
        neighbors['y_upper'] = self._coords_to_rank((i, j+1, k), dims) if j < py-1 else None

        # Z direction
        neighbors['z_lower'] = self._coords_to_rank((i, j, k-1), dims) if k > 0 else None
        neighbors['z_upper'] = self._coords_to_rank((i, j, k+1), dims) if k < pz-1 else None

        return neighbors

    def _factorize_3d(self, n):
        """Simple 3D factorization (as cubic as possible)."""
        # Find all divisors of n
        candidates = np.arange(1, int(n**0.5) + 1)
        divisors = np.concatenate([candidates[n % candidates == 0],
                                   n // candidates[n % candidates == 0]])
        divisors = np.unique(divisors)

        # Find triplet (i,j,k) with i*j*k=n that minimizes variance
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

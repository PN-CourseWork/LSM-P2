"""PETSc DMDA-inspired domain decomposition.

This module provides a clean abstraction for domain decomposition, inspired by
PETSc's DMDA (Distributed Mesh Data Array) approach. The DomainDecomposition
object encapsulates all partitioning logic and provides a query interface.

Design Philosophy:
- Decomposition is a first-class object (not a strategy pattern)
- Ranks query "what's my portion?" rather than computing it themselves
- Single source of truth for domain partitioning
- Supports analysis mode for visualization without MPI
"""

import numpy as np
from mpi4py import MPI
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
    """PETSc DMDA-inspired domain decomposition object.

    Encapsulates ALL information about how a 3D grid is partitioned across
    MPI ranks. Provides query interface for local domains, ghosts, and neighbors.

    Parameters
    ----------
    N : int
        Global grid size (including boundaries)
    comm : MPI.Comm, optional
        MPI communicator. If None, uses MPI.COMM_WORLD
    stencil_type : str
        Decomposition type: 'sliced' (1D along Z) or 'cubic' (3D)
    analyze_only : bool
        If True, creates analysis-only mode for visualization without MPI.
        Provide size parameter instead of comm.
    size : int, optional
        Number of ranks for analysis mode

    Examples
    --------
    # Real MPI execution:
    >>> da = DomainDecomposition(N=100, stencil_type='sliced')
    >>> local_shape = da.get_local_shape()
    >>> u_local = da.create_local_vector()

    # Analysis for visualization:
    >>> da = DomainDecomposition(N=100, size=8, stencil_type='sliced', analyze_only=True)
    >>> for rank in range(8):
    ...     info = da.get_rank_info(rank)
    ...     print(f"Rank {rank}: {info.local_shape}")
    """

    def __init__(self, N, comm=None, stencil_type='sliced', analyze_only=False, size=None):
        self.N = N
        self.stencil_type = stencil_type
        self.analyze_only = analyze_only

        if analyze_only:
            # Analysis mode - no actual MPI
            if size is None:
                raise ValueError("Must provide 'size' parameter in analyze_only mode")
            self.comm = None
            self.rank = 0
            self.size = size
        else:
            # Real MPI mode
            self.comm = comm if comm is not None else MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()

        # Decompose domain
        if stencil_type == 'sliced':
            self._decompose_sliced()
        elif stencil_type == 'cubic':
            self._decompose_cubic()
        else:
            raise ValueError(f"Unknown stencil_type: {stencil_type}")

    # =========================================================================
    # Query Interface (PETSc DMDA-style)
    # =========================================================================

    def get_local_shape(self, with_ghosts=True):
        """Get local array shape for this rank.

        Parameters
        ----------
        with_ghosts : bool
            If True, includes ghost zones (default)

        Returns
        -------
        shape : tuple
            Array shape (nx, ny, nz)
        """
        if self.analyze_only:
            raise RuntimeError("Use get_rank_info() in analyze_only mode")

        if with_ghosts:
            return self._rank_info[self.rank].ghosted_shape
        else:
            return self._rank_info[self.rank].local_shape

    def get_corners(self):
        """Get local domain extent (PETSc DMDAGetCorners).

        Returns
        -------
        starts : tuple
            (i_start, j_start, k_start) in global grid
        sizes : tuple
            (ni, nj, nk) interior points owned by this rank
        """
        if self.analyze_only:
            raise RuntimeError("Use get_rank_info() in analyze_only mode")

        info = self._rank_info[self.rank]
        return info.global_start, info.local_shape

    def get_neighbors(self):
        """Get neighbor ranks for ghost exchange.

        Returns
        -------
        neighbors : dict
            Keys: direction string ('x_lower', 'x_upper', etc)
            Values: neighbor rank (int) or None if at boundary
        """
        if self.analyze_only:
            raise RuntimeError("Use get_rank_info() in analyze_only mode")

        return self._rank_info[self.rank].neighbors

    def get_rank_info(self, rank):
        """Get decomposition info for a specific rank (analysis mode).

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
        """Get decomposition info for all ranks (analysis mode).

        Returns
        -------
        list of RankInfo
            Information for all ranks
        """
        return self._rank_info

    # =========================================================================
    # Array Creation (PETSc-style)
    # =========================================================================

    def create_local_vector(self):
        """Create local array with ghost zones (PETSc DMCreateLocalVector).

        Returns
        -------
        array : np.ndarray
            Initialized to zeros, includes ghost zones
        """
        if self.analyze_only:
            raise RuntimeError("Cannot create arrays in analyze_only mode")

        shape = self.get_local_shape(with_ghosts=True)
        return np.zeros(shape, dtype=np.float64)

    def create_global_vector(self):
        """Create global array without ghost zones (PETSc DMCreateGlobalVector).

        Only makes sense on rank 0 or for sequential execution.

        Returns
        -------
        array : np.ndarray
            Full global array (N, N, N)
        """
        if self.analyze_only:
            raise RuntimeError("Cannot create arrays in analyze_only mode")

        return np.zeros((self.N, self.N, self.N), dtype=np.float64)

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
        if not self.analyze_only:
            self.dims = MPI.Compute_dims(self.size, 3)
        else:
            # Simple factorization for analysis
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
        # Try to find factors that make a cubic decomposition
        best = (n, 1, 1)
        best_score = float('inf')

        for i in range(1, int(n**0.34) + 2):
            if n % i == 0:
                remaining = n // i
                for j in range(1, int(remaining**0.5) + 2):
                    if remaining % j == 0:
                        k = remaining // j
                        # Score: variance from cubic
                        score = (i - j)**2 + (j - k)**2 + (k - i)**2
                        if score < best_score:
                            best = (i, j, k)
                            best_score = score

        return best

    # =========================================================================
    # Analysis Methods
    # =========================================================================

    def compute_surface_to_volume_ratio(self):
        """Compute average surface-to-volume ratio for ghost communication.

        Returns
        -------
        ratio : float
            Average ratio of ghost cells to interior cells
        """
        total_interior = 0
        total_ghosts = 0

        for info in self._rank_info:
            interior_cells = np.prod(info.local_shape)
            total_interior += interior_cells
            total_ghosts += info.ghost_cells_total

        return total_ghosts / total_interior if total_interior > 0 else 0.0

    def compute_load_balance(self):
        """Compute load balance metrics.

        Returns
        -------
        dict
            - 'min_cells': minimum interior cells per rank
            - 'max_cells': maximum interior cells per rank
            - 'avg_cells': average interior cells per rank
            - 'imbalance': (max - min) / avg
        """
        cell_counts = [np.prod(info.local_shape) for info in self._rank_info]

        min_cells = min(cell_counts)
        max_cells = max(cell_counts)
        avg_cells = sum(cell_counts) / len(cell_counts)

        imbalance = (max_cells - min_cells) / avg_cells if avg_cells > 0 else 0.0

        return {
            'min_cells': min_cells,
            'max_cells': max_cells,
            'avg_cells': avg_cells,
            'imbalance': imbalance
        }

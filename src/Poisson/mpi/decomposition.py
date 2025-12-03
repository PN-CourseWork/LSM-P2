"""Domain decomposition with MPI Cartesian topology."""

from __future__ import annotations
from mpi4py import MPI


class CartesianDecomposition:
    """Handles MPI Cartesian topology and domain splitting.

    Creates a Cartesian communicator and computes how the global
    domain is distributed across ranks.

    Parameters
    ----------
    N : int
        Global grid size (N x N x N including boundaries).
    comm : MPI.Comm
        MPI communicator.
    strategy : str
        'sliced' for 1D decomposition along z-axis,
        'cubic' for 3D decomposition.
    """

    def __init__(self, N: int, comm: MPI.Comm, strategy: str = "sliced"):
        self.N = N
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.strategy = strategy

        # Determine processor grid dimensions
        self.dims = self._compute_dims(strategy)
        self.pz, self.py, self.px = self.dims

        # Create Cartesian topology
        self.cart_comm, self._proc_coords = self._create_cartesian_topology()

        # Discover neighbors
        self.neighbors = self._find_neighbors()

        # Compute local domain
        self.local_shape, self.halo_shape, self.global_start, self.global_end = (
            self._compute_local_domain()
        )

        # Track physical boundaries
        self.is_boundary = self._find_boundaries()

    def _compute_dims(self, strategy: str) -> list[int]:
        """Compute processor grid dimensions [pz, py, px]."""
        if strategy == "sliced":
            return [self.size, 1, 1]
        elif strategy == "cubic":
            dims = list(MPI.Compute_dims(self.size, 3))
            return [dims[2], dims[1], dims[0]]
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'sliced' or 'cubic'.")

    def _create_cartesian_topology(self):
        """Create MPI Cartesian communicator."""
        # dims in [pz, py, px] order, but Create_cart expects [px, py, pz]
        cart_dims = [self.px, self.py, self.pz]
        periods = [False, False, False]

        cart_comm = self.comm.Create_cart(
            dims=cart_dims, periods=periods, reorder=False
        )

        # Get coordinates [ix, iy, iz] and convert to [iz, iy, ix] for array indexing
        cart_coords = cart_comm.Get_coords(self.rank)
        proc_coords = (cart_coords[2], cart_coords[1], cart_coords[0])

        return cart_comm, proc_coords

    def _find_neighbors(self) -> dict[str, int | None]:
        """Use Cart_shift to find neighbor ranks."""
        neighbors = {}

        # X direction (cart direction 0)
        x_src, x_dest = self.cart_comm.Shift(0, 1)
        neighbors["x_lower"] = x_src if x_src >= 0 else None
        neighbors["x_upper"] = x_dest if x_dest >= 0 else None

        # Y direction (cart direction 1)
        y_src, y_dest = self.cart_comm.Shift(1, 1)
        neighbors["y_lower"] = y_src if y_src >= 0 else None
        neighbors["y_upper"] = y_dest if y_dest >= 0 else None

        # Z direction (cart direction 2)
        z_src, z_dest = self.cart_comm.Shift(2, 1)
        neighbors["z_lower"] = z_src if z_src >= 0 else None
        neighbors["z_upper"] = z_dest if z_dest >= 0 else None

        return neighbors

    def _compute_local_domain(self):
        """Compute local domain size and position.

        Decomposes the interior points (indices 1 to N-2).
        Boundaries (indices 0 and N-1) are handled separately.
        """
        N = self.N
        interior_N = N - 2

        def split_interior(n_interior, n_parts):
            """Split n_interior points among n_parts ranks."""
            base = n_interior // n_parts
            rem = n_interior % n_parts
            counts = [base + (1 if i < rem else 0) for i in range(n_parts)]
            starts = [1 + sum(counts[:i]) for i in range(n_parts)]
            return counts, starts

        iz, iy, ix = self._proc_coords

        z_counts, z_starts = split_interior(interior_N, self.pz)
        y_counts, y_starts = split_interior(interior_N, self.py)
        x_counts, x_starts = split_interior(interior_N, self.px)

        local_nz = z_counts[iz]
        local_ny = y_counts[iy]
        local_nx = x_counts[ix]

        global_start_z = z_starts[iz]
        global_start_y = y_starts[iy]
        global_start_x = x_starts[ix]

        local_shape = (local_nz, local_ny, local_nx)
        halo_shape = (local_nz + 2, local_ny + 2, local_nx + 2)
        global_start = (global_start_z, global_start_y, global_start_x)
        global_end = (
            global_start_z + local_nz,
            global_start_y + local_ny,
            global_start_x + local_nx,
        )

        return local_shape, halo_shape, global_start, global_end

    def _find_boundaries(self) -> dict[str, bool]:
        """Determine which faces are physical boundaries."""
        N = self.N
        return {
            "z_lower": self.global_start[0] == 1,
            "z_upper": self.global_end[0] == N - 1,
            "y_lower": self.global_start[1] == 1,
            "y_upper": self.global_end[1] == N - 1,
            "x_lower": self.global_start[2] == 1,
            "x_upper": self.global_end[2] == N - 1,
        }

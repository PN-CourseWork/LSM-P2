"""Distributed grid abstraction for parallel computation.

This module provides a unified DistributedGrid class that encapsulates:
- Domain decomposition (sliced or cubic) with MPI Cartesian topology
- Halo exchange communication (numpy buffers or MPI datatypes)
- Array allocation and interior/boundary handling

Solvers interact with this single interface rather than managing
MPI details directly.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from mpi4py import MPI


@dataclass
class RankGeometry:
    """Geometry information for a single rank."""
    rank: int

    # Local domain shape (interior points owned by this rank)
    local_shape: tuple[int, int, int]

    # Shape including halo zones
    halo_shape: tuple[int, int, int]

    # Global indices of owned region (start inclusive, end exclusive)
    global_start: tuple[int, int, int]
    global_end: tuple[int, int, int]

    # Neighbor ranks (None if at physical boundary)
    neighbors: dict[str, int | None]


class DistributedGrid:
    """Unified distributed grid for parallel Poisson solvers.

    Handles both 1D (sliced) and 3D (cubic) decomposition using MPI
    Cartesian topology for proper neighbor discovery.

    Parameters
    ----------
    N : int
        Global grid size (N x N x N including boundaries)
    comm : MPI.Comm
        MPI communicator
    strategy : str
        'sliced' for 1D decomposition along z-axis
        'cubic' for 3D decomposition
        'auto' to choose based on rank count
    halo_exchange : str
        'numpy' for buffer-based exchange (default)
        'custom' for MPI derived datatypes (zero-copy)

    Example
    -------
    >>> grid = DistributedGrid(N=65, comm=MPI.COMM_WORLD, strategy='cubic')
    >>> u = grid.allocate()  # Allocate array with halos
    >>> grid.fill_source_term(f)  # Fill source term
    >>> grid.sync_halos(u)   # Exchange halo data
    """

    def __init__(
        self,
        N: int,
        comm: MPI.Comm = None,
        strategy: str = 'auto',
        halo_exchange: str = 'numpy'
    ):
        self.N = N
        self.comm = comm if comm is not None else MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.strategy = strategy
        self.halo_exchange_type = halo_exchange

        # Determine processor grid dimensions
        if strategy == 'sliced':
            # 1D decomposition along z-axis: [pz, 1, 1]
            self.dims = [self.size, 1, 1]
        elif strategy == 'cubic':
            # 3D decomposition
            dims = list(MPI.Compute_dims(self.size, 3))  # [px, py, pz]
            # Reorder to [pz, py, px] for array indexing (z,y,x)
            self.dims = [dims[2], dims[1], dims[0]]
        elif strategy == 'auto':
            if self.size == 1:
                self.dims = [1, 1, 1]
            else:
                dims = list(MPI.Compute_dims(self.size, 3))
                self.dims = [dims[2], dims[1], dims[0]]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        self.pz, self.py, self.px = self.dims

        # Create MPI Cartesian topology
        self._create_cartesian_topology()

        # Compute decomposition
        self._compute_decomposition()

        # Set up halo exchange (buffers or datatypes)
        self._setup_halo_exchange()

        # Grid spacing
        self.h = 2.0 / (N - 1)

    def _create_cartesian_topology(self):
        """Create MPI Cartesian communicator for neighbor discovery."""
        # dims in [pz, py, px] order, but Create_cart expects [px, py, pz]
        cart_dims = [self.px, self.py, self.pz]
        periods = [False, False, False]  # Non-periodic boundaries

        self.cart_comm = self.comm.Create_cart(
            dims=cart_dims,
            periods=periods,
            reorder=False  # Keep rank ordering consistent
        )

        # Get this rank's coordinates in the Cartesian grid
        # Returns [ix, iy, iz] in the order of cart_dims
        coords = self.cart_comm.Get_coords(self.rank)
        self._cart_coords = coords  # [ix, iy, iz]

        # Convert to [iz, iy, ix] for array indexing
        self._proc_coords = (coords[2], coords[1], coords[0])

        # Use Cart_shift to get neighbors
        # Shift direction 0 = x, 1 = y, 2 = z (matching cart_dims order)
        self.neighbors = {}

        # X direction (cart direction 0)
        x_src, x_dest = self.cart_comm.Shift(0, 1)
        self.neighbors['x_lower'] = x_src if x_src >= 0 else None
        self.neighbors['x_upper'] = x_dest if x_dest >= 0 else None

        # Y direction (cart direction 1)
        y_src, y_dest = self.cart_comm.Shift(1, 1)
        self.neighbors['y_lower'] = y_src if y_src >= 0 else None
        self.neighbors['y_upper'] = y_dest if y_dest >= 0 else None

        # Z direction (cart direction 2)
        z_src, z_dest = self.cart_comm.Shift(2, 1)
        self.neighbors['z_lower'] = z_src if z_src >= 0 else None
        self.neighbors['z_upper'] = z_dest if z_dest >= 0 else None

    def _compute_decomposition(self):
        """Compute how the INTERIOR domain is split across ranks.

        Key design: We decompose the interior points (indices 1 to N-2).
        Boundaries (indices 0 and N-1) are handled separately.
        This ensures consistent indexing for multigrid operators.
        """
        N = self.N
        interior_N = N - 2  # Number of interior points per dimension

        # Split interior points along each decomposed dimension
        def split_interior(n_interior, n_parts):
            """Split n_interior points among n_parts ranks."""
            base = n_interior // n_parts
            rem = n_interior % n_parts
            counts = [base + (1 if i < rem else 0) for i in range(n_parts)]
            # Starts are relative to first interior point (global index 1)
            starts = [1 + sum(counts[:i]) for i in range(n_parts)]
            return counts, starts

        # Get this rank's position in the processor grid [iz, iy, ix]
        iz, iy, ix = self._proc_coords

        # Compute local sizes and global starts for each dimension
        z_counts, z_starts = split_interior(interior_N, self.pz)
        y_counts, y_starts = split_interior(interior_N, self.py)
        x_counts, x_starts = split_interior(interior_N, self.px)

        local_nz = z_counts[iz]
        local_ny = y_counts[iy]
        local_nx = x_counts[ix]

        global_start_z = z_starts[iz]
        global_start_y = y_starts[iy]
        global_start_x = x_starts[ix]

        # Store geometry
        self.local_shape = (local_nz, local_ny, local_nx)
        self.halo_shape = (local_nz + 2, local_ny + 2, local_nx + 2)
        self.global_start = (global_start_z, global_start_y, global_start_x)
        self.global_end = (
            global_start_z + local_nz,
            global_start_y + local_ny,
            global_start_x + local_nx
        )

        # Track which faces are physical boundaries
        self.is_boundary = {
            'z_lower': self.global_start[0] == 1,
            'z_upper': self.global_end[0] == N - 1,
            'y_lower': self.global_start[1] == 1,
            'y_upper': self.global_end[1] == N - 1,
            'x_lower': self.global_start[2] == 1,
            'x_upper': self.global_end[2] == N - 1,
        }

    def _setup_halo_exchange(self):
        """Set up buffers or datatypes for halo exchange."""
        nz, ny, nx = self.local_shape

        if self.halo_exchange_type == 'numpy':
            # Pre-allocate buffers for each face
            self._send_bufs = {}
            self._recv_bufs = {}

            face_sizes = {
                'z': ny * nx,
                'y': nz * nx,
                'x': nz * ny,
            }

            for axis in ['z', 'y', 'x']:
                for direction in ['lower', 'upper']:
                    key = f'{axis}_{direction}'
                    if self.neighbors[key] is not None:
                        self._send_bufs[key] = np.empty(face_sizes[axis], dtype=np.float64)
                        self._recv_bufs[key] = np.empty(face_sizes[axis], dtype=np.float64)

        elif self.halo_exchange_type == 'custom':
            # Create MPI derived datatypes for zero-copy exchange
            self._datatypes = self._create_datatypes()
        else:
            raise ValueError(f"Unknown halo_exchange type: {self.halo_exchange_type}")

    def _create_datatypes(self):
        """Create MPI datatypes for each face (zero-copy halo exchange)."""
        nz, ny, nx = self.halo_shape  # Include halos

        datatypes = {}

        # Z-face: ny_int x nx_int contiguous block
        # Interior is [1:-1, 1:-1] = (ny-2) x (nx-2)
        ny_int, nx_int = ny - 2, nx - 2
        nz_int = nz - 2

        # Z-face datatype: contiguous rows with stride nx
        dt = MPI.DOUBLE.Create_vector(ny_int, nx_int, nx)
        dt.Commit()
        datatypes['z'] = dt

        # Y-face datatype: nz_int planes, nx_int per plane, stride ny*nx
        dt = MPI.DOUBLE.Create_vector(nz_int, nx_int, ny * nx)
        dt.Commit()
        datatypes['y'] = dt

        # X-face datatype: 2D strided (most complex)
        # Each row: 1 element with stride nx
        # Stack nz_int * ny_int such rows
        row = MPI.DOUBLE.Create_vector(ny_int, 1, nx)
        row.Commit()
        dt = row.Create_hvector(nz_int, 1, ny * nx * MPI.DOUBLE.Get_size())
        dt.Commit()
        row.Free()
        datatypes['x'] = dt

        return datatypes

    def allocate(self, dtype=np.float64) -> np.ndarray:
        """Allocate a local array with halo zones."""
        return np.zeros(self.halo_shape, dtype=dtype)

    def sync_halos(self, arr: np.ndarray):
        """Exchange halo data with all neighbors.

        Uses either numpy buffers or MPI datatypes depending on
        halo_exchange_type setting.
        """
        if self.halo_exchange_type == 'numpy':
            self._sync_halos_numpy(arr)
        else:
            self._sync_halos_datatype(arr)

    def _sync_halos_numpy(self, arr: np.ndarray):
        """Halo exchange using numpy buffer copies and Sendrecv."""
        nz, ny, nx = self.local_shape

        # Z-direction
        if self.neighbors['z_lower'] is not None or self.neighbors['z_upper'] is not None:
            lo = self.neighbors['z_lower'] if self.neighbors['z_lower'] is not None else MPI.PROC_NULL
            hi = self.neighbors['z_upper'] if self.neighbors['z_upper'] is not None else MPI.PROC_NULL

            # Send to upper, receive from lower
            send_buf = np.ascontiguousarray(arr[-2, 1:-1, 1:-1])
            recv_buf = np.empty_like(send_buf)
            self.cart_comm.Sendrecv(send_buf, hi, 0, recv_buf, lo, 0)
            if self.neighbors['z_lower'] is not None:
                arr[0, 1:-1, 1:-1] = recv_buf.reshape(ny, nx)

            # Send to lower, receive from upper
            send_buf = np.ascontiguousarray(arr[1, 1:-1, 1:-1])
            recv_buf = np.empty_like(send_buf)
            self.cart_comm.Sendrecv(send_buf, lo, 1, recv_buf, hi, 1)
            if self.neighbors['z_upper'] is not None:
                arr[-1, 1:-1, 1:-1] = recv_buf.reshape(ny, nx)

        # Y-direction
        if self.neighbors['y_lower'] is not None or self.neighbors['y_upper'] is not None:
            lo = self.neighbors['y_lower'] if self.neighbors['y_lower'] is not None else MPI.PROC_NULL
            hi = self.neighbors['y_upper'] if self.neighbors['y_upper'] is not None else MPI.PROC_NULL

            send_buf = np.ascontiguousarray(arr[1:-1, -2, 1:-1])
            recv_buf = np.empty_like(send_buf)
            self.cart_comm.Sendrecv(send_buf, hi, 2, recv_buf, lo, 2)
            if self.neighbors['y_lower'] is not None:
                arr[1:-1, 0, 1:-1] = recv_buf.reshape(nz, nx)

            send_buf = np.ascontiguousarray(arr[1:-1, 1, 1:-1])
            recv_buf = np.empty_like(send_buf)
            self.cart_comm.Sendrecv(send_buf, lo, 3, recv_buf, hi, 3)
            if self.neighbors['y_upper'] is not None:
                arr[1:-1, -1, 1:-1] = recv_buf.reshape(nz, nx)

        # X-direction
        if self.neighbors['x_lower'] is not None or self.neighbors['x_upper'] is not None:
            lo = self.neighbors['x_lower'] if self.neighbors['x_lower'] is not None else MPI.PROC_NULL
            hi = self.neighbors['x_upper'] if self.neighbors['x_upper'] is not None else MPI.PROC_NULL

            send_buf = np.ascontiguousarray(arr[1:-1, 1:-1, -2])
            recv_buf = np.empty_like(send_buf)
            self.cart_comm.Sendrecv(send_buf, hi, 4, recv_buf, lo, 4)
            if self.neighbors['x_lower'] is not None:
                arr[1:-1, 1:-1, 0] = recv_buf.reshape(nz, ny)

            send_buf = np.ascontiguousarray(arr[1:-1, 1:-1, 1])
            recv_buf = np.empty_like(send_buf)
            self.cart_comm.Sendrecv(send_buf, lo, 5, recv_buf, hi, 5)
            if self.neighbors['x_upper'] is not None:
                arr[1:-1, 1:-1, -1] = recv_buf.reshape(nz, ny)

    def _sync_halos_datatype(self, arr: np.ndarray):
        """Halo exchange using MPI derived datatypes (zero-copy)."""
        nz, ny, nx = self.halo_shape
        arr_flat = arr.ravel()

        def flat_idx(z, y, x):
            return z * ny * nx + y * nx + x

        # Z-direction
        if self.neighbors['z_lower'] is not None or self.neighbors['z_upper'] is not None:
            lo = self.neighbors['z_lower'] if self.neighbors['z_lower'] is not None else MPI.PROC_NULL
            hi = self.neighbors['z_upper'] if self.neighbors['z_upper'] is not None else MPI.PROC_NULL
            dt = self._datatypes['z']

            # Send upper interior plane, receive into lower halo
            send_off = flat_idx(nz - 2, 1, 1)
            recv_off = flat_idx(0, 1, 1)
            self.cart_comm.Sendrecv(
                [arr_flat[send_off:], 1, dt], hi, 0,
                [arr_flat[recv_off:], 1, dt], lo, 0
            )
            if self.neighbors['z_lower'] is None:
                arr[0, 1:-1, 1:-1] = 0.0

            # Send lower interior plane, receive into upper halo
            send_off = flat_idx(1, 1, 1)
            recv_off = flat_idx(nz - 1, 1, 1)
            self.cart_comm.Sendrecv(
                [arr_flat[send_off:], 1, dt], lo, 1,
                [arr_flat[recv_off:], 1, dt], hi, 1
            )
            if self.neighbors['z_upper'] is None:
                arr[-1, 1:-1, 1:-1] = 0.0

        # Y-direction
        if self.neighbors['y_lower'] is not None or self.neighbors['y_upper'] is not None:
            lo = self.neighbors['y_lower'] if self.neighbors['y_lower'] is not None else MPI.PROC_NULL
            hi = self.neighbors['y_upper'] if self.neighbors['y_upper'] is not None else MPI.PROC_NULL
            dt = self._datatypes['y']

            send_off = flat_idx(1, ny - 2, 1)
            recv_off = flat_idx(1, 0, 1)
            self.cart_comm.Sendrecv(
                [arr_flat[send_off:], 1, dt], hi, 2,
                [arr_flat[recv_off:], 1, dt], lo, 2
            )
            if self.neighbors['y_lower'] is None:
                arr[1:-1, 0, 1:-1] = 0.0

            send_off = flat_idx(1, 1, 1)
            recv_off = flat_idx(1, ny - 1, 1)
            self.cart_comm.Sendrecv(
                [arr_flat[send_off:], 1, dt], lo, 3,
                [arr_flat[recv_off:], 1, dt], hi, 3
            )
            if self.neighbors['y_upper'] is None:
                arr[1:-1, -1, 1:-1] = 0.0

        # X-direction
        if self.neighbors['x_lower'] is not None or self.neighbors['x_upper'] is not None:
            lo = self.neighbors['x_lower'] if self.neighbors['x_lower'] is not None else MPI.PROC_NULL
            hi = self.neighbors['x_upper'] if self.neighbors['x_upper'] is not None else MPI.PROC_NULL
            dt = self._datatypes['x']

            send_off = flat_idx(1, 1, nx - 2)
            recv_off = flat_idx(1, 1, 0)
            self.cart_comm.Sendrecv(
                [arr_flat[send_off:], 1, dt], hi, 4,
                [arr_flat[recv_off:], 1, dt], lo, 4
            )
            if self.neighbors['x_lower'] is None:
                arr[1:-1, 1:-1, 0] = 0.0

            send_off = flat_idx(1, 1, 1)
            recv_off = flat_idx(1, 1, nx - 1)
            self.cart_comm.Sendrecv(
                [arr_flat[send_off:], 1, dt], lo, 5,
                [arr_flat[recv_off:], 1, dt], hi, 5
            )
            if self.neighbors['x_upper'] is None:
                arr[1:-1, 1:-1, -1] = 0.0

    def apply_boundary_conditions(self, arr: np.ndarray, value: float = 0.0):
        """Apply Dirichlet boundary conditions at physical boundaries."""
        if self.is_boundary['z_lower']:
            arr[0, :, :] = value
        if self.is_boundary['z_upper']:
            arr[-1, :, :] = value
        if self.is_boundary['y_lower']:
            arr[:, 0, :] = value
        if self.is_boundary['y_upper']:
            arr[:, -1, :] = value
        if self.is_boundary['x_lower']:
            arr[:, :, 0] = value
        if self.is_boundary['x_upper']:
            arr[:, :, -1] = value

    def fill_source_term(self, f: np.ndarray):
        """Fill source term for sinusoidal test problem.

        f = 3π² sin(πx) sin(πy) sin(πz)
        """
        h = self.h
        gs = self.global_start
        nz, ny, nx = self.local_shape

        # Compute physical coordinates for interior points
        z_global = np.arange(gs[0], gs[0] + nz)
        y_global = np.arange(gs[1], gs[1] + ny)
        x_global = np.arange(gs[2], gs[2] + nx)

        z_phys = -1.0 + z_global * h
        y_phys = -1.0 + y_global * h
        x_phys = -1.0 + x_global * h

        Z, Y, X = np.meshgrid(z_phys, y_phys, x_phys, indexing='ij')

        f[1:-1, 1:-1, 1:-1] = (
            3 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)
        )

    def compute_exact_solution(self, u: np.ndarray):
        """Fill exact solution for validation.

        u = sin(πx) sin(πy) sin(πz)
        """
        h = self.h
        gs = self.global_start
        nz, ny, nx = self.local_shape

        z_global = np.arange(gs[0], gs[0] + nz)
        y_global = np.arange(gs[1], gs[1] + ny)
        x_global = np.arange(gs[2], gs[2] + nx)

        z_phys = -1.0 + z_global * h
        y_phys = -1.0 + y_global * h
        x_phys = -1.0 + x_global * h

        Z, Y, X = np.meshgrid(z_phys, y_phys, x_phys, indexing='ij')

        u[1:-1, 1:-1, 1:-1] = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)

    def compute_l2_error(self, u: np.ndarray) -> float:
        """Compute L2 error against exact solution."""
        u_exact = self.allocate()
        self.compute_exact_solution(u_exact)

        local_sq_error = np.sum((u[1:-1, 1:-1, 1:-1] - u_exact[1:-1, 1:-1, 1:-1])**2)
        global_sq_error = self.cart_comm.allreduce(local_sq_error, op=MPI.SUM)

        return float(np.sqrt(self.h**3 * global_sq_error))

    def interior_slice(self):
        """Return slice tuple for interior points."""
        return (slice(1, -1), slice(1, -1), slice(1, -1))

    def get_halo_size_bytes(self) -> int:
        """Calculate total bytes transferred per halo exchange.

        Returns the total data size for all active neighbor communications
        (both send and receive), accounting for float64 (8 bytes per element).
        """
        nz, ny, nx = self.local_shape
        bytes_per_element = 8  # float64

        total_bytes = 0

        # Z-faces: ny * nx elements each
        if self.neighbors['z_lower'] is not None:
            total_bytes += ny * nx * bytes_per_element * 2  # send + recv
        if self.neighbors['z_upper'] is not None:
            total_bytes += ny * nx * bytes_per_element * 2

        # Y-faces: nz * nx elements each
        if self.neighbors['y_lower'] is not None:
            total_bytes += nz * nx * bytes_per_element * 2
        if self.neighbors['y_upper'] is not None:
            total_bytes += nz * nx * bytes_per_element * 2

        # X-faces: nz * ny elements each
        if self.neighbors['x_lower'] is not None:
            total_bytes += nz * ny * bytes_per_element * 2
        if self.neighbors['x_upper'] is not None:
            total_bytes += nz * ny * bytes_per_element * 2

        return total_bytes

    def get_geometry(self) -> RankGeometry:
        """Return geometry info for this rank."""
        return RankGeometry(
            rank=self.rank,
            local_shape=self.local_shape,
            halo_shape=self.halo_shape,
            global_start=self.global_start,
            global_end=self.global_end,
            neighbors=self.neighbors.copy(),
        )

    def coarsen(self) -> 'DistributedGrid':
        """Create a coarsened grid for multigrid.

        Returns a new DistributedGrid with N_coarse = (N-1)//2 + 1.
        """
        N_coarse = (self.N - 1) // 2 + 1
        if N_coarse < 3:
            raise ValueError("Grid too small to coarsen further")

        return DistributedGrid(
            N_coarse,
            self.comm,
            strategy=self.strategy,
            halo_exchange=self.halo_exchange_type
        )

    def __del__(self):
        """Clean up MPI datatypes."""
        if hasattr(self, '_datatypes'):
            for dt in self._datatypes.values():
                if dt != MPI.DATATYPE_NULL:
                    dt.Free()

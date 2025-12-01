"""Distributed grid abstraction for parallel computation.

This module provides a unified DistributedGrid class that encapsulates:
- Domain decomposition (sliced or cubic, unified logic)
- Halo exchange communication
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

    Handles both 1D (sliced) and 3D (cubic) decomposition with identical
    logic - just different processor grid dimensions.

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

    Example
    -------
    >>> grid = DistributedGrid(N=65, comm=MPI.COMM_WORLD, strategy='cubic')
    >>> u = grid.allocate()  # Allocate array with halos
    >>> grid.fill_source(f)  # Fill source term
    >>> grid.sync_halos(u)   # Exchange halo data
    """

    def __init__(self, N: int, comm: MPI.Comm = None, strategy: str = 'auto'):
        self.N = N
        self.comm = comm if comm is not None else MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.strategy = strategy

        # Determine processor grid dimensions
        if strategy == 'sliced':
            # 1D decomposition along z-axis
            self.dims = [self.size, 1, 1]  # [pz, py, px]
        elif strategy == 'cubic':
            # 3D decomposition
            self.dims = list(MPI.Compute_dims(self.size, 3))  # [px, py, pz]
            # Reorder to [pz, py, px] for array indexing (z,y,x)
            self.dims = [self.dims[2], self.dims[1], self.dims[0]]
        elif strategy == 'auto':
            if self.size == 1:
                self.dims = [1, 1, 1]
            else:
                self.dims = list(MPI.Compute_dims(self.size, 3))
                self.dims = [self.dims[2], self.dims[1], self.dims[0]]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        self.pz, self.py, self.px = self.dims

        # Compute decomposition
        self._compute_decomposition()

        # Pre-allocate communication buffers
        self._setup_buffers()

        # Grid spacing
        self.h = 2.0 / (N - 1)

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

        # Get this rank's position in the processor grid
        # Rank ordering: rank = iz + iy * pz + ix * pz * py
        iz = self.rank % self.pz
        iy = (self.rank // self.pz) % self.py
        ix = self.rank // (self.pz * self.py)
        self._proc_coords = (iz, iy, ix)

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

        # Compute neighbors
        self.neighbors = {}
        self.neighbors['z_lower'] = self._get_neighbor(iz - 1, iy, ix)
        self.neighbors['z_upper'] = self._get_neighbor(iz + 1, iy, ix)
        self.neighbors['y_lower'] = self._get_neighbor(iz, iy - 1, ix)
        self.neighbors['y_upper'] = self._get_neighbor(iz, iy + 1, ix)
        self.neighbors['x_lower'] = self._get_neighbor(iz, iy, ix - 1)
        self.neighbors['x_upper'] = self._get_neighbor(iz, iy, ix + 1)

        # Track which faces are physical boundaries
        self.is_boundary = {
            'z_lower': self.global_start[0] == 1,  # At z=1 means we touch z=0 boundary
            'z_upper': self.global_end[0] == N - 1,  # At z=N-2 means we touch z=N-1 boundary
            'y_lower': self.global_start[1] == 1,
            'y_upper': self.global_end[1] == N - 1,
            'x_lower': self.global_start[2] == 1,
            'x_upper': self.global_end[2] == N - 1,
        }

    def _get_neighbor(self, iz, iy, ix) -> int | None:
        """Get neighbor rank or None if out of bounds."""
        if iz < 0 or iz >= self.pz:
            return None
        if iy < 0 or iy >= self.py:
            return None
        if ix < 0 or ix >= self.px:
            return None
        return iz + iy * self.pz + ix * self.pz * self.py

    def _setup_buffers(self):
        """Pre-allocate buffers for halo exchange."""
        nz, ny, nx = self.local_shape

        # Buffers for each face
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

    def allocate(self, dtype=np.float64) -> np.ndarray:
        """Allocate a local array with halo zones."""
        return np.zeros(self.halo_shape, dtype=dtype)

    def sync_halos(self, arr: np.ndarray):
        """Exchange halo data with all neighbors.

        Uses non-blocking sends/receives for efficiency.
        """
        requests = []
        nz, ny, nx = self.local_shape

        # Z-direction
        if self.neighbors['z_lower'] is not None:
            # Send our first interior z-plane, receive into lower halo
            self._send_bufs['z_lower'][:] = arr[1, 1:-1, 1:-1].ravel()
            requests.append(self.comm.Isend(self._send_bufs['z_lower'], self.neighbors['z_lower']))
            requests.append(self.comm.Irecv(self._recv_bufs['z_lower'], self.neighbors['z_lower']))

        if self.neighbors['z_upper'] is not None:
            # Send our last interior z-plane, receive into upper halo
            self._send_bufs['z_upper'][:] = arr[-2, 1:-1, 1:-1].ravel()
            requests.append(self.comm.Isend(self._send_bufs['z_upper'], self.neighbors['z_upper']))
            requests.append(self.comm.Irecv(self._recv_bufs['z_upper'], self.neighbors['z_upper']))

        # Y-direction
        if self.neighbors['y_lower'] is not None:
            self._send_bufs['y_lower'][:] = arr[1:-1, 1, 1:-1].ravel()
            requests.append(self.comm.Isend(self._send_bufs['y_lower'], self.neighbors['y_lower']))
            requests.append(self.comm.Irecv(self._recv_bufs['y_lower'], self.neighbors['y_lower']))

        if self.neighbors['y_upper'] is not None:
            self._send_bufs['y_upper'][:] = arr[1:-1, -2, 1:-1].ravel()
            requests.append(self.comm.Isend(self._send_bufs['y_upper'], self.neighbors['y_upper']))
            requests.append(self.comm.Irecv(self._recv_bufs['y_upper'], self.neighbors['y_upper']))

        # X-direction
        if self.neighbors['x_lower'] is not None:
            self._send_bufs['x_lower'][:] = arr[1:-1, 1:-1, 1].ravel()
            requests.append(self.comm.Isend(self._send_bufs['x_lower'], self.neighbors['x_lower']))
            requests.append(self.comm.Irecv(self._recv_bufs['x_lower'], self.neighbors['x_lower']))

        if self.neighbors['x_upper'] is not None:
            self._send_bufs['x_upper'][:] = arr[1:-1, 1:-1, -2].ravel()
            requests.append(self.comm.Isend(self._send_bufs['x_upper'], self.neighbors['x_upper']))
            requests.append(self.comm.Irecv(self._recv_bufs['x_upper'], self.neighbors['x_upper']))

        # Wait for all communication to complete
        MPI.Request.Waitall(requests)

        # Unpack received data into halos
        if self.neighbors['z_lower'] is not None:
            arr[0, 1:-1, 1:-1] = self._recv_bufs['z_lower'].reshape(ny, nx)
        if self.neighbors['z_upper'] is not None:
            arr[-1, 1:-1, 1:-1] = self._recv_bufs['z_upper'].reshape(ny, nx)
        if self.neighbors['y_lower'] is not None:
            arr[1:-1, 0, 1:-1] = self._recv_bufs['y_lower'].reshape(nz, nx)
        if self.neighbors['y_upper'] is not None:
            arr[1:-1, -1, 1:-1] = self._recv_bufs['y_upper'].reshape(nz, nx)
        if self.neighbors['x_lower'] is not None:
            arr[1:-1, 1:-1, 0] = self._recv_bufs['x_lower'].reshape(nz, ny)
        if self.neighbors['x_upper'] is not None:
            arr[1:-1, 1:-1, -1] = self._recv_bufs['x_upper'].reshape(nz, ny)

    def apply_boundary_conditions(self, arr: np.ndarray, value: float = 0.0):
        """Apply Dirichlet boundary conditions at physical boundaries."""
        # At physical boundaries, the halo zone represents the boundary value
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
        # Interior at array indices [1:-1, 1:-1, 1:-1]
        # Global index = global_start + local_index
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
        global_sq_error = self.comm.allreduce(local_sq_error, op=MPI.SUM)

        return float(np.sqrt(self.h**3 * global_sq_error))

    def interior_slice(self):
        """Return slice tuple for interior points."""
        return (slice(1, -1), slice(1, -1), slice(1, -1))

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
        The decomposition is consistent so that:
        - coarse global index i corresponds to fine global index 2*i
        """
        N_coarse = (self.N - 1) // 2 + 1
        if N_coarse < 3:
            raise ValueError("Grid too small to coarsen further")

        # Create coarse grid with same strategy
        return DistributedGrid(N_coarse, self.comm, strategy=self.strategy)

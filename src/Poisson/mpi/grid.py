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
from mpi4py import MPI

from ..datastructures import RankGeometry
from .decomposition import CartesianDecomposition
from .halo import create_halo_exchanger


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
        'sliced' for 1D decomposition along z-axis (default)
        'cubic' for 3D decomposition
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
        strategy: str = "sliced",
        halo_exchange: str = "numpy",
    ):
        self.N = N
        self.comm = comm if comm is not None else MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.strategy = strategy
        self.halo_exchange_type = halo_exchange

        # Domain decomposition
        self._decomp = CartesianDecomposition(N, self.comm, strategy)

        # Copy decomposition attributes for direct access
        self.dims = self._decomp.dims
        self.pz, self.py, self.px = self.dims
        self.cart_comm = self._decomp.cart_comm
        self.neighbors = self._decomp.neighbors
        self.local_shape = self._decomp.local_shape
        self.halo_shape = self._decomp.halo_shape
        self.global_start = self._decomp.global_start
        self.global_end = self._decomp.global_end
        self.is_boundary = self._decomp.is_boundary

        # Halo exchange strategy
        self._halo_exchanger = create_halo_exchanger(halo_exchange)
        self._halo_exchanger.setup(self.local_shape, self.neighbors)

        # Grid spacing
        self.h = 2.0 / (N - 1)

    def allocate(self, dtype=np.float64) -> np.ndarray:
        """Allocate a local array with halo zones."""
        return np.zeros(self.halo_shape, dtype=dtype)

    def sync_halos(self, arr: np.ndarray):
        """Exchange halo data with all neighbors."""
        self._halo_exchanger.exchange(arr, self.cart_comm, self.neighbors)

    def apply_boundary_conditions(self, arr: np.ndarray, value: float = 0.0):
        """Apply Dirichlet boundary conditions at physical boundaries."""
        if self.is_boundary["z_lower"]:
            arr[0, :, :] = value
        if self.is_boundary["z_upper"]:
            arr[-1, :, :] = value
        if self.is_boundary["y_lower"]:
            arr[:, 0, :] = value
        if self.is_boundary["y_upper"]:
            arr[:, -1, :] = value
        if self.is_boundary["x_lower"]:
            arr[:, :, 0] = value
        if self.is_boundary["x_upper"]:
            arr[:, :, -1] = value

    def fill_source_term(self, f: np.ndarray):
        """Fill source term for sinusoidal test problem.

        f = 3pi^2 sin(pi*x) sin(pi*y) sin(pi*z)
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

        Z, Y, X = np.meshgrid(z_phys, y_phys, x_phys, indexing="ij")

        f[1:-1, 1:-1, 1:-1] = (
            3 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)
        )

    def compute_exact_solution(self, u: np.ndarray):
        """Fill exact solution for validation.

        u = sin(pi*x) sin(pi*y) sin(pi*z)
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

        Z, Y, X = np.meshgrid(z_phys, y_phys, x_phys, indexing="ij")

        u[1:-1, 1:-1, 1:-1] = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)

    def compute_l2_error(self, u: np.ndarray) -> float:
        """Compute L2 error against exact solution."""
        u_exact = self.allocate()
        self.compute_exact_solution(u_exact)

        local_sq_error = np.sum((u[1:-1, 1:-1, 1:-1] - u_exact[1:-1, 1:-1, 1:-1]) ** 2)
        global_sq_error = self.cart_comm.allreduce(local_sq_error, op=MPI.SUM)

        return float(np.sqrt(self.h**3 * global_sq_error))

    def interior_slice(self):
        """Return slice tuple for interior points."""
        return (slice(1, -1), slice(1, -1), slice(1, -1))

    def get_halo_size_bytes(self) -> int:
        """Calculate total bytes transferred per halo exchange."""
        nz, ny, nx = self.local_shape
        bytes_per_element = 8  # float64

        total_bytes = 0

        # Z-faces
        if self.neighbors["z_lower"] is not None:
            total_bytes += ny * nx * bytes_per_element * 2
        if self.neighbors["z_upper"] is not None:
            total_bytes += ny * nx * bytes_per_element * 2

        # Y-faces
        if self.neighbors["y_lower"] is not None:
            total_bytes += nz * nx * bytes_per_element * 2
        if self.neighbors["y_upper"] is not None:
            total_bytes += nz * nx * bytes_per_element * 2

        # X-faces
        if self.neighbors["x_lower"] is not None:
            total_bytes += nz * ny * bytes_per_element * 2
        if self.neighbors["x_upper"] is not None:
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

    def coarsen(self) -> "DistributedGrid":
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
            halo_exchange=self.halo_exchange_type,
        )

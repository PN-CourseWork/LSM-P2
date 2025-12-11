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

from ..datastructures import LocalParams
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
        comm: MPI.Comm = MPI.COMM_WORLD,
        strategy: str = "sliced",
        halo_exchange: str = "numpy",
    ):
        self.N = N
        self.comm = comm
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

    # Boundary face slices: direction -> array index tuple
    _BOUNDARY_SLICES = {
        "z_lower": (0, slice(None), slice(None)),
        "z_upper": (-1, slice(None), slice(None)),
        "y_lower": (slice(None), 0, slice(None)),
        "y_upper": (slice(None), -1, slice(None)),
        "x_lower": (slice(None), slice(None), 0),
        "x_upper": (slice(None), slice(None), -1),
    }

    def apply_boundary_conditions(self, arr: np.ndarray, value: float = 0.0):
        """Apply Dirichlet boundary conditions at physical boundaries."""
        for direction, is_bc in self.is_boundary.items():
            if is_bc:
                arr[self._BOUNDARY_SLICES[direction]] = value

    def get_rank_info(self) -> LocalParams:
        """Get topology info for this rank (for MLflow artifact)."""
        import os

        # Get CPU affinity (cores this rank can run on)
        try:
            cpu_ids = sorted(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            cpu_ids = None  # Not available on all platforms (e.g., macOS)

        return LocalParams(
            rank=self.rank,
            hostname=MPI.Get_processor_name(),
            cart_coords=tuple(self.cart_comm.Get_coords(self.rank)),
            neighbors=self.neighbors.copy(),
            local_shape=self.local_shape,
            global_start=self.global_start,
            global_end=self.global_end,
            cpu_ids=cpu_ids,
        )

    def _get_physical_coords(self):
        """Compute physical coordinate meshgrid for local interior."""
        gs = self.global_start
        nz, ny, nx = self.local_shape

        z_phys = -1.0 + np.arange(gs[0], gs[0] + nz) * self.h
        y_phys = -1.0 + np.arange(gs[1], gs[1] + ny) * self.h
        x_phys = -1.0 + np.arange(gs[2], gs[2] + nx) * self.h

        return np.meshgrid(z_phys, y_phys, x_phys, indexing="ij")

    def fill_source_term(self, f: np.ndarray):
        """Fill source term: f = 3pi^2 sin(pi*x) sin(pi*y) sin(pi*z)."""
        Z, Y, X = self._get_physical_coords()
        f[1:-1, 1:-1, 1:-1] = (
            3 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)
        )

    def compute_exact_solution(self, u: np.ndarray):
        """Fill exact solution: u = sin(pi*x) sin(pi*y) sin(pi*z)."""
        Z, Y, X = self._get_physical_coords()
        u[1:-1, 1:-1, 1:-1] = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)

    def compute_l2_error(self, u: np.ndarray) -> float:
        """Compute L2 error against exact solution."""
        u_exact = self.allocate()
        self.compute_exact_solution(u_exact)

        local_sq_error = np.sum((u[1:-1, 1:-1, 1:-1] - u_exact[1:-1, 1:-1, 1:-1]) ** 2)
        global_sq_error = np.empty(1)
        self.cart_comm.Allreduce(
            np.array([local_sq_error]), global_sq_error, op=MPI.SUM
        )

        return float(np.sqrt(self.h**3 * global_sq_error[0]))

    def get_halo_size_bytes(self) -> int:
        """Calculate total bytes transferred per halo exchange."""
        nz, ny, nx = self.local_shape
        # Face sizes for each axis: z-face=(ny*nx), y-face=(nz*nx), x-face=(nz*ny)
        face_sizes = [ny * nx, nz * nx, nz * ny]
        axes = ["z", "y", "x"]

        total = 0
        for axis, face_size in zip(axes, face_sizes):
            for side in ["lower", "upper"]:
                if self.neighbors.get(f"{axis}_{side}") is not None:
                    total += face_size * 8 * 2  # float64, send+recv
        return total

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

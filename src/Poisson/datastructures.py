"""Data structures for solver configuration and results.

All datastructures defined here are local to each rank. In an MPI context,
each rank instantiates its own copy of these structures with rank-specific data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


# ============================================================================
# Kernel
# ============================================================================


@dataclass
class KernelParams:
    """Kernel configuration parameters.

    Note: N is the LOCAL grid size (after domain decomposition + halo zones for MPI).
    For standalone usage, N is the full problem size.
    """

    N: int  # Local grid size (including halo zones for MPI)
    omega: float
    tolerance: float = 1e-10
    max_iter: int = 100000
    numba_threads: int | None = None  # None for NumPy

    # Derived values (computed in __post_init__)
    h: float = field(init=False)

    def __post_init__(self):
        """Compute derived values after initialization."""
        self.h = 2.0 / (self.N - 1)


@dataclass
class KernelMetrics:
    """Final convergence metrics (updated during kernel execution)."""

    converged: bool = False
    iterations: int = 0
    final_residual: float | None = None
    total_compute_time: float = 0.0


@dataclass
class KernelSeries:
    """Per-iteration tracking arrays.

    The kernel automatically populates residuals and compute_times during step().
    Physical errors can be optionally appended by the caller for validation.
    """

    residuals: list[float] = field(default_factory=list)
    compute_times: list[float] = field(default_factory=list)
    physical_errors: list[float] | None = None


# ============================================================================
# Solver - Global (identical across ranks, or rank 0 only)
# ============================================================================


@dataclass
class GlobalMetrics:
    """Final convergence metrics (computed/stored on rank 0 only)."""

    iterations: int = 0
    converged: bool = False
    final_error: float | None = None
    wall_time: float | None = None
    # Timing breakdown (sum across all iterations)
    total_compute_time: float | None = None
    total_halo_time: float | None = None
    total_mpi_comm_time: float | None = None
    # Performance metrics
    mlups: float | None = None  # Million Lattice Updates per Second
    bandwidth_gb_s: float | None = None  # Memory bandwidth in GB/s


# ============================================================================
# Solver - Local (each rank has different values)
# ============================================================================


@dataclass
class LocalSeries:
    """Per-iteration/operation timing arrays (each rank).

    Each rank accumulates its own timing data for each iteration/operation.
    Rank 0 additionally stores residual history.

    For Jacobi: one entry per iteration
    For Multigrid: one entry per smoothing operation (reveals grid hierarchy)

    Note: With non-blocking Iallreduce, MPI time overlaps with halo exchange,
    so we only track compute and halo times. "Other" overhead can be computed
    as: wall_time - sum(compute_times) - sum(halo_exchange_times)
    """

    compute_times: list[float] = field(default_factory=list)
    halo_exchange_times: list[float] = field(default_factory=list)
    residual_history: list[float] = field(default_factory=list)
    # For multigrid: track which level each operation is on (0=finest)
    level_indices: list[int] = field(default_factory=list)


# ============================================================================
# Multigrid
# ============================================================================


@dataclass
class GridLevel:
    """One level in the multigrid hierarchy.

    Each level has its own grid size, arrays, and smoothing kernel.
    The grid member is only set for MPI solvers (DistributedGrid).
    """

    level: int
    N: int
    h: float
    u: np.ndarray
    u_temp: np.ndarray
    f: np.ndarray
    r: np.ndarray
    kernel: object
    grid: object = None  # DistributedGrid for MPI, None for sequential


# ============================================================================
# MPI Geometry
# ============================================================================


@dataclass
class RankGeometry:
    """Geometry information for a single MPI rank.

    Attributes
    ----------
    rank : int
        MPI rank number.
    local_shape : tuple[int, int, int]
        Local domain shape (interior points owned by this rank).
    halo_shape : tuple[int, int, int]
        Shape including halo zones (local_shape + 2 in each dimension).
    global_start : tuple[int, int, int]
        Global indices of owned region start (inclusive).
    global_end : tuple[int, int, int]
        Global indices of owned region end (exclusive).
    neighbors : dict[str, int | None]
        Neighbor ranks for each face. Keys: 'x_lower', 'x_upper',
        'y_lower', 'y_upper', 'z_lower', 'z_upper'.
        Value is None if at physical boundary.
    """

    rank: int
    local_shape: tuple[int, int, int]
    halo_shape: tuple[int, int, int]
    global_start: tuple[int, int, int]
    global_end: tuple[int, int, int]
    neighbors: dict[str, int | None]

"""Data structures for solver configuration and results.

All datastructures defined here are local to each rank. In an MPI context,
each rank instantiates its own copy of these structures with rank-specific data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from mpi4py import MPI


# ============================================================================
# Kernel 
# ============================================================================

@dataclass
class KernelParams:
    """Kernel configuration parameters.

    Note: N is the LOCAL grid size (after domain decomposition + ghost zones for MPI).
    For standalone usage, N is the full problem size.
    """
    N: int  # Local grid size (including ghost zones for MPI)
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
class GlobalParams:
    """Global problem definition (all ranks have identical copy).

    Note: N is the GLOBAL grid size (before domain decomposition).
    The solver internally computes N_local for each rank after decomposition.
    """
    # Global problem parameters
    N: int = 0  # Global grid size (before decomposition)
    omega: float = 0.75
    tolerance: float = 1e-10
    max_iter: int = 100000

    # MPI configuration
    mpi_size: int = 1
    decomposition: str = "none"  # "none", "sliced", "cubic"
    communicator: str = "none"   # "none", "numpy", "custom"

    # Kernel backend selection
    use_numba: bool = False
    numba_threads: int = 4


@dataclass
class GlobalMetrics:
    """Final convergence metrics (computed/stored on rank 0 only)."""
    iterations: int = 0
    converged: bool = False
    final_error: float | None = None


# ============================================================================
# Solver - Local (each rank has different values)
# ============================================================================

@dataclass
class LocalParams:
    """Local rank-specific parameters (computed after decomposition).

    Each rank has its own LocalParams with rank-specific values including
    the kernel configuration for that rank's local domain size.
    """
    N_local: int  # Local grid size including ghost zones

    # Domain coordinates in global grid
    local_start: tuple[int, int, int]  # (i_start, j_start, k_start)
    local_end: tuple[int, int, int]    # (i_end, j_end, k_end)

    # Kernel configuration
    kernel: KernelParams

    # Auto-populated from MPI
    rank: int = field(init=False)

    def __post_init__(self):
        """Auto-populate rank from MPI."""
        self.rank = MPI.COMM_WORLD.Get_rank()


@dataclass
class LocalFields:
    """Local domain arrays with ghost zones (each rank)."""
    u1: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    u2: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    f: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))


@dataclass
class LocalSeries:
    """Per-iteration MPI timing arrays (each rank).

    Each rank accumulates its own timing data for each iteration.
    Rank 0 additionally stores residual history.
    """
    compute_times: list[float] = field(default_factory=list)
    mpi_comm_times: list[float] = field(default_factory=list)
    halo_exchange_times: list[float] = field(default_factory=list)
    residual_history: list[float] = field(default_factory=list)



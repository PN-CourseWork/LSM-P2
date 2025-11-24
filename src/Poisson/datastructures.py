"""Data structures for solver configuration and results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

# ============================================================================
# Local dataclasses -> Each Rank owns these 
# ============================================================================

# ============================================================================
# Kernel 
# ============================================================================

@dataclass
class KernelParameters:
    """Kernel configuration parameters."""
    N: int
    omega: float
    tolerance: float = 1e-10
    max_iter: int = 100000
    num_threads: int = None  # None for NumPy

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
    final_residual: float = None
    total_compute_time: float = 0.0


@dataclass
class KernelTimeseries:
    """Per-iteration tracking arrays.

    The kernel automatically populates residuals and compute_times during step().
    Physical errors can be optionally appended by the caller for validation.
    """
    residuals: list[float] = field(default_factory=list)
    compute_times: list[float] = field(default_factory=list)
    physical_errors: list[float] = field(default_factory=list)  # Optional


# ============================================================================
# Solver (MPI-enabled wrapper around kernel)
# ============================================================================

@dataclass
class SolverTimeseries:
    """Per-iteration MPI timing arrays (all ranks).

    Each rank accumulates timing data for each iteration.
    Rank 0 additionally stores residual history.
    """
    compute_times: list[float] = field(default_factory=list)
    mpi_comm_times: list[float] = field(default_factory=list)
    halo_exchange_times: list[float] = field(default_factory=list)
    residual_history: list[float] = field(default_factory=list)  # Rank 0 only

@dataclass
class SolverFields:
    """Local domain arrays with ghost zones (all ranks)."""
    u1_local: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    u2_local: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    f_local: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))



# ===========================================================================
# Global dataclasses -> Only rank 0 owns these 
# ===========================================================================
@dataclass
class SolverParameters:
    """Solver configuration parameters (all ranks have a copy)."""
    # Problem size
    N: int = 0

    # MPI configuration
    mpi_size: int = 1
    decomposition: str = "none"  # "none", "sliced", "cubic"
    communicator: str = "none"   # "none", "numpy", "custom"

    # Jacobi solver parameters
    omega: float = 0.75
    use_numba: bool = False
    numba_threads: int = 4
    max_iter: int = 100000
    tolerance: float = 1e-10


@dataclass
class SolverMetrics:
    """Final convergence metrics (computed/stored on rank 0 only)."""
    iterations: int = 0
    converged: bool = False
    final_error: float = 0.0 # Optional 



"""Data structures for solver configuration and results."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

def create_grid_3d(N: int, value: float = 0.0, boundary_value: float = 0.0) -> np.ndarray:
    """Create 3D grid with specified interior and boundary values."""
    u = np.full((N, N, N), value, dtype=np.float64)
    u[[0, -1], :, :] = boundary_value
    u[:, [0, -1], :] = boundary_value
    u[:, :, [0, -1]] = boundary_value
    return u


def sinusoidal_exact_solution(N: int) -> np.ndarray:
    """Exact solution: sin(π x)sin(π y)sin(π z) on [-1,1]³."""
    xs, ys, zs = np.ogrid[-1 : 1 : complex(N), -1 : 1 : complex(N), -1 : 1 : complex(N)]
    return np.sin(np.pi * xs) * np.sin(np.pi * ys) * np.sin(np.pi * zs)


def sinusoidal_source_term(N: int) -> np.ndarray:
    """Source term: f = 3π² sin(π x)sin(π y)sin(π z).

    For -∇²u = f with u = sin(πx)sin(πy)sin(πz):
    ∇²u = -π²sin(πx)sin(πy)sin(πz) - π²sin(πx)sin(πy)sin(πz) - π²sin(πx)sin(πy)sin(πz)
    ∇²u = -3π²sin(πx)sin(πy)sin(πz)
    Therefore f = 3π²sin(πx)sin(πy)sin(πz)
    """
    xs, ys, zs = np.ogrid[-1 : 1 : complex(N), -1 : 1 : complex(N), -1 : 1 : complex(N)]
    return 3 * np.pi**2 * np.sin(np.pi * xs) * np.sin(np.pi * ys) * np.sin(np.pi * zs)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    """Runtime configuration (all ranks have a copy)."""
    # Problem size
    N: int = 0

    # MPI configuration
    mpi_size: int = 1
    decomposition: str = "none"  # "none", "sliced", "cubic"
    communicator: str = "none"   # "none", "numpy", "custom"

    # Jacobi solver parameters
    omega: float = 0.75
    use_numba: bool = False
    num_threads: int = 1
    max_iter: int = 100000
    tolerance: float = 1e-10


# ============================================================================
# Local fields (per-rank arrays)
# ============================================================================

@dataclass
class LocalFields:
    """Local domain arrays with ghost zones (all ranks)."""
    # Local domain size (including ghosts)
    u1_local: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    u2_local: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    f_local: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))


# ============================================================================
# Results
# ============================================================================

@dataclass
class Results:
    """Convergence information (computed/stored on rank 0 only)."""
    iterations: int = 0
    converged: bool = False
    final_error: float = 0.0


# ============================================================================
# Timing data (per-rank)
# ============================================================================

@dataclass
class Timeseries:
    """Per-iteration timing arrays (all ranks).

    Each rank accumulates timing data for each iteration.
    Rank 0 additionally stores residual history.
    """
    compute_times: list[float] = field(default_factory=list)
    mpi_comm_times: list[float] = field(default_factory=list)
    halo_exchange_times: list[float] = field(default_factory=list)
    residual_history: list[float] = field(default_factory=list)  # Rank 0 only

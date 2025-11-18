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
    """Source term: f = 2π² sin(π x)sin(π y)sin(π z)."""
    xs, ys, zs = np.ogrid[-1 : 1 : complex(N), -1 : 1 : complex(N), -1 : 1 : complex(N)]
    return 2 * np.pi**2 * np.sin(np.pi * xs) * np.sin(np.pi * ys) * np.sin(np.pi * zs)

@dataclass
class GlobalConfig:
    """Global runtime configuration (same for all ranks)."""
    # Problem
    N: int = 0

    # Specs
    mpi_size: int = 1
    method: str = ""

    # Jacobi Solver
    omega: float = 0.75
    use_numba: bool = False
    num_threads: int = 1
    max_iter: int = 0
    tolerance: float = 0.0


@dataclass
class GlobalFields:
    """Problem definition for the Poisson solver."""
    N: int = 0
    h: float = 0.0

    f: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    u1: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    u2: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    u_exact: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))

    def __post_init__(self):
        self.h = 2.0 / (self.N - 1)
        self.u1 = create_grid_3d(self.N) 
        self.u2 = self.u1.copy()
        self.f = sinusoidal_source_term(self.N)
        self.u_exact = sinusoidal_exact_solution(self.N)

@dataclass
class LocalFields: 
    local_N: int = 0
    u1: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    u2: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))

@dataclass
class GlobalResults:
    """Global solver results (same for all ranks)."""
    # Convergence info
    iterations: int = 0
    converged: bool = False
    final_error: float = 0.0
    # Global timings
    wall_time: float = 0.0
    compute_time: float = 0.0
    mpi_comm_time: float = 0.0
    halo_exchange_time: float = 0.0

@dataclass
class TimeSeriesLocal:    
    """pr rank time series data."""
    compute_times: list[float] = field(default_factory=list)
    mpi_comm_times: list[float] = field(default_factory=list)
    halo_exchange_times: list[float] = field(default_factory=list)

@dataclass 
class TimeSeriesGlobal(TimeSeriesLocal):
    residual_history: list[float] = field(default_factory=list)


@dataclass
class LocalResults:
    """Per-rank performance results."""
    mpi_rank: int = 0
    hostname: str = ""
    wall_time: float = 0.0
    compute_time: float = 0.0
    mpi_comm_time: float = 0.0
    halo_exchange_time: float = 0.0


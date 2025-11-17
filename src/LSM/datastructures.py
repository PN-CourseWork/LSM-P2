"""Data structures for solver configuration and results."""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class SolverConfig:
    """Solver initialization configuration."""
    omega: float = 0.75
    use_numba: bool = True
    verbose: bool = False


@dataclass
class RuntimeConfig:
    """Global runtime configuration (same for all ranks)."""
    N: int = 0
    h: float = 0.0
    method: str = ""
    omega: float = 0.0
    tolerance: float = 0.0
    max_iter: int = 0
    use_numba: bool = False
    num_threads: int = 1
    mpi_size: int = 1
    timestamp: str = ""


@dataclass
class GlobalResults:
    """Global solver results (same for all ranks)."""
    iterations: int = 0
    converged: bool = False
    final_residual: float = 0.0
    final_error: float = 0.0
    wall_time: float = 0.0
    compute_time: float = 0.0
    mpi_comm_time: float = 0.0


@dataclass
class PerRankResults:
    """Per-rank performance results."""
    mpi_rank: int = 0
    hostname: str = ""
    wall_time: float = 0.0
    compute_time: float = 0.0
    mpi_comm_time: float = 0.0

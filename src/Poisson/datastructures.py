"""Data structures for solver configuration and results.

Architecture: 2x2 matrix of Params vs Metrics × Global vs Local

                 Params (input/config)         Metrics (output/results)
                 ─────────────────────         ────────────────────────
Global           GlobalParams                  GlobalMetrics
(same across     N, solver, omega,             wall_time, mlups,
ranks / agg)     n_ranks, strategy...          converged, iterations...

Local            LocalParams                   LocalMetrics
(per-rank)       rank, hostname,               compute_times[],
                 neighbors, local_shape...     halo_times[]...
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# Global (identical across ranks, or aggregated on rank 0)
# ============================================================================


@dataclass
class GlobalParams:
    """Run configuration - validated by Hydra, logged to MLflow as params.

    Immutable configuration set before the run. Identical across all MPI ranks.
    """

    # Required
    N: int

    # Solver
    solver: str = "jacobi"  # "jacobi" | "fmg"
    omega: float = 0.8
    tolerance: float = 1e-6
    max_iter: int = 1000

    # FMG-specific (ignored by Jacobi)
    n_smooth: int = 3
    fmg_post_vcycles: int = 1

    # Parallelization
    n_ranks: int = 1
    strategy: Optional[str] = None  # "sliced" | "cubic"
    communicator: Optional[str] = None  # "numpy" | "custom"

    # Numba
    use_numba: bool = False
    specified_numba_threads: int = 1  # What user requested

    # Experiment tracking
    experiment_name: str = "default"

    # Auto-detected at runtime (not from config)
    environment: str = field(init=False)
    h: float = field(init=False)

    def __post_init__(self):
        """Compute derived values after initialization."""
        self.h = 2.0 / (self.N - 1)
        self.environment = (
            "hpc"
            if os.environ.get("LSB_JOBID") or os.environ.get("SLURM_JOB_ID")
            else "local"
        )

    def to_mlflow(self) -> dict:
        """Convert to MLflow-compatible params dict (bools as int, exclude derived)."""
        exclude = {"h"}  # Derived from N, redundant
        return {
            k: (int(v) if isinstance(v, bool) else v)
            for k, v in self.__dict__.items()
            if k not in exclude
        }


@dataclass
class GlobalMetrics:
    """Aggregated results - logged to MLflow as metrics.

    Final results computed/aggregated on rank 0.
    """

    converged: bool = False
    iterations: int = 0
    final_residual: Optional[float] = None
    final_error: Optional[float] = None  # L2 error vs analytical solution
    final_alg_error: Optional[float] = None  # ||f - Au|| algebraic residual
    wall_time: Optional[float] = None

    # Timing breakdown (sum across all iterations)
    total_compute_time: Optional[float] = None
    total_halo_time: Optional[float] = None

    # Performance metrics
    mlups: Optional[float] = None  # Million Lattice Updates per Second
    bandwidth_gb_s: Optional[float] = None  # Memory bandwidth in GB/s

    # Numba runtime info (what was actually available)
    observed_numba_threads: Optional[int] = None

    def to_mlflow(self) -> dict:
        """Convert to MLflow-compatible dict (no None, bools as int)."""
        return {
            k: (int(v) if isinstance(v, bool) else v)
            for k, v in self.__dict__.items()
            if v is not None
        }


# ============================================================================
# Local (per-rank)
# ============================================================================


@dataclass
class LocalParams:
    """Per-rank geometry - gathered to rank 0, logged as artifact.

    Per-rank topology information.
    """

    rank: int
    hostname: str = ""
    cart_coords: Optional[Tuple[int, int, int]] = None
    neighbors: Dict[str, Optional[int]] = field(default_factory=dict)
    local_shape: Optional[Tuple[int, int, int]] = None
    global_start: Optional[Tuple[int, int, int]] = None
    global_end: Optional[Tuple[int, int, int]] = None
    # CPU binding info for socket/node visualization
    cpu_ids: Optional[List[int]] = None  # Cores this rank can run on


@dataclass
class LocalMetrics:
    """Per-rank timeseries - gathered for topology artifact.

    Per-rank timing data. Accumulated during solve, logged post-solve.
    """

    # Per-rank timing (gathered to artifact for load balancing analysis)
    compute_times: List[float] = field(default_factory=list)
    halo_times: List[float] = field(default_factory=list)

    # Global (rank 0 only - logged as step metrics for convergence charts)
    residual_history: List[float] = field(default_factory=list)

    def clear(self):
        """Clear all timeseries data."""
        self.compute_times.clear()
        self.halo_times.clear()
        self.residual_history.clear()

    def to_mlflow_batch(self) -> list:
        """Convert timeseries to MLflow Metric objects for batch logging."""
        from mlflow.entities import Metric

        return [
            Metric(key=name, value=value, timestamp=0, step=step)
            for name, values in self.__dict__.items()
            for step, value in enumerate(values)
        ]


# ============================================================================
# MPI Grid Geometry
# ============================================================================


@dataclass
class RankGeometry:
    """Per-rank grid geometry information.

    Used by DistributedGrid to describe the local portion of the domain.
    """

    rank: int
    local_shape: Tuple[int, int, int]
    halo_shape: Tuple[int, int, int]
    global_start: Tuple[int, int, int]
    global_end: Tuple[int, int, int]
    neighbors: Dict[str, Optional[int]]


# ============================================================================
# Multigrid-specific
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
# Legacy aliases (for backwards compatibility during migration)
# ============================================================================

# These will be removed after full migration
LocalSeries = LocalMetrics  # Alias for backwards compatibility

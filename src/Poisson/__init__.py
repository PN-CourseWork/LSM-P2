"""MPI Poisson Solver package.

A modular framework for studying parallel performance of 3D Poisson equation
solvers using MPI domain decomposition. Supports pluggable decomposition
strategies (sliced, cubic) and communication methods (NumPy arrays, custom
MPI datatypes).

Solvers
-------
Sequential (no MPI):
- JacobiSolver: Kernel benchmarks and single-process solving
- FMGSolver: Full Multigrid without MPI

Parallel (MPI):
- JacobiMPISolver: Jacobi with domain decomposition
- FMGMPISolver: Full Multigrid with domain decomposition
"""

from pathlib import Path

from .datastructures import (
    GlobalParams,
    GlobalMetrics,
    LocalParams,
    LocalMetrics,
    RankGeometry,
    GridLevel,
)
from .kernels import NumPyKernel, NumbaKernel
from .solvers import (
    JacobiSolver,
    JacobiMPISolver,
    FMGSolver,
    FMGMPISolver,
)
from .mpi import DistributedGrid
from .problems import (
    create_grid_3d,
    sinusoidal_exact_solution,
    sinusoidal_source_term,
    setup_sinusoidal_problem,
)

__all__ = [
    # Data structures
    "GlobalParams",
    "GlobalMetrics",
    "LocalParams",
    "LocalMetrics",
    "RankGeometry",
    "GridLevel",
    # Kernels
    "NumPyKernel",
    "NumbaKernel",
    # Solvers - Sequential
    "JacobiSolver",
    "FMGSolver",
    # Solvers - MPI
    "JacobiMPISolver",
    "FMGMPISolver",
    # Grid
    "DistributedGrid",
    # Problem setup
    "create_grid_3d",
    "sinusoidal_exact_solution",
    "sinusoidal_source_term",
    "setup_sinusoidal_problem",
    # Utilities
    "get_project_root",
]


def get_project_root() -> Path:
    """Get project root directory.

    Works reliably in both standalone scripts and Sphinx-Gallery execution.

    Returns
    -------
    Path
        Project root directory (contains pyproject.toml).
    """
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent

    # Fallback: assume standard src layout
    return Path(__file__).resolve().parent.parent.parent

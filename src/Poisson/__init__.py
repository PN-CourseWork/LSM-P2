"""Large Scale Modeling package."""

from pathlib import Path

from .datastructures import (
    GlobalParams,
    GlobalMetrics,
    LocalParams,
    LocalFields,
    LocalSeries,
    KernelParams,
    KernelMetrics,
    KernelSeries,
)
from .kernels import NumPyKernel, NumbaKernel
from .solver import JacobiPoisson
from .mpi import (
    DomainDecomposition,
    RankInfo,
    NoDecomposition,
    NumpyHaloExchange,
    CustomHaloExchange,
)
from .problems import (
    create_grid_3d,
    sinusoidal_exact_solution,
    sinusoidal_source_term,
    setup_sinusoidal_problem,
)
from .helpers import run_solver

__all__ = [
    # Data structures - Kernel
    "KernelParams",
    "KernelMetrics",
    "KernelSeries",
    # Data structures - Solver
    "GlobalParams",
    "GlobalMetrics",
    "LocalParams",
    "LocalFields",
    "LocalSeries",
    # Kernels
    "NumPyKernel",
    "NumbaKernel",
    # Solver
    "JacobiPoisson",
    # Decomposition (DMDA-style)
    "NoDecomposition",
    "DomainDecomposition",
    "RankInfo",
    # Communicators
    "NumpyHaloExchange",
    "CustomHaloExchange",
    # Problem setup
    "create_grid_3d",
    "sinusoidal_exact_solution",
    "sinusoidal_source_term",
    "setup_sinusoidal_problem",
    # Runner
    "run_solver",
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

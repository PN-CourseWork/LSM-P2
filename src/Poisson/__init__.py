"""Large Scale Modeling package."""

from .datastructures import GlobalConfig, GlobalFields, LocalFields, GlobalResults, LocalResults, TimeSeriesLocal, TimeSeriesGlobal
from .kernels import jacobi_step_numpy, jacobi_step_numba
from .jacobi import JacobiPoisson
from .strategies import (
    NoDecomposition,
    SlicedDecomposition,
    CubicDecomposition,
    CustomMPICommunicator,
    NumpyCommunicator,
)
from .problems import create_grid_3d, sinusoidal_exact_solution, sinusoidal_source_term, setup_sinusoidal_problem

__all__ = [
    # Data structures
    "GlobalConfig",
    "GlobalFields",
    "LocalFields",
    "GlobalResults",
    "LocalResults",
    "TimeSeriesLocal",
    "TimeSeriesGlobal",
    # Kernels
    "jacobi_step_numpy",
    "jacobi_step_numba",
    # Solver
    "JacobiPoisson",
    # Strategies
    "NoDecomposition",
    "SlicedDecomposition",
    "CubicDecomposition",
    "CustomMPICommunicator",
    "NumpyCommunicator",
    # Problem setup
    "create_grid_3d",
    "sinusoidal_exact_solution",
    "sinusoidal_source_term",
    "setup_sinusoidal_problem",
]

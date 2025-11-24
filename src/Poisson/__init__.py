"""Large Scale Modeling package."""

from .datastructures import Config, LocalFields, Results, Timeseries
from .kernels import NumPyKernel, NumbaKernel
from .jacobi import JacobiPoisson
from .strategies import (
    NoDecomposition,
    SlicedDecomposition,
    CubicDecomposition,
    CustomMPICommunicator,
    NumpyCommunicator,
)
from .problems import create_grid_3d, sinusoidal_exact_solution, sinusoidal_source_term, setup_sinusoidal_problem
from .postprocessing import PostProcessor

__all__ = [
    # Data structures
    "Config",
    "LocalFields",
    "Results",
    "Timeseries",
    # Kernels
    "NumPyKernel",
    "NumbaKernel",
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
    # Post-processing
    "PostProcessor",
]

"""MPI domain decomposition and communication.

This package provides:
- DistributedGrid: Unified interface for parallel grids
- CartesianDecomposition: Domain splitting with MPI topology
- HaloExchanger: Strategies for halo exchange (numpy/datatype)
- RankGeometry: Exported from datastructures for convenience
"""

from .grid import DistributedGrid
from .decomposition import CartesianDecomposition
from .halo import HaloExchanger, NumpyHaloExchanger, DatatypeHaloExchanger
from ..datastructures import RankGeometry

__all__ = [
    "DistributedGrid",
    "CartesianDecomposition",
    "HaloExchanger",
    "NumpyHaloExchanger",
    "DatatypeHaloExchanger",
    "RankGeometry",
]

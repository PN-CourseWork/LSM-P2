"""MPI domain decomposition and communication."""

from .decomposition import DomainDecomposition, RankInfo, NoDecomposition
from .communicators import NumpyHaloExchange, CustomHaloExchange

__all__ = [
    "DomainDecomposition",
    "RankInfo",
    "NoDecomposition",
    "NumpyHaloExchange",
    "CustomHaloExchange",
]

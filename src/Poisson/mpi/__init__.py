"""MPI domain decomposition and communication."""

#from .decomposition import DomainDecomposition, RankInfo, NoDecomposition
#from .communicators import NumpyHaloExchange, CustomHaloExchange
from .grid import DistributedGrid, RankGeometry

__all__ = [
    # Legacy decomposition (used by JacobiPoisson)
    #"DomainDecomposition",
    #"RankInfo",
    #"NoDecomposition",
    # Legacy communicators
    #"NumpyHaloExchange",
    #"CustomHaloExchange",
    # Unified grid (used by MultigridPoisson)
    "DistributedGrid",
    "RankGeometry",
]

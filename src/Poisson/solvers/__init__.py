"""Poisson Solvers.

Consistent naming: {Method}Solver for sequential, {Method}MPISolver for parallel.

Sequential (no MPI):
- JacobiSolver: Kernel benchmarks and single-process solving
- FMGSolver: Full Multigrid without MPI

Parallel (MPI):
- JacobiMPISolver: Jacobi with domain decomposition
- FMGMPISolver: Full Multigrid with domain decomposition
"""

from .jacobi import JacobiSolver
from .jacobi_mpi import JacobiMPISolver
from .fmg import FMGSolver
from .fmg_mpi import FMGMPISolver

__all__ = [
    "JacobiSolver",
    "JacobiMPISolver",
    "FMGSolver",
    "FMGMPISolver",
]

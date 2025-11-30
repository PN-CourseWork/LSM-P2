"""Command-line interface utilities.

Provides:
- Argument parser creation for experiments
- Common CLI patterns for numerical solvers
"""

from .args import create_parser, add_common_args, add_mpi_args

__all__ = [
    "create_parser",
    "add_common_args",
    "add_mpi_args",
]

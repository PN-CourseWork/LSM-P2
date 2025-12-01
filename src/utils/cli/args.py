"""Argument parsing utilities for numerical experiments.

Provides flexible argument parser creation with common options
for grid-based solvers and MPI experiments.
"""

from argparse import ArgumentParser
from typing import List, Optional


def create_parser(
    description: str = "Numerical solver experiment",
    methods: Optional[List[str]] = None,
    default_method: Optional[str] = None,
) -> ArgumentParser:
    """Create a base argument parser for experiments.

    Parameters
    ----------
    description : str
        Parser description.
    methods : list of str, optional
        Available solver methods. If provided, adds --method argument.
    default_method : str, optional
        Default method (defaults to first in list).

    Returns
    -------
    ArgumentParser
        Configured argument parser.

    Examples
    --------
    >>> parser = create_parser("Poisson solver", methods=["jacobi", "gauss-seidel"])
    >>> args = parser.parse_args()
    """
    parser = ArgumentParser(description=description)

    if methods:
        if default_method is None:
            default_method = methods[0]
        parser.add_argument(
            "--method",
            choices=methods,
            default=default_method,
            help=f"Solver method (default: {default_method})",
        )

    return parser


def add_common_args(
    parser: ArgumentParser,
    grid_size: int = 20,
    iterations: int = 100,
    tolerance: float = 1e-8,
) -> ArgumentParser:
    """Add common numerical solver arguments.

    Parameters
    ----------
    parser : ArgumentParser
        Parser to add arguments to.
    grid_size : int
        Default grid size.
    iterations : int
        Default max iterations.
    tolerance : float
        Default convergence tolerance.

    Returns
    -------
    ArgumentParser
        The modified parser.
    """
    parser.add_argument(
        "-N",
        type=int,
        default=grid_size,
        help=f"Grid size (divisions per dimension, default: {grid_size})",
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=iterations,
        help=f"Maximum iterations (default: {iterations})",
    )

    parser.add_argument(
        "-v0",
        "--value0",
        type=float,
        default=0.0,
        help="Initial value for the grid (default: 0.0)",
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=tolerance,
        help=f"Convergence tolerance (default: {tolerance})",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename (default: auto-generated)",
    )

    return parser


def add_mpi_args(parser: ArgumentParser) -> ArgumentParser:
    """Add MPI-related arguments.

    Parameters
    ----------
    parser : ArgumentParser
        Parser to add arguments to.

    Returns
    -------
    ArgumentParser
        The modified parser.
    """
    parser.add_argument(
        "--decomposition",
        choices=["sliced", "cubic"],
        default="sliced",
        help="Domain decomposition strategy (default: sliced)",
    )

    parser.add_argument(
        "--comm-method",
        choices=["numpy", "custom"],
        default="numpy",
        help="Halo exchange method (default: numpy)",
    )

    return parser


def add_logging_args(parser: ArgumentParser) -> ArgumentParser:
    """Add logging/experiment tracking arguments.

    Parameters
    ----------
    parser : ArgumentParser
        Parser to add arguments to.

    Returns
    -------
    ArgumentParser
        The modified parser.
    """
    parser.add_argument(
        "--job-name",
        type=str,
        default=None,
        help="Job name for logging",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for log files (default: logs)",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="MLflow experiment name",
    )

    return parser

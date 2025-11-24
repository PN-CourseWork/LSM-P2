"""Data utilities for creating, loading, and saving simulation data.

This module provides utilities for:
- Path management (automatic directory structure mirroring Experiments/)
- Data I/O (loading and saving dataframes)
- Numerical computations (norms and errors)
- DataFrame utilities
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd


# ==============================================================================
# Path Management
# ==============================================================================


def get_repo_root() -> Path:
    """Get repository root directory.

    Returns the repository root by detecting the presence of pyproject.toml.
    Works from any subdirectory of the repository.

    Returns
    -------
    Path
        Absolute path to the repository root

    """
    # Start from this file's location
    current = Path(__file__).resolve().parent

    # Walk up until we find pyproject.toml (marks repo root)
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent

    # Fallback: assume two levels up from utils
    return current.parent.parent


def get_experiment_name(caller_file: Path | str | None = None) -> str:
    """Get experiment name from the calling script's location.

    Extracts the experiment name from the path relative to Experiments/.
    For example:
    - Experiments/sequential/compute.py → "sequential"
    - Experiments/parallel/mpi/compute.py → "parallel/mpi"

    Parameters
    ----------
    caller_file : Path or str, optional
        Path to the calling file. If None, automatically detects the caller.

    Returns
    -------
    str
        Experiment name (relative path from Experiments/)

    Raises
    ------
    ValueError
        If the calling file is not in an Experiments/ subdirectory

    """
    if caller_file is None:
        # Get the calling file from the stack
        # We need to go up 2 frames: this function -> get_data_dir/get_figures_dir -> actual caller
        frame = inspect.currentframe()
        if frame is None or frame.f_back is None or frame.f_back.f_back is None:
            raise RuntimeError("Cannot detect caller file")
        caller_file = Path(frame.f_back.f_back.f_globals["__file__"])
    else:
        caller_file = Path(caller_file)

    caller_file = caller_file.resolve()

    # Find Experiments directory in the path
    experiments_idx = None
    for i, part in enumerate(caller_file.parts):
        if part == "Experiments":
            experiments_idx = i
            break

    if experiments_idx is None:
        raise ValueError(
            f"File {caller_file} is not in an Experiments/ subdirectory. "
            "This utility is designed for scripts in Experiments/*/"
        )

    # Get the path components between Experiments/ and the filename
    # e.g., for Experiments/sequential/compute.py → ["sequential"]
    # or for Experiments/parallel/mpi/compute.py → ["parallel", "mpi"]
    experiment_parts = caller_file.parts[experiments_idx + 1 : -1]

    if not experiment_parts:
        raise ValueError(
            f"File {caller_file} is directly in Experiments/. "
            "Scripts should be in a subdirectory (e.g., Experiments/sequential/)"
        )

    return "/".join(experiment_parts)


def get_data_dir(caller_file: Path | str | None = None, create: bool = True) -> Path:
    """Get data directory for the calling experiment.

    Automatically determines the correct data directory based on the
    calling script's location in Experiments/, mirroring the structure.

    Parameters
    ----------
    caller_file : Path or str, optional
        Path to the calling file. If None, automatically detects the caller.
    create : bool, default True
        Whether to create the directory if it doesn't exist

    Returns
    -------
    Path
        Data directory path (e.g., repo_root/data/sequential/)

    Examples
    --------
    From Experiments/sequential/compute.py:
    >>> data_dir = get_data_dir()  # Returns repo_root/data/sequential/

    From Experiments/parallel/mpi/compute.py:
    >>> data_dir = get_data_dir()  # Returns repo_root/data/parallel/mpi/

    """
    experiment_name = get_experiment_name(caller_file)
    repo_root = get_repo_root()
    data_dir = repo_root / "data" / experiment_name

    if create:
        data_dir.mkdir(parents=True, exist_ok=True)

    return data_dir


def get_figures_dir(caller_file: Path | str | None = None, create: bool = True) -> Path:
    """Get figures directory for the calling experiment.

    Automatically determines the correct figures directory based on the
    calling script's location in Experiments/, mirroring the structure.

    Parameters
    ----------
    caller_file : Path or str, optional
        Path to the calling file. If None, automatically detects the caller.
    create : bool, default True
        Whether to create the directory if it doesn't exist

    Returns
    -------
    Path
        Figures directory path (e.g., repo_root/figures/sequential/)

    Examples
    --------
    From Experiments/sequential/plot.py:
    >>> figures_dir = get_figures_dir()  # Returns repo_root/figures/sequential/

    From Experiments/parallel/mpi/plot.py:
    >>> figures_dir = get_figures_dir()  # Returns repo_root/figures/parallel/mpi/

    """
    experiment_name = get_experiment_name(caller_file)
    repo_root = get_repo_root()
    figures_dir = repo_root / "figures" / experiment_name

    if create:
        figures_dir.mkdir(parents=True, exist_ok=True)

    return figures_dir


def ensure_output_dir(path: Path | str) -> Path:
    """Ensure output directory exists, creating it if necessary.

    Parameters
    ----------
    path : Path or str
        Directory path to create

    Returns
    -------
    Path
        The created/existing directory path

    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path



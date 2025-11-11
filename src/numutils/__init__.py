"""
Numerical Utilities Package
============================

A simple package demonstrating basic numerical operations for University projects.

This package provides:
- Basic linear algebra operations
- Numerical integration methods
- Plotting utilities
"""

import matplotlib.pyplot as plt
from pathlib import Path

from numutils.integration import integrate_trapz, integrate_simpson
from numutils.linalg import norm, solve_linear


def use_style(style_name="ana"):
    """Load the numutils matplotlib style.

    Parameters
    ----------
    style_name : str, default="ana"
        Name of the style file (without .mplstyle extension)

    Raises
    ------
    FileNotFoundError
        If the specified style file does not exist

    Examples
    --------
    >>> import numutils
    >>> numutils.use_style("ana")  # Load the default ana style
    """
    style_path = Path(__file__).parent / "styles" / f"{style_name}.mplstyle"
    if not style_path.exists():
        raise FileNotFoundError(f"Style file not found: {style_path}")
    plt.style.use(str(style_path))

__all__ = [
    "integrate_trapz",
    "integrate_simpson",
    "norm",
    "solve_linear",
    "use_style",
]

__version__ = "0.1.0"

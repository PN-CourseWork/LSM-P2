"""Plotting utilities for spectral method visualizations.

This module provides utilities for:
- Automatic style application (seaborn + custom mplstyle)
- Formatting labels and parameters for plots
- Common plotting helpers

Automatically applies seaborn style and custom utils.mplstyle on import.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


# ==============================================================================
# Style Application
# ==============================================================================


def _apply_styles():
    """Apply seaborn style and custom utils.mplstyle."""
    # Apply seaborn style first
    try:
        plt.style.use("seaborn-v0_8")
    except OSError:
        # Fallback if seaborn style not available
        pass

    # Then apply custom style on top
    style_path = Path(__file__).parent / "utils.mplstyle"
    if style_path.exists():
        plt.style.use(str(style_path))


# Apply styles when module is imported
_apply_styles()


# ==============================================================================
# Formatting Utilities
# ==============================================================================


def format_dt_latex(dt: float | str) -> str:
    """Format a timestep value as LaTeX scientific notation.

    Parameters
    ----------
    dt : float or str
        Timestep value to format. If str and equals '?', returns '?'

    Returns
    -------
    str
        LaTeX-formatted string in the form 'mantissa \\times 10^{exponent}'

    Examples
    --------
    >>> format_dt_latex(0.001)
    '1.00 \\times 10^{-3}'

    """
    if dt == "?":
        return "?"

    dt_str = f"{float(dt):.2e}"
    mantissa, exp = dt_str.split("e")
    exp_int = int(exp)
    return rf"{mantissa} \times 10^{{{exp_int}}}"


def format_parameter_range(
    values: list | tuple,
    name: str,
    latex: bool = True,
) -> str:
    """Format a parameter range for display.

    Parameters
    ----------
    values : list or tuple
        Parameter values (should be sorted)
    name : str
        Parameter name (e.g., 'N', 'L', 'dt')
    latex : bool, default True
        Whether to use LaTeX formatting

    Returns
    -------
    str
        Formatted string

    Examples
    --------
    >>> format_parameter_range([10, 20, 30], 'N')
    '$N \\in [10, 30]$'

    """
    if len(values) == 0:
        return f"{name} = ?"

    if len(values) == 1:
        val = values[0]
        if latex:
            return rf"${name} = {val}$"
        return f"{name} = {val}"

    min_val, max_val = min(values), max(values)

    # Format based on type
    if isinstance(min_val, int) and isinstance(max_val, int):
        range_str = f"[{min_val}, {max_val}]"
    else:
        range_str = f"[{min_val:.1f}, {max_val:.1f}]"

    if latex:
        return rf"${name} \in {range_str}$"
    return f"{name} ∈ {range_str}"


def build_parameter_string(
    params: dict[str, Any],
    separator: str = ", ",
    latex: bool = True,
) -> str:
    """Build a parameter string from a dictionary.

    Parameters
    ----------
    params : dict
        Dictionary of parameter names and values
    separator : str, default ', '
        Separator between parameters
    latex : bool, default True
        Whether to use LaTeX formatting (wraps each param in $ $)

    Returns
    -------
    str
        Formatted parameter string

    Examples
    --------
    >>> build_parameter_string({'N': 100, 'dt': 0.001})
    '$N = 100$, $dt = 1.00 \\times 10^{-3}$'

    """
    parts = []
    for name, value in params.items():
        if isinstance(value, (list, tuple)):
            parts.append(format_parameter_range(value, name, latex=latex))
        else:
            # Handle special formatting for dt
            if "dt" in name.lower() or "Delta t" in name:
                value_str = format_dt_latex(value)
                if latex:
                    parts.append(rf"${name} = {value_str}$")
                else:
                    parts.append(f"{name} = {value_str}")
            else:
                if latex:
                    parts.append(rf"${name} = {value}$")
                else:
                    parts.append(f"{name} = {value}")

    return separator.join(parts)

"""Formatting utilities for scientific plot labels and annotations.

Provides LaTeX-compatible formatting for:
- Scientific notation (e.g., 1.00 × 10⁻³)
- Parameter ranges (e.g., N ∈ [10, 100])
- Parameter strings for titles/legends
"""

from __future__ import annotations

from typing import Any


def format_scientific_latex(value: float | str, precision: int = 2) -> str:
    """Format a value as LaTeX scientific notation.

    Parameters
    ----------
    value : float or str
        Value to format. If str and equals '?', returns '?'
    precision : int, default 2
        Number of decimal places for mantissa

    Returns
    -------
    str
        LaTeX-formatted string in the form 'mantissa \\times 10^{exponent}'

    Examples
    --------
    >>> format_scientific_latex(0.001)
    '1.00 \\times 10^{-3}'
    >>> format_scientific_latex(1.5e-6, precision=1)
    '1.5 \\times 10^{-6}'
    """
    if value == "?":
        return "?"

    value_str = f"{float(value):.{precision}e}"
    mantissa, exp = value_str.split("e")
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
            # Handle special formatting for timestep-like parameters
            if "dt" in name.lower() or "delta" in name.lower():
                value_str = format_scientific_latex(value)
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

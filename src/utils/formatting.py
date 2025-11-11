"""Formatting utilities for plot labels and parameters."""

from __future__ import annotations

from typing import Any

import pandas as pd


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

    """
    if dt == "?":
        return "?"

    dt_str = f"{float(dt):.2e}"
    mantissa, exp = dt_str.split("e")
    exp_int = int(exp)
    return rf"{mantissa} \times 10^{{{exp_int}}}"


def extract_metadata(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    row_idx: int = 0,
) -> dict[str, Any]:
    """Extract metadata from a DataFrame.

    Assumes metadata columns have constant values across rows.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing metadata
    cols : list of str, optional
        List of column names to extract. If None, extracts all columns.
    row_idx : int, default 0
        Row index to extract from (typically 0 for constant columns)

    Returns
    -------
    dict
        Dictionary mapping column names to values

    """
    if cols is None:
        cols = df.columns.tolist()

    return {col: df[col].iloc[row_idx] for col in cols if col in df.columns}


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
        Formatted string'

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

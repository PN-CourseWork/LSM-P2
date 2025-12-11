"""Plotting utilities for scientific visualizations.

This module provides:
- Automatic style application (seaborn + custom scientific style)
- LaTeX formatting utilities for labels and parameters

Automatically applies styles on import:
    from utils import plotting  # Styles applied!

Or import specific utilities:
    from utils.plotting import format_scientific_latex
"""

from .styles import apply_styles
from .formatters import (
    format_scientific_latex,
    format_parameter_range,
    build_parameter_string,
)

# Apply styles when module is imported
apply_styles()

__all__ = [
    "apply_styles",
    "format_scientific_latex",
    "format_parameter_range",
    "build_parameter_string",
]

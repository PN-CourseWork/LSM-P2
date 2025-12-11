"""Style application for matplotlib plots.

Provides consistent styling for scientific publications using
seaborn as a base with custom overrides for LaTeX integration.
"""

from pathlib import Path

import matplotlib.pyplot as plt


def apply_styles(base_style: str = "seaborn-v0_8") -> None:
    """Apply seaborn style and custom scientific style.

    Parameters
    ----------
    base_style : str, default "seaborn-v0_8"
        Base matplotlib style to use. Falls back silently if unavailable.
    """
    # Apply base style first
    try:
        plt.style.use(base_style)
    except OSError:
        pass

    # Then apply custom style on top
    style_path = Path(__file__).parent / "scientific.mplstyle"
    if style_path.exists():
        plt.style.use(str(style_path))


def get_style_path() -> Path:
    """Return path to the scientific.mplstyle file."""
    return Path(__file__).parent / "scientific.mplstyle"

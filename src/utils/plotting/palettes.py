"""Color palettes for scientific visualizations.

Provides colorblind-friendly palettes optimized for:
- Line plots (categorical)
- Heatmaps (sequential/diverging)
- Scaling/convergence plots
"""

from typing import List

# Colorblind-friendly categorical palette (Paul Tol's vibrant)
CATEGORICAL = [
    "#0077BB",  # Blue
    "#EE7733",  # Orange
    "#009988",  # Teal
    "#CC3311",  # Red
    "#33BBEE",  # Cyan
    "#EE3377",  # Magenta
    "#BBBBBB",  # Grey
]

# For sequential data (light to dark blue)
SEQUENTIAL_BLUE = [
    "#F7FBFF",
    "#DEEBF7",
    "#C6DBEF",
    "#9ECAE1",
    "#6BAED6",
    "#4292C6",
    "#2171B5",
    "#084594",
]

# For diverging data (blue-white-red)
DIVERGING = [
    "#2166AC",
    "#4393C3",
    "#92C5DE",
    "#D1E5F0",
    "#F7F7F7",
    "#FDDBC7",
    "#F4A582",
    "#D6604D",
    "#B2182B",
]

# For scaling plots (strong/weak scaling)
SCALING = {
    "ideal": "#888888",      # Grey dashed line for ideal scaling
    "strong": "#0077BB",     # Blue for strong scaling
    "weak": "#EE7733",       # Orange for weak scaling
    "efficiency": "#009988", # Teal for parallel efficiency
}

# For convergence plots
CONVERGENCE = {
    "error": "#CC3311",      # Red for error
    "reference": "#888888",  # Grey for reference slopes
    "first_order": "#33BBEE",
    "second_order": "#0077BB",
}


def get_categorical(n: int = None) -> List[str]:
    """Get categorical palette colors.

    Parameters
    ----------
    n : int, optional
        Number of colors needed. If None, returns full palette.

    Returns
    -------
    list of str
        Hex color codes
    """
    if n is None:
        return CATEGORICAL.copy()
    return (CATEGORICAL * ((n // len(CATEGORICAL)) + 1))[:n]


def get_sequential(name: str = "blue") -> List[str]:
    """Get sequential palette.

    Parameters
    ----------
    name : str
        Palette name. Currently only "blue" supported.

    Returns
    -------
    list of str
        Hex color codes from light to dark
    """
    palettes = {
        "blue": SEQUENTIAL_BLUE,
    }
    return palettes.get(name, SEQUENTIAL_BLUE).copy()


def get_diverging() -> List[str]:
    """Get diverging palette (blue-white-red)."""
    return DIVERGING.copy()

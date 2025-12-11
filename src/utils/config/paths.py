"""Path configuration utilities."""

from pathlib import Path


def get_repo_root() -> Path:
    """Find the project root directory (where pyproject.toml is).

    Returns
    -------
    Path
        Path to repository root.
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: assume src/utils/config structure
    return current.parent.parent.parent.parent

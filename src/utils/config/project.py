"""Project configuration utilities.

Provides functions for finding the repository root and loading
project configuration from YAML files.
"""

from pathlib import Path
from typing import Any, Dict

import yaml


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


def load_project_config(config_name: str = "project_config.yaml") -> Dict[str, Any]:
    """Load project configuration from YAML.

    Parameters
    ----------
    config_name : str
        Name of the config file in repo root.

    Returns
    -------
    dict
        Parsed configuration, or empty dict if not found.
    """
    repo_root = get_repo_root()
    config_path = repo_root / config_name
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def get_config_section(section: str) -> Dict[str, Any]:
    """Get a specific section from project config.

    Parameters
    ----------
    section : str
        Section name (e.g., "hpc", "mlflow").

    Returns
    -------
    dict
        Section contents, or empty dict if not found.
    """
    config = load_project_config()
    return config.get(section, {})

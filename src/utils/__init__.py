"""Utility modules for project management and visualization.

Submodules:
- plotting: Scientific plot styling, formatters, palettes
- runners: Script discovery and execution
- hpc: HPC job generation and submission
- config: Project configuration and cleanup
- mlflow: MLflow artifact handling and log uploading
- cli: Command-line argument parsing

Import examples:
    from utils import plotting     # Auto-applies scientific styles
    from utils import runners      # Script execution
    from utils import hpc          # HPC job management
    from utils import mlflow       # MLflow utilities
    from utils.config import get_repo_root, load_project_config
"""

import warnings

# Suppress MLflow FutureWarning about filesystem backend deprecation
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")

from . import plotting, runners, hpc, config, mlflow, cli

# Re-export common config functions for convenience
from .config import get_repo_root

__all__ = [
    "plotting",
    "runners",
    "hpc",
    "config",
    "mlflow",
    "cli",
    "get_repo_root",
]

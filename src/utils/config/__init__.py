"""Project configuration and cleanup utilities.

Provides:
- Repository root detection
- Project config loading from YAML
- Cleanup utilities for generated files
"""

from .project import (
    get_repo_root,
    load_project_config,
    get_config_section,
)
from .clean import (
    clean_all,
    clean_directories,
    clean_files,
    clean_patterns,
    clean_data_directory,
    clean_experiment_outputs,
)

__all__ = [
    # Project config
    "get_repo_root",
    "load_project_config",
    "get_config_section",
    # Cleanup
    "clean_all",
    "clean_directories",
    "clean_files",
    "clean_patterns",
    "clean_data_directory",
    "clean_experiment_outputs",
]

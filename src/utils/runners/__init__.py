"""Script execution utilities.

Provides functions for discovering and running scripts:
- discover_scripts: Find scripts by pattern in Experiments/
- run_scripts_parallel: Run scripts concurrently
- run_scripts_sequential: Run scripts one at a time
"""

from .scripts import (
    discover_scripts,
    run_scripts_parallel,
    run_scripts_sequential,
    run_plot_scripts,
    run_compute_scripts,
    copy_to_report,
)

__all__ = [
    "discover_scripts",
    "run_scripts_parallel",
    "run_scripts_sequential",
    "run_plot_scripts",
    "run_compute_scripts",
    "copy_to_report",
]

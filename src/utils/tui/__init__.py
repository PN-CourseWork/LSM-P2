"""Terminal User Interface utilities.

Provides:
- TuiApp: Full-featured blessed-based TUI
- main_menu: Simple menu-based interface
- handle_args: CLI batch mode handler
"""

from .tui import TuiApp, run_tui
from .main import main_menu
from .batch import handle_args

__all__ = [
    "TuiApp",
    "run_tui",
    "main_menu",
    "handle_args",
]

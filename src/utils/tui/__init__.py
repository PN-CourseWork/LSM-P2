"""Terminal User Interface utilities.

Provides:
- TuiApp: Full-featured blessed-based TUI
- TuiRunner: Action runner with output capture and TUI widgets
- main_menu: Simple menu-based interface
- handle_args: CLI batch mode handler
"""

from .tui import TuiApp, run_tui
from .runner import TuiRunner, SelectOption, ActionResult
from .main import main_menu
from .batch import handle_args

__all__ = [
    "TuiApp",
    "run_tui",
    "TuiRunner",
    "SelectOption",
    "ActionResult",
    "main_menu",
    "handle_args",
]

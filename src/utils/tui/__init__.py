"""Terminal User Interface utilities.

Provides:
- ProjectTUI: Textual-based TUI (recommended)
- TuiApp: Legacy blessed-based TUI
- TuiRunner: Action runner with output capture and TUI widgets
- main_menu: Simple menu-based interface
- handle_args: CLI batch mode handler
"""

from .app import ProjectTUI, run_tui
from .tui import TuiApp
from .runner import TuiRunner, SelectOption, ActionResult
from .main import main_menu
from .batch import handle_args

__all__ = [
    "ProjectTUI",
    "TuiApp",
    "run_tui",
    "TuiRunner",
    "SelectOption",
    "ActionResult",
    "main_menu",
    "handle_args",
]

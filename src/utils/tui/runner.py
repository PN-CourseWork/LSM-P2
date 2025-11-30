"""TUI-integrated action runner with output capture and selection."""

import io
import sys
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Any
from blessed import Terminal


@dataclass
class ActionResult:
    """Result of running an action."""
    success: bool
    output: List[str]
    error: Optional[str] = None


@dataclass
class SelectOption:
    """Option for TUI selection."""
    label: str
    value: Any
    description: str = ""


class TuiRunner:
    """Run actions within the TUI, capturing output and handling selections."""

    def __init__(self, term: Terminal):
        self.term = term
        self.output_lines: List[str] = []
        self.scroll_offset = 0

    def capture_output(self, func: Callable, *args, **kwargs) -> ActionResult:
        """Run a function and capture its stdout/stderr.

        Parameters
        ----------
        func : Callable
            Function to run
        *args, **kwargs
            Arguments to pass to function

        Returns
        -------
        ActionResult
            Result with captured output
        """
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                result = func(*args, **kwargs)

            output = stdout_capture.getvalue().splitlines()
            errors = stderr_capture.getvalue()

            return ActionResult(
                success=True,
                output=output,
                error=errors if errors else None,
            )
        except Exception as e:
            return ActionResult(
                success=False,
                output=stdout_capture.getvalue().splitlines(),
                error=str(e),
            )

    def select(
        self,
        title: str,
        options: List[SelectOption],
        allow_multiple: bool = False,
    ) -> Optional[Any]:
        """Show a selection menu within the TUI.

        Parameters
        ----------
        title : str
            Title for the selection
        options : List[SelectOption]
            Options to choose from
        allow_multiple : bool
            Allow selecting multiple options

        Returns
        -------
        Selected value(s) or None if cancelled
        """
        term = self.term
        selected_idx = 0
        selected_items = set() if allow_multiple else None

        while True:
            # Draw selection UI
            print(term.home + term.clear)

            # Header
            print(term.black_on_white(f" {title} ".center(term.width)))
            print()

            # Options
            for i, opt in enumerate(options):
                y = 3 + i
                if allow_multiple:
                    checkbox = "[x]" if i in selected_items else "[ ]"
                    prefix = f" {checkbox} "
                else:
                    prefix = " > " if i == selected_idx else "   "

                style = term.bold if i == selected_idx else term.normal
                print(term.move_xy(0, y) + f"{prefix}{style}{opt.label}{term.normal}")

                if opt.description and i == selected_idx:
                    print(term.move_xy(4, y + 1) + term.bright_black(opt.description))

            # Help
            if allow_multiple:
                help_text = " [j/k] Navigate | [Space] Toggle | [Enter] Confirm | [q] Cancel"
            else:
                help_text = " [j/k] Navigate | [Enter] Select | [q] Cancel"

            print(
                term.move_xy(0, term.height - 1)
                + term.black_on_white(f"{help_text:<{term.width}}")
            )

            key = term.inkey()

            if key.lower() == "q" or key.name == "KEY_ESCAPE":
                return None
            elif key.name == "KEY_UP" or key == "k":
                selected_idx = max(0, selected_idx - 1)
            elif key.name == "KEY_DOWN" or key == "j":
                selected_idx = min(len(options) - 1, selected_idx + 1)
            elif key == " " and allow_multiple:
                if selected_idx in selected_items:
                    selected_items.remove(selected_idx)
                else:
                    selected_items.add(selected_idx)
            elif key.name == "KEY_ENTER":
                if allow_multiple:
                    return [options[i].value for i in sorted(selected_items)]
                else:
                    return options[selected_idx].value

    def confirm(self, message: str, default: bool = False) -> bool:
        """Show a confirmation dialog.

        Parameters
        ----------
        message : str
            Confirmation message
        default : bool
            Default selection

        Returns
        -------
        bool
            True if confirmed
        """
        term = self.term
        selected = default

        while True:
            print(term.home + term.clear)
            print()
            print(f"  {message}")
            print()

            yes_style = term.reverse if selected else term.normal
            no_style = term.reverse if not selected else term.normal

            print(f"    {yes_style} Yes {term.normal}    {no_style} No {term.normal}")
            print()
            print(term.bright_black("  [h/l] or [y/n] to select, [Enter] to confirm"))

            key = term.inkey()

            if key.lower() == "y":
                return True
            elif key.lower() == "n":
                return False
            elif key.name == "KEY_LEFT" or key == "h":
                selected = True
            elif key.name == "KEY_RIGHT" or key == "l":
                selected = False
            elif key.name == "KEY_ENTER":
                return selected
            elif key.name == "KEY_ESCAPE" or key.lower() == "q":
                return False

    def show_output(
        self,
        title: str,
        lines: List[str],
        wait_for_key: bool = True,
    ) -> None:
        """Display output in a scrollable view.

        Parameters
        ----------
        title : str
            Title for the output view
        lines : List[str]
            Lines to display
        wait_for_key : bool
            Wait for keypress before returning
        """
        term = self.term
        scroll = 0
        max_scroll = max(0, len(lines) - (term.height - 4))

        while True:
            print(term.home + term.clear)

            # Header
            print(term.black_on_white(f" {title} ".center(term.width)))
            print(term.cyan("─" * term.width))

            # Content
            visible_height = term.height - 4
            visible = lines[scroll : scroll + visible_height]

            for i, line in enumerate(visible):
                y = 2 + i
                disp = line[: term.width - 1] if len(line) >= term.width else line
                print(term.move_xy(0, y) + disp)

            # Scroll indicator
            if max_scroll > 0:
                pct = int((scroll / max_scroll) * 100) if max_scroll else 0
                indicator = f"[{pct}%] {scroll + 1}-{min(scroll + visible_height, len(lines))}/{len(lines)}"
                print(
                    term.move_xy(term.width - len(indicator) - 1, term.height - 2)
                    + term.bright_black(indicator)
                )

            # Help
            help_text = " [j/k] Scroll | [g/G] Top/Bottom | [Enter/q] Close"
            print(
                term.move_xy(0, term.height - 1)
                + term.black_on_white(f"{help_text:<{term.width}}")
            )

            if not wait_for_key:
                return

            key = term.inkey()

            if key.name == "KEY_ENTER" or key.lower() == "q" or key.name == "KEY_ESCAPE":
                return
            elif key.name == "KEY_UP" or key == "k":
                scroll = max(0, scroll - 1)
            elif key.name == "KEY_DOWN" or key == "j":
                scroll = min(max_scroll, scroll + 1)
            elif key == "g":
                scroll = 0
            elif key == "G":
                scroll = max_scroll
            elif key.name == "KEY_PGUP":
                scroll = max(0, scroll - visible_height)
            elif key.name == "KEY_PGDOWN":
                scroll = min(max_scroll, scroll + visible_height)

    def input(self, prompt: str, default: str = "") -> Optional[str]:
        """Get text input from user.

        Parameters
        ----------
        prompt : str
            Input prompt
        default : str
            Default value

        Returns
        -------
        str or None if cancelled
        """
        term = self.term
        value = default
        cursor_pos = len(value)

        while True:
            print(term.home + term.clear)
            print()
            print(f"  {prompt}")
            print()
            print(f"  > {value}")
            print()
            print(term.bright_black("  [Enter] Confirm | [Esc] Cancel"))

            # Show cursor position
            print(term.move_xy(4 + cursor_pos, 4), end="", flush=True)

            key = term.inkey()

            if key.name == "KEY_ENTER":
                return value
            elif key.name == "KEY_ESCAPE":
                return None
            elif key.name == "KEY_BACKSPACE":
                if cursor_pos > 0:
                    value = value[: cursor_pos - 1] + value[cursor_pos:]
                    cursor_pos -= 1
            elif key.name == "KEY_DELETE":
                if cursor_pos < len(value):
                    value = value[:cursor_pos] + value[cursor_pos + 1 :]
            elif key.name == "KEY_LEFT":
                cursor_pos = max(0, cursor_pos - 1)
            elif key.name == "KEY_RIGHT":
                cursor_pos = min(len(value), cursor_pos + 1)
            elif key.name == "KEY_HOME":
                cursor_pos = 0
            elif key.name == "KEY_END":
                cursor_pos = len(value)
            elif len(key) == 1 and key.isprintable():
                value = value[:cursor_pos] + key + value[cursor_pos:]
                cursor_pos += 1

        return value

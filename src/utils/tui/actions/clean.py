import sys

import questionary

from src.utils.config import clean_all
from src.utils.tui.io import getch, clear_screen
from src.utils.tui.styles import get_custom_style

def run_clean_menu():
    """Clean & Maintenance Submenu"""
    while True:
        clear_screen()
        print("\n--- Clean & Maintenance ---")
        print("  [c] Clean All Generated Files")
        print("  -----------------------")
        print("  [b] Back")
        print("  [q] Quit")
        print("\nSelect an action: ", end="", flush=True)

        key = getch().lower()

        if key == 'c':
            if questionary.confirm(
                "Are you sure you want to delete all generated files (docs, caches, data, figures)?",
                default=False,
                style=get_custom_style()
            ).ask():
                clean_all()
            input("Press Enter to continue...")
        elif key == 'q':
            clear_screen()
            sys.exit(0)
        elif key == 'b':
            break

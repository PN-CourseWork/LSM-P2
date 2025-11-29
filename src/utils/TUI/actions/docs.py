import sys
from src.utils import manage
from src.utils.TUI.io import getch, clear_screen

def run_docs_menu():
    """Documentation Submenu"""
    clear_screen()
    print("\n--- Documentation ---")
    print("  Building documentation...")
    manage.build_docs()
    input("Press Enter to continue...")

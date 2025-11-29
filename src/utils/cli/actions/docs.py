import sys
from src.utils import manage
from src.utils.cli.io import getch

def run_docs_menu():
    """Documentation Submenu"""
    print("\n--- Documentation ---")
    print("  Building documentation...")
    manage.build_docs()
    input("Press Enter to continue...")
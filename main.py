#!/usr/bin/env python3
"""Main entry point for project management."""

import sys
from src.utils.cli import main as interactive_menu
from src.utils.cli import batch

def main():
    """Run interactive menu if no args, else run batch mode."""
    if len(sys.argv) > 1:
        batch.handle_args()
    else:
        interactive_menu.main_menu()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Main entry point for project management."""

import sys
from src.utils.tui import handle_args, run_tui

def main():
    """Run interactive TUI if no args, else run batch mode."""
    if len(sys.argv) > 1:
        handle_args()
    else:
        run_tui()

if __name__ == "__main__":
    main()
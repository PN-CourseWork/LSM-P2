"""Simple menu-based TUI."""

import sys

from utils.tui.actions import hpc, run, data, docs, clean
from utils.tui.io import getch, clear_screen


def main_menu():
    """Main CLI Entry Point"""

    while True:
        clear_screen()
        print("\n=== LSM Project 2 Manager ===")
        print("  [h] HPC & Scheduling")
        print("  [e] Execution & Processing")
        print("  [d] Data & Results")
        print("  [o] Documentation")
        print("  [c] Clean & Maintenance")
        print("  -----------------------")
        print("  [q] Quit")
        print("\nSelect an action: ", end="", flush=True)

        key = getch().lower()

        if key == "h":
            hpc.run_hpc_menu()
        elif key == "e":
            run.run_execution_menu()
        elif key == "d":
            data.run_data_menu()
        elif key == "o":
            docs.run_docs_menu()
        elif key == "c":
            clean.run_clean_menu()
        elif key == "q":
            clear_screen()
            print("Exiting.")
            sys.exit(0)


if __name__ == "__main__":
    main_menu()

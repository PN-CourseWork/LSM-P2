import sys
from utils import runners
from utils.tui.io import getch, clear_screen

def run_execution_menu():
    """Execution Submenu"""
    while True:
        clear_screen()
        print("\n--- Execution & Processing ---")
        print("  [c] Run Compute Scripts (Sequential)")
        print("  [p] Run Plot Scripts (Parallel)")
        print("  [r] Copy Plots to Report (Overleaf)")
        print("  -----------------------")
        print("  [b] Back")
        print("  [q] Quit")
        print("\nSelect an action: ", end="", flush=True)

        key = getch().lower()

        if key == 'c':
            runners.run_compute_scripts()
            input("Press Enter to continue...")
        elif key == 'p':
            runners.run_plot_scripts()
            input("Press Enter to continue...")
        elif key == 'r':
            runners.copy_to_report()
            input("Press Enter to continue...")
        elif key == 'q':
            clear_screen()
            sys.exit(0)
        elif key == 'b':
            break

import sys
from src.utils import manage
from src.utils.TUI.io import getch, clear_screen

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
            manage.run_compute_scripts()
            input("Press Enter to continue...")
        elif key == 'p':
            manage.run_plot_scripts()
            input("Press Enter to continue...")
        elif key == 'r':
            manage.copy_plots()
            input("Press Enter to continue...")
        elif key == 'q':
            clear_screen()
            sys.exit(0)
        elif key == 'b':
            break
import sys
from src.utils import manage
from src.utils.cli.io import getch

def run_execution_menu():
    """Execution Submenu"""
    while True:
        print("\n--- Execution & Processing ---")
        print("  [c] Run Compute Scripts (Sequential)")
        print("  [p] Run Plot Scripts (Parallel)")
        print("  [r] Copy Plots to Report (Overleaf)")
        print("  -----------------------")
        print("  [b] Back")
        print("  [q] Quit")
        print("\nSelect an action: ", end="", flush=True)
        
        key = getch().lower()
        print(key)
        
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
            sys.exit(0)
        elif key == 'b':
            break

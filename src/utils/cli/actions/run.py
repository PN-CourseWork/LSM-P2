import questionary
import sys
from src.utils import manage
from src.utils.cli.styles import get_custom_style

def run_execution_menu():
    """Execution Submenu"""
    while True:
        action = questionary.select(
            "Execution & Processing:",
            choices=[
                "Run Compute Scripts (Sequential)",
                "Run Plot Scripts (Parallel)",
                "Copy Plots to Report (Overleaf)",
                questionary.Separator(),
                "[b] Back",
                "[q] Quit"
            ],
            style=get_custom_style(),
            use_shortcuts=True
        ).ask()
        
        if action == "Run Compute Scripts (Sequential)":
            manage.run_compute_scripts()
            input("Press Enter to continue...")
        elif action == "Run Plot Scripts (Parallel)":
            manage.run_plot_scripts()
            input("Press Enter to continue...")
        elif action == "Copy Plots to Report (Overleaf)":
            manage.copy_plots()
            input("Press Enter to continue...")
        elif action == "[q] Quit":
            sys.exit(0)
        else:
            break
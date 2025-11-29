import sys
import questionary
from src.utils.cli.styles import get_custom_style
from src.utils.cli.actions import hpc, run, data, docs, clean

def main_menu():
    """Main CLI Entry Point"""
    
    while True:
        print("\n=== LSM Project 2 Manager ===")
        
        action = questionary.select(
            "Select an action:",
            choices=[
                "HPC & Scheduling",
                "Execution & Processing",
                "Data & Results",
                "Documentation",
                "Clean & Maintenance",
                questionary.Separator(),
                "[q] Quit"
            ],
            style=get_custom_style(),
            use_shortcuts=True
        ).ask()
        
        if action == "HPC & Scheduling":
            hpc.run_hpc_menu()
        elif action == "Execution & Processing":
            run.run_execution_menu()
        elif action == "Data & Results":
            data.run_data_menu()
        elif action == "Documentation":
            docs.run_docs_menu()
        elif action == "Clean & Maintenance":
            clean.run_clean_menu()
        elif action == "[q] Quit" or action is None:
            print("Exiting.")
            sys.exit(0)

if __name__ == "__main__":
    main_menu()
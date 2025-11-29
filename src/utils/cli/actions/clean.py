import questionary
import sys
from src.utils import manage
from src.utils.cli.styles import get_custom_style

def run_clean_menu():
    """Clean & Maintenance Submenu"""
    while True:
        action = questionary.select(
            "Clean & Maintenance:",
            choices=[
                "Clean All Generated Files",
                questionary.Separator(),
                "[b] Back",
                "[q] Quit"
            ],
            style=get_custom_style(),
            use_shortcuts=True
        ).ask()
        
        if action == "Clean All Generated Files":
            if questionary.confirm(
                "Are you sure you want to delete all generated files (docs, caches, data, figures)?", 
                default=False, 
                style=get_custom_style()
            ).ask():
                manage.clean_all()
            input("Press Enter to continue...")
        elif action == "[q] Quit":
            sys.exit(0)
        else:
            break

import questionary
from src.utils import manage
from src.utils.cli.styles import get_custom_style

def run_docs_menu():
    """Documentation Submenu"""
    manage.build_docs()
    input("Press Enter to continue...")

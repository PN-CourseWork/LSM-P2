import sys

from utils import mlflow
from utils.config import load_project_config, get_repo_root
from utils.tui.io import getch, clear_screen

def run_data_menu():
    """Data & Results Submenu"""
    while True:
        clear_screen()
        print("\n--- Data & Results ---")
        print("  [f] Fetch MLflow Artifacts")
        print("  -----------------------")
        print("  [b] Back")
        print("  [q] Quit")
        print("\nSelect an action: ", end="", flush=True)

        key = getch().lower()

        if key == 'f':
            config = load_project_config()
            mlflow_conf = config.get("mlflow", {})
            repo_root = get_repo_root()

            mlflow.setup_mlflow_tracking()

            output_dir = repo_root / mlflow_conf.get("download_dir", "data")
            mlflow.fetch_project_artifacts(output_dir)
            input("Press Enter to continue...")
        elif key == 'q':
            clear_screen()
            sys.exit(0)
        elif key == 'b':
            break

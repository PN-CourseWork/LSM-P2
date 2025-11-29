import questionary
import sys
from src.utils import manage, mlflow_io
from src.utils.cli.styles import get_custom_style

def run_data_menu():
    """Data & Results Submenu"""
    while True:
        action = questionary.select(
            "Data & Results:",
            choices=[
                "Fetch MLflow Artifacts",
                questionary.Separator(),
                "[b] Back",
                "[q] Quit"
            ],
            style=get_custom_style(),
            use_shortcuts=True
        ).ask()
        
        if action == "Fetch MLflow Artifacts":
            config = manage.load_project_config()
            mlflow_conf = config.get("mlflow", {})
            repo_root = manage.get_repo_root()
            
            mlflow_io.setup_mlflow_auth(mlflow_conf.get("tracking_uri"))
            
            output_dir = repo_root / mlflow_conf.get("download_dir", "data/downloaded")
            experiments = mlflow_conf.get("experiments", [])
            
            if not experiments:
                print("  (No experiments configured in project_config.yaml)")
            else:
                mlflow_io.fetch_project_artifacts(experiments, output_dir)
            input("Press Enter to continue...")
        elif action == "[q] Quit":
            sys.exit(0)
        else:
            break
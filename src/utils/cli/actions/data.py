import sys
from src.utils import manage, mlflow_io
from src.utils.cli.io import getch

def run_data_menu():
    """Data & Results Submenu"""
    while True:
        print("\n--- Data & Results ---")
        print("  [f] Fetch MLflow Artifacts")
        print("  -----------------------")
        print("  [b] Back")
        print("  [q] Quit")
        print("\nSelect an action: ", end="", flush=True)
        
        key = getch().lower()
        print(key)
        
        if key == 'f':
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
        elif key == 'q':
            sys.exit(0)
        elif key == 'b':
            break

"""
Interactive Setup for MLflow Environment
========================================

This script interactively prompts the user to create a .env file
for configuring MLflow tracking, either for local development or
for use with a remote Databricks workspace.
"""

import os
from pathlib import Path
import questionary
from dotenv import set_key

# Get project root to find the .env file
project_root = Path(__file__).parent.resolve()
env_path = project_root / ".env"


def main():
    """Main function to run the interactive setup."""
    print("--- MLflow Environment Setup ---")

    if env_path.exists():
        overwrite = questionary.confirm(
            f"An .env file already exists at {env_path}.\nDo you want to overwrite it?",
            default=False
        ).ask()
        if not overwrite:
            print("Setup cancelled.")
            return

    setup_type = questionary.select(
        "Choose your MLflow setup type:",
        choices=[
            "Local: For development and testing without a remote server.",
            "Databricks: To connect to a remote Databricks tracking server.",
        ],
    ).ask()

    if setup_type is None:
        print("Setup cancelled.")
        return

    # Clear the .env file before writing new values
    open(env_path, 'w').close()

    if "Databricks" in setup_type:
        print("\nPlease provide your Databricks credentials.")
        
        host = questionary.text(
            "Enter your Databricks host (e.g., https://dbc-xxxx.cloud.databricks.com):"
        ).ask()
        if not host:
            print("Setup cancelled. Host is required.")
            return

        token = questionary.password("Enter your Databricks token (PAT):").ask()
        if not token:
            print("Setup cancelled. Token is required.")
            return
            
        experiment_name = questionary.text(
            "Enter a default MLflow experiment name:",
            default="/Shared/LSM-Project-2/Poisson-Scaling"
        ).ask()

        set_key(env_path, "DATABRICKS_HOST", host)
        set_key(env_path, "DATABRICKS_TOKEN", token)
        set_key(env_path, "MLFLOW_EXPERIMENT_NAME", experiment_name)
        
        print("\n✅ Success! Your .env file has been configured for Databricks.")
        print("   MLflow will now use your remote tracking server.")

    elif "Local" in setup_type:
        # For local, we create an .env file that points to the local defaults
        set_key(env_path, "MLFLOW_TRACKING_URI", "file:./mlruns")
        set_key(env_path, "MLFLOW_EXPERIMENT_NAME", "Poisson Solver Local")
        set_key(env_path, "DATABRICKS_HOST", "")
        set_key(env_path, "DATABRICKS_TOKEN", "")

        print("\n✅ Success! Your .env file has been configured for local tracking.")
        print("   MLflow will now store runs in a local 'mlruns' directory.")
        print("   To use a remote server later, run this script again.")

if __name__ == "__main__":
    main()
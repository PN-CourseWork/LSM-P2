"""Helper script to upload LSF logs to MLflow after a job finishes.

This script is designed to run after the main computation, regardless of success or failure.
It relies on a .runid file created by the main script to know which MLflow run to attach logs to.
"""

import argparse
import os
import time
from pathlib import Path
import mlflow

def main():
    parser = argparse.ArgumentParser(description="Upload LSF logs to MLflow")
    parser.add_argument("--job-name", type=str, required=True, help="LSF Job Name")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory containing logs")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    job_name = args.job_name
    run_id_file = log_dir / f"{job_name}.runid"
    out_log = log_dir / f"{job_name}.out"
    err_log = log_dir / f"{job_name}.err"

    # Give the filesystem a moment to sync logs if the job just crashed
    time.sleep(2)

    run_id = None
    if run_id_file.exists():
        try:
            with open(run_id_file, "r") as f:
                run_id = f.read().strip()
            print(f"Found Run ID: {run_id}")
        except Exception as e:
            print(f"Error reading run ID file: {e}")

    # Logic to handle upload
    try:
        active_run = None
        
        if run_id:
            # Resume existing run
            active_run = mlflow.start_run(run_id=run_id, log_system_metrics=False)
        else:
            # Create new run for startup failure
            print("Run ID file not found. Creating new run for startup failure.")
            experiment_name = "HPC-Poisson-Scaling" # Default for HPC context
            
            # Ensure experiment exists
            if mlflow.get_experiment_by_name(experiment_name) is None:
                try:
                    mlflow.create_experiment(name=experiment_name)
                except:
                    pass # concurrent creation might fail, ignore
            
            mlflow.set_experiment(experiment_name)
            active_run = mlflow.start_run(run_name=f"{job_name} (Startup Failure)")
            mlflow.set_tag("status", "startup_failure")

        with active_run:
            if out_log.exists():
                print(f"Uploading stdout: {out_log}")
                mlflow.log_artifact(str(out_log), artifact_path="logs")
            else:
                print(f"Warning: stdout log not found at {out_log}")

            if err_log.exists():
                print(f"Uploading stderr: {err_log}")
                mlflow.log_artifact(str(err_log), artifact_path="logs")
            else:
                print(f"Warning: stderr log not found at {err_log}")
                
            print("Log upload complete.")
            
    except Exception as e:
        print(f"Failed to upload logs to MLflow: {e}")

if __name__ == "__main__":
    main()

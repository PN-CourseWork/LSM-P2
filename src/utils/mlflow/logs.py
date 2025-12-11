"""Log uploading utilities for HPC jobs.

Uploads stdout/stderr logs to MLflow after job completion.
"""

import time
from pathlib import Path
from typing import Optional

import mlflow


def upload_logs(
    job_name: str,
    log_dir: str = "logs",
    experiment_name: str = "HPC-Experiment",
    run_id: Optional[str] = None,
) -> bool:
    """Upload job logs to MLflow.

    Parameters
    ----------
    job_name : str
        Job name (used to find log files).
    log_dir : str
        Directory containing log files.
    experiment_name : str
        MLflow experiment name (used if creating new run).
    run_id : str, optional
        Existing run ID to attach logs to. If not provided,
        attempts to read from {log_dir}/{job_name}.runid file.

    Returns
    -------
    bool
        True if upload succeeded.
    """
    log_path = Path(log_dir)
    run_id_file = log_path / f"{job_name}.runid"
    out_log = log_path / f"{job_name}.out"
    err_log = log_path / f"{job_name}.err"

    # Give filesystem time to sync if job just finished
    time.sleep(2)

    # Try to get run_id from file if not provided
    if run_id is None and run_id_file.exists():
        try:
            with open(run_id_file, "r") as f:
                run_id = f.read().strip()
            print(f"Found Run ID: {run_id}")
        except Exception as e:
            print(f"Error reading run ID file: {e}")

    try:
        active_run = None

        if run_id:
            active_run = mlflow.start_run(run_id=run_id, log_system_metrics=False)
        else:
            print("Run ID file not found. Creating new run for startup failure.")

            if mlflow.get_experiment_by_name(experiment_name) is None:
                try:
                    mlflow.create_experiment(name=experiment_name)
                except Exception:
                    pass  # concurrent creation might fail

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
            return True

    except Exception as e:
        print(f"Failed to upload logs to MLflow: {e}")
        return False


def main():
    """CLI entry point for log uploading."""
    import argparse

    parser = argparse.ArgumentParser(description="Upload job logs to MLflow")
    parser.add_argument("--job-name", type=str, required=True, help="Job name")
    parser.add_argument(
        "--log-dir", type=str, default="logs", help="Directory containing logs"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="HPC-Experiment",
        help="MLflow experiment name",
    )
    args = parser.parse_args()

    upload_logs(
        job_name=args.job_name,
        log_dir=args.log_dir,
        experiment_name=args.experiment_name,
    )


if __name__ == "__main__":
    main()

"""
Script to upload LSF logs to MLflow.

This script scans a directory for *.runid files (created by the experiment runners),
reads the MLflow Run ID, finds the corresponding .out and .err files, and uploads
them as artifacts to that run.

Processed files are moved to a 'processed' subdirectory.

Usage:
    uv run python src/utils/mlflow/upload_logs.py --log-dir logs/lsf
"""

import argparse
import shutil
from pathlib import Path
import mlflow
from utils.mlflow.io import setup_mlflow_tracking


def upload_logs(log_dir: Path, dry_run: bool = False):
    """
    Uploads LSF logs to MLflow.

    Parameters
    ----------
    log_dir : Path
        Directory containing .runid, .out, and .err files.
    dry_run : bool
        If True, does not perform upload or move files.
    """
    log_dir = Path(log_dir)
    if not log_dir.exists():
        print(f"Log directory not found: {log_dir}")
        return

    processed_dir = log_dir / "processed"
    if not dry_run:
        processed_dir.mkdir(exist_ok=True)

    # Find all runid files
    runid_files = list(log_dir.glob("*.runid"))
    if not runid_files:
        print(f"No .runid files found in {log_dir}")
        return

    print(f"Found {len(runid_files)} pending log sets in {log_dir}...")
    client = mlflow.tracking.MlflowClient()

    for runid_file in runid_files:
        job_name = runid_file.stem
        out_file = log_dir / f"{job_name}.out"
        err_file = log_dir / f"{job_name}.err"

        # Check if log files exist
        if not out_file.exists() or not err_file.exists():
            print(f"  [SKIP] {job_name}: Missing .out or .err file.")
            continue

        try:
            # Read Run ID
            with open(runid_file, "r") as f:
                run_id = f.read().strip()

            print(f"  [PROCESSING] {job_name} (Run ID: {run_id})")

            if not dry_run:
                # Verify run exists
                try:
                    client.get_run(run_id)  # Just check if run exists
                except Exception:
                    print(f"    ! Run {run_id} not found in MLflow. Skipping.")
                    continue

                # Upload artifacts
                print(f"    Uploading {out_file.name}...")
                client.log_artifact(run_id, str(out_file))

                print(f"    Uploading {err_file.name}...")
                client.log_artifact(run_id, str(err_file))

                # Move to processed
                shutil.move(str(runid_file), str(processed_dir / runid_file.name))
                shutil.move(str(out_file), str(processed_dir / out_file.name))
                shutil.move(str(err_file), str(processed_dir / err_file.name))
                print("    Done.")
            else:
                print(f"    (Dry Run) Would upload {out_file.name} and {err_file.name}")

        except Exception as e:
            print(f"    ! ERROR processing {job_name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload LSF logs to MLflow")
    parser.add_argument(
        "--log-dir", type=str, default="logs/lsf", help="Directory to scan"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Simulate without changes"
    )
    args = parser.parse_args()

    setup_mlflow_tracking()
    upload_logs(args.log_dir, args.dry_run)

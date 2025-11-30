"""MLflow I/O utilities for experiment tracking, and fetching runs and artifacts.

This module provides helpers for:
- Setting up MLflow tracking (local or Databricks) via environment variables.
- Orchestrating MLflow runs (context manager for parent/nested runs).
- Logging parameters, metrics, and artifacts.
- Retrieving experiment data from MLflow.
"""

import os
import shutil
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional
import argparse
from contextlib import contextmanager

import mlflow
import pandas as pd

# Conditional import for type hint to avoid circular dependency
try:
    from Poisson.solver import JacobiPoisson
except ImportError:
    JacobiPoisson = None


def setup_mlflow_tracking():
    """
    Configures MLflow tracking via Databricks.
    Attempts a non-interactive login assuming credentials have been
    previously configured via 'uv run python setup_mlflow.py'.
    """
    try:
        mlflow.login(backend="databricks", interactive=False)
        mlflow.set_tracking_uri("databricks")
        print("INFO: Connected to Databricks MLflow tracking.")
    except Exception as e:
        raise RuntimeError(
            "MLflow setup failed. Please run 'uv run python setup_mlflow.py' first."
        ) from e

def get_mlflow_client() -> mlflow.tracking.MlflowClient:
    """Get an MLflow tracking client."""
    return mlflow.tracking.MlflowClient()


@contextmanager
def start_mlflow_run_context(
    experiment_name: str,
    parent_run_name: str,
    child_run_name: str,
    project_prefix: str = "/Shared/LSM-PoissonMPI",
    args: Optional[argparse.Namespace] = None,
):
    """
    Context manager to start a nested MLflow run.
    """
    if mlflow.get_tracking_uri() == "databricks" and not experiment_name.startswith("/"):
        experiment_name = f"{project_prefix}/{experiment_name}"

    mlflow.set_experiment(experiment_name)
    print(f"INFO: Using MLflow experiment: {experiment_name}")

    client = get_mlflow_client()
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        exp_id = client.create_experiment(experiment_name)
        exp = client.get_experiment(exp_id)

    parent_runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.mlflow.runName = '{parent_run_name}' AND tags.is_parent = 'true'",
        max_results=1,
    )
    parent_run_id = parent_runs[0].info.run_id if parent_runs else None

    with mlflow.start_run(run_id=parent_run_id, run_name=parent_run_name, tags={"is_parent": "true"}) as parent_mlflow_run:
        with mlflow.start_run(run_name=child_run_name, nested=True) as child_mlflow_run:
            print(f"INFO: Started MLflow run '{child_mlflow_run.info.run_name}' ({child_mlflow_run.info.run_id})")
            if args and args.job_name:
                try:
                    from Poisson import get_project_root
                    project_root = get_project_root()
                    log_path = project_root / args.log_dir
                    log_path.mkdir(parents=True, exist_ok=True)
                    run_id_file = log_path / f"{args.job_name}.runid"
                    with open(run_id_file, "w") as f:
                        f.write(child_mlflow_run.info.run_id)
                    print(f"  ✓ Saved run ID to {run_id_file}")
                except Exception as e:
                    print(f"  ✗ WARNING: Could not save run ID to file: {e}")
            yield child_mlflow_run


def log_parameters(params: dict):
    """Log a dictionary of parameters to the active MLflow run."""
    mlflow.log_params(params)


def log_metrics_dict(metrics: dict):
    """Log a dictionary of metrics to the active MLflow run, filtering out None values."""
    filtered_metrics = {k: v for k, v in metrics.items() if v is not None}
    mlflow.log_metrics(filtered_metrics)


def log_timeseries_metrics(timeseries_data: object):
    """Log time series data as step-based metrics to the active MLflow run."""
    if not mlflow.active_run():
        return
    client = get_mlflow_client()
    run_id = mlflow.active_run().info.run_id
    timestamp = int(time.time() * 1000)
    metrics_to_log = []
    ts_dict = asdict(timeseries_data)
    for name, values in ts_dict.items():
        if values:
            for step, value in enumerate(values):
                try:
                    val = float(value)
                    metrics_to_log.append(mlflow.entities.Metric(name, val, timestamp, step))
                except (ValueError, TypeError):
                    continue
    if metrics_to_log:
        for i in range(0, len(metrics_to_log), 1000):
            chunk = metrics_to_log[i: i + 1000]
            client.log_batch(run_id=run_id, metrics=chunk, synchronous=True)
        print(f"  ✓ Logged {len(metrics_to_log)} time-series metrics.")


def log_artifact_file(filepath: Path):
    """Log a file as an artifact to the active MLflow run."""
    if filepath.exists():
        mlflow.log_artifact(str(filepath))
        print(f"  ✓ Logged artifact: {filepath.name}")
    else:
        print(f"  ✗ WARNING: Artifact file not found at {filepath}")


def fetch_project_artifacts(output_dir: Path):
    """
    Dynamically discovers and fetches artifacts from all experiments under the
    project's configured Databricks directory.
    """
    from utils.config import load_project_config # Lazy import to avoid circular dependency
    
    mlflow_conf = load_project_config().get("mlflow", {})
    databricks_dir = mlflow_conf.get("databricks_dir")
    
    if not databricks_dir:
        print("ERROR: 'mlflow.databricks_dir' not set in project_config.yaml")
        return

    print(f"INFO: Searching for all experiments under project path: '{databricks_dir}'...")
    all_experiments = mlflow.search_experiments()
    
    # Filter experiments that are part of the project directory on Databricks
    # Handles both /Shared/ and user-specific paths
    project_experiments = [
        exp for exp in all_experiments if databricks_dir in exp.name
    ]
    
    if not project_experiments:
        print("INFO: No project-related experiments found.")
        return

    print(f"INFO: Found {len(project_experiments)} project experiments.")
    output_dir = Path(output_dir)

    for exp in project_experiments:
        # Create a local directory that mirrors the MLflow experiment structure
        # e.g., /Shared/LSM-PoissonMPI/Experiment-01-Kernels -> data/01-kernels
        # We need a mapping from experiment name to local dir name
        local_dir_name = exp.name.split("/")[-1].replace("Experiment-", "").lower()
        exp_dir = output_dir / local_dir_name
        
        print(f"\nProcessing Experiment: {exp.name}")
        
        try:
            paths = download_artifacts(exp.name, exp_dir)
            if paths:
                print(f"  ✓ Downloaded {len(paths)} files to {exp_dir}")
            else:
                print("  - No artifacts found to download.")
        except Exception as e:
            print(f"  ✗ Failed to fetch artifacts for experiment '{exp.name}': {e}")

def load_runs(
    experiment: str,
    converged_only: bool = True,
    exclude_parent_runs: bool = True,
) -> pd.DataFrame:
    """Load runs from an MLflow experiment."""
    # Build filter string
    filters = []
    if converged_only:
        filters.append("metrics.converged = 1")

    filter_string = " and ".join(filters) if filters else ""

    # Fetch runs
    df = mlflow.search_runs(
        experiment_names=[experiment],
        filter_string=filter_string,
        order_by=["start_time DESC"],
    )

    # Filter out parent runs in pandas (MLflow filter doesn't handle None well)
    if exclude_parent_runs and "tags.is_parent" in df.columns:
        df = df[df["tags.is_parent"] != "true"]

    return df

def download_artifacts(
    experiment_name: str,
    output_dir: Path,
    exclude_parent_runs: bool = True,
) -> List[Path]:
    """
    Download all artifacts from all non-parent runs in a given experiment.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = get_mlflow_client()
    exp = client.get_experiment_by_name(experiment_name)
    if not exp:
        print(f"  - Experiment '{experiment_name}' not found.")
        return []

    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    if not runs:
        return []

    # Filter out parent runs in Python (Databricks doesn't support OR in filters)
    if exclude_parent_runs:
        runs = [r for r in runs if r.data.tags.get("is_parent") != "true"]

    downloaded = []
    for run in runs:
        run_id = run.info.run_id
        artifacts = client.list_artifacts(run_id)

        for artifact in artifacts:
            local_path = client.download_artifacts(run_id, artifact.path, str(output_dir))
            downloaded.append(Path(local_path))

    return downloaded

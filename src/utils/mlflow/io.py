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
from dotenv import load_dotenv

# Conditional import for type hint to avoid circular dependency
try:
    from Poisson.solver import JacobiPoisson
except ImportError:
    JacobiPoisson = None


def setup_mlflow_tracking():
    """Configure MLflow tracking URI based on environment variables.

    Loads .env and default.env, then sets the tracking URI.
    If DATABRICKS_HOST is set, it configures tracking for Databricks.
    Otherwise, it falls back to MLFLOW_TRACKING_URI.
    """
    # Load environment variables from .env files
    # load_dotenv(override=True) will load .env and override system vars if present
    # load_dotenv() will not override existing system vars
    load_dotenv(dotenv_path=".env", override=True)  # User-specific .env first
    load_dotenv(dotenv_path="default.env")  # Defaults for missing vars

    if os.getenv("DATABRICKS_HOST"):
        print("INFO: DATABRICKS_HOST found, setting tracking URI to 'databricks'.")
        mlflow.set_tracking_uri("databricks")
    else:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
        print(f"INFO: Using MLflow tracking URI: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)


def get_mlflow_client() -> mlflow.tracking.MlflowClient:
    """Get an MLflow tracking client."""
    return mlflow.tracking.MlflowClient()


@contextmanager
def start_mlflow_run_context(
    experiment_name: str,
    parent_run_name: str,
    child_run_name: str,
    project_prefix: str = "/Shared/LSM-Project-2",
    args: Optional[argparse.Namespace] = None,
):
    """
    Context manager to start a nested MLflow run.
    It finds or creates a parent run and then starts a child run nested within it.
    If 'args' are provided, it can also handle saving the run ID for external loggers.

    Parameters
    ----------
    experiment_name : str
        The name of the MLflow experiment.
    parent_run_name : str
        The name for the parent run.
    child_run_name : str
        The name for the nested child run.
    project_prefix : str, optional
        Prefix for Databricks experiment paths if not absolute.
    args : argparse.Namespace, optional
        Command-line arguments, used to find 'job_name' and 'log_dir'
        for saving the run ID.
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

    # Search for existing parent run
    parent_runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.mlflow.runName = '{parent_run_name}' AND tags.is_parent = 'true'",
        max_results=1,
    )

    parent_run_id = parent_runs[0].info.run_id if parent_runs else None

    with mlflow.start_run(run_id=parent_run_id, run_name=parent_run_name, tags={"is_parent": "true"}) as parent_mlflow_run:
        with mlflow.start_run(run_name=child_run_name, nested=True) as child_mlflow_run:
            print(f"INFO: Started MLflow run '{child_mlflow_run.info.run_name}' ({child_mlflow_run.info.run_id})")
            
            # Save Run ID for external log uploader if job_name is provided
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
            chunk = metrics_to_log[i : i + 1000]
            client.log_batch(run_id=run_id, metrics=chunk, synchronous=True)
        print(f"  ✓ Logged {len(metrics_to_log)} time-series metrics.")

def log_artifact_file(filepath: Path):
    """Log a file as an artifact to the active MLflow run."""
    if filepath.exists():
        mlflow.log_artifact(str(filepath))
        print(f"  ✓ Logged artifact: {filepath.name}")
    else:
        print(f"  ✗ WARNING: Artifact file not found at {filepath}")

def fetch_project_artifacts(experiments: List[str], output_dir: Path) -> None:
    """Fetch artifacts for a list of experiments.

    Parameters
    ----------
    experiments : list of str
        List of experiment names.
    output_dir : Path
        Base directory to save artifacts.
    """
    output_dir = Path(output_dir)

    for exp in experiments:
        print(f"\nProcessing Experiment: {exp}")
        exp_dir = output_dir / exp
        exp_dir.mkdir(parents=True, exist_ok=True)

        try:
            paths = download_artifacts_with_naming(exp, exp_dir)
            print(f"  ✓ Downloaded {len(paths)} files to {exp_dir}")
        except Exception as e:
            print(f"  ✗ Failed to fetch experiment '{exp}': {e}")

def load_runs(
    experiment: str,
    converged_only: bool = True,
    exclude_parent_runs: bool = True,
    experiment_prefix: str = "/Shared/LSM-Project-2",
) -> pd.DataFrame:
    """Load runs from an MLflow experiment.

    Parameters
    ----------
    experiment : str
        Experiment name (e.g., "HPC-Solver" or full path "/Shared/Project/HPC-Solver").
    converged_only : bool, default True
        Only return runs where metrics.converged = 1.
    exclude_parent_runs : bool, default True
        Exclude parent runs (nested run containers).
    experiment_prefix : str
        Prefix to prepend if experiment doesn't start with "/".

    Returns
    -------
    pd.DataFrame
        DataFrame with run info, parameters (params.*), and metrics (metrics.*).
    """
    # Normalize experiment name
    if not experiment.startswith("/"):
        experiment = f"{experiment_prefix}/{experiment}"

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
    experiment: str,
    output_dir: Path,
    converged_only: bool = True,
    artifact_filter: Optional[List[str]] = None,
) -> List[Path]:
    """Download artifacts from MLflow runs to local directory.

    Parameters
    ----------
    experiment : str
        Experiment name (e.g., "HPC-Solver").
    output_dir : Path
        Directory to save artifacts.
    converged_only : bool, default True
        Only download from converged runs.
    artifact_filter : list of str, optional
        Only download artifacts matching these extensions (e.g., [".h5", ".png"])

    Returns
    -------
    list of Path
        Paths to downloaded files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_runs(experiment, converged_only=converged_only)
    if df.empty:
        print(f"No runs found for {experiment}")
        return []

    client = get_mlflow_client()
    downloaded = []

    for _, row in df.iterrows():
        run_id = row["run_id"]
        artifacts = client.list_artifacts(run_id)

        for artifact in artifacts:
            if artifact_filter:
                if not any(artifact.path.endswith(ext) for ext in artifact_filter):
                    continue

            local_path = client.download_artifacts(run_id, artifact.path, str(output_dir))
            downloaded.append(Path(local_path))
            print(f"  Downloaded: {artifact.path}")

    return downloaded

def download_artifacts_with_naming(
    experiment: str,
    output_dir: Path,
    converged_only: bool = True,
    name_template: str = "{prefix}_N{n}_{filename}",
    prefix: str = "Result",
) -> List[Path]:
    """Download HDF5 artifacts with standardized naming.

    Parameters
    ----------
    experiment : str
        Experiment name.
    output_dir : Path
        Directory to save artifacts.
    converged_only : bool, default True
        Only download from converged runs.
    name_template : str
        Template for output filenames. Available placeholders: {prefix}, {n}, {filename}.
    prefix : str
        Prefix for output filenames.

    Returns
    -------
    list of Path
        Paths to downloaded files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_runs(experiment, converged_only=converged_only)
    if df.empty:
        print(f"No runs found for {experiment}")
        return []

    client = get_mlflow_client()
    downloaded = []

    for _, row in df.iterrows():
        run_id = row["run_id"]

        # Extract parameters for naming
        n_param = next((col for col in row.index if col.endswith(".n") or col.endswith(".N")), None)
        n = row.get(n_param, "unknown") if n_param else "unknown"


        artifacts = client.list_artifacts(run_id)

        for artifact in artifacts:
            if artifact.path.endswith(".h5"):
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = client.download_artifacts(run_id, artifact.path, tmpdir)

                    filename = artifact.path.split("/")[-1]
                    new_name = name_template.format(
                        prefix=prefix, n=n, filename=filename
                    )
                    final_path = output_dir / new_name

                    shutil.copy(tmp_path, final_path)
                    downloaded.append(final_path)
                    print(f"  {artifact.path} -> {new_name}")

    return downloaded
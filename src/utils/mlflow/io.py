"""MLflow I/O utilities for experiment tracking, and fetching runs and artifacts.

This module provides helpers for:
- Setting up MLflow tracking (local or Databricks) via environment variables.
- Orchestrating MLflow runs (context manager for parent/nested runs).
- Logging parameters, metrics, and artifacts.
- Retrieving experiment data from MLflow.
"""

import os
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


def setup_mlflow_tracking(mode: str = "databricks"):
    """
    Configures MLflow tracking.

    Parameters
    ----------
    mode : str
        "databricks" or "local".
    """
    if mode == "databricks":
        try:
            mlflow.login(backend="databricks", interactive=False)
            mlflow.set_tracking_uri("databricks")
            print("INFO: Connected to Databricks MLflow tracking.")
        except Exception as e:
            raise RuntimeError(
                "MLflow Databricks setup failed. Ensure credentials are configured."
            ) from e
    elif mode == "local":
        # Use default local file-based backend (./mlruns)
        # Setting it to None or "" often defaults to ./mlruns, but explicit is better if env var is set differently.
        # However, the standard way to 'unset' to default is just not setting it, or setting it to a local path.
        # Let's explicitly set it to the local ./mlruns directory to be safe and clear.
        mlruns_path = Path.cwd() / "mlruns"
        mlruns_uri = f"file://{mlruns_path}"
        mlflow.set_tracking_uri(mlruns_uri)
        print(f"INFO: Using local file-based MLflow tracking backend: {mlruns_uri}")
    else:
        print(
            f"WARNING: Unknown MLflow mode '{mode}'. Using existing URI: {mlflow.get_tracking_uri()}"
        )


def get_mlflow_client() -> mlflow.tracking.MlflowClient:
    """Get an MLflow tracking client."""
    return mlflow.tracking.MlflowClient()


@contextmanager
def start_mlflow_run_context(
    experiment_name: str,
    parent_run_name: str,
    child_run_name: str,
    project_prefix: str = "/Shared/LSM-PoissonMPI-v3",
    args: Optional[argparse.Namespace] = None,
):
    """
    Context manager to start a nested MLflow run.
    """
    if mlflow.get_tracking_uri() == "databricks" and not experiment_name.startswith(
        "/"
    ):
        original_experiment_name = experiment_name
        experiment_name = f"{project_prefix}/{experiment_name}"
        print(
            f"DEBUG: Adjusted experiment name for Databricks: {original_experiment_name} -> {experiment_name}"
        )

    print(f"DEBUG: Attempting to set MLflow experiment: {experiment_name}")
    mlflow.set_experiment(experiment_name)
    print(f"INFO: Using MLflow experiment: {experiment_name}")

    client = get_mlflow_client()
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        try:
            exp_id = client.create_experiment(experiment_name)
            exp = client.get_experiment(exp_id)
            print(
                f"DEBUG: Created new MLflow experiment: {experiment_name} with ID {exp_id}"
            )
        except Exception as e:
            print(f"ERROR: Failed to create MLflow experiment '{experiment_name}': {e}")
            raise  # Re-raise to ensure failure is visible

    parent_runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.mlflow.runName = '{parent_run_name}' AND tags.is_parent = 'true'",
        max_results=1,
    )
    parent_run_id = parent_runs[0].info.run_id if parent_runs else None

    with mlflow.start_run(
        run_id=parent_run_id, run_name=parent_run_name, tags={"is_parent": "true"}
    ) as _parent_mlflow_run:  # noqa: F841
        with mlflow.start_run(run_name=child_run_name, nested=True) as child_mlflow_run:
            # Tag run with environment (HPC vs local) for easy filtering
            env = (
                "hpc"
                if os.environ.get("LSB_JOBID") or os.environ.get("SLURM_JOB_ID")
                else "local"
            )
            mlflow.set_tag("environment", env)

            print(
                f"INFO: Started MLflow run '{child_mlflow_run.info.run_name}' ({child_mlflow_run.info.run_id}) [{env}]"
            )
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
                    metrics_to_log.append(
                        mlflow.entities.Metric(name, val, timestamp, step)
                    )
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


def log_lsf_logs(job_name: Optional[str], log_dir: str = "logs/lsf"):
    """
    Upload LSF .out and .err log files as MLflow artifacts.

    Parameters
    ----------
    job_name : str or None
        The LSF job name (used to find log files)
    log_dir : str
        Directory containing LSF logs (default: logs/lsf)
    """
    if not job_name:
        return

    try:
        from Poisson import get_project_root

        project_root = get_project_root()
    except ImportError:
        project_root = Path.cwd()

    log_path = project_root / log_dir

    for ext in [".out", ".err"]:
        log_file = log_path / f"{job_name}{ext}"
        if log_file.exists():
            mlflow.log_artifact(str(log_file), artifact_path="lsf_logs")
            print(f"  ✓ Logged LSF log: {log_file.name}")
        # Don't warn if not found - logs may not exist yet during local testing


def load_runs(
    experiment: str,
    converged_only: bool = True,
    exclude_parent_runs: bool = True,
    project_prefix: str = "/Shared/LSM-PoissonMPI-v3",
) -> pd.DataFrame:
    """Load runs from ALL MLflow experiments matching the name.

    Parameters
    ----------
    experiment : str
        Experiment name (will be prefixed for Databricks)
    converged_only : bool
        Only include converged runs
    exclude_parent_runs : bool
        Exclude parent runs (keep only child/nested runs)
    project_prefix : str
        Databricks workspace prefix for experiment names
    """
    # Apply prefix for Databricks
    if mlflow.get_tracking_uri() == "databricks" and not experiment.startswith("/"):
        full_experiment_name = f"{project_prefix}/{experiment}"
    else:
        full_experiment_name = experiment

    # Find ALL experiments matching this name (there can be duplicates)
    client = get_mlflow_client()
    all_experiments = client.search_experiments(
        filter_string=f"name = '{full_experiment_name}'"
    )

    if not all_experiments:
        return pd.DataFrame()

    experiment_ids = [exp.experiment_id for exp in all_experiments]

    # Build filter string
    filters = []
    if converged_only:
        filters.append("metrics.converged = 1")

    filter_string = " and ".join(filters) if filters else ""

    # Fetch runs from ALL matching experiments
    df = mlflow.search_runs(
        experiment_ids=experiment_ids,
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
    force: bool = False,
    max_workers: int = 8,
) -> List[Path]:
    """
    Download artifacts from the newest run per run name in an experiment.

    When multiple runs have the same name, only artifacts from the most recent
    run are downloaded to avoid duplicates and ensure latest data.

    Parameters
    ----------
    experiment_name : str
        MLflow experiment name
    output_dir : Path
        Local directory to download to
    exclude_parent_runs : bool
        Skip parent runs (default True)
    force : bool
        Re-download even if file exists locally (default False)
    max_workers : int
        Number of parallel download threads (default 8)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = get_mlflow_client()
    exp = client.get_experiment_by_name(experiment_name)
    if not exp:
        print(f"  - Experiment '{experiment_name}' not found.")
        return []

    # Order by start_time DESC to get newest first
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
    )
    if not runs:
        return []

    # Filter out parent runs
    if exclude_parent_runs:
        runs = [r for r in runs if r.data.tags.get("is_parent") != "true"]

    # Keep only the newest run per run name (first occurrence since sorted DESC)
    seen_names = set()
    unique_runs = []
    for run in runs:
        run_name = run.info.run_name or run.info.run_id
        if run_name not in seen_names:
            seen_names.add(run_name)
            unique_runs.append(run)

    # Collect all artifacts to download
    download_tasks = []
    for run in unique_runs:
        run_id = run.info.run_id
        artifacts = client.list_artifacts(run_id)
        for artifact in artifacts:
            # Check if already exists locally (skip if not forcing)
            local_file = output_dir / artifact.path
            if not force and local_file.exists():
                continue
            download_tasks.append((run_id, artifact.path))

    if not download_tasks:
        return []

    # Download in parallel
    downloaded = []

    def download_one(task):
        run_id, artifact_path = task
        return client.download_artifacts(run_id, artifact_path, str(output_dir))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_one, task): task for task in download_tasks}
        for future in as_completed(futures):
            try:
                local_path = future.result()
                downloaded.append(Path(local_path))
            except Exception as e:
                task = futures[future]
                print(f"    ✗ Failed to download {task[1]}: {e}")

    return downloaded

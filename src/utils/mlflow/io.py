"""MLflow I/O utilities for fetching runs and artifacts.

This module provides helpers for retrieving experiment data from
MLflow/Databricks and downloading artifacts to the local data directory.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

import mlflow
import pandas as pd


def setup_mlflow_auth(tracking_uri: Optional[str] = None) -> None:
    """Configure MLflow authentication.

    Uses DATABRICKS_TOKEN environment variable if available (for CI),
    otherwise falls back to interactive login or provided tracking URI.

    Parameters
    ----------
    tracking_uri : str, optional
        MLflow tracking URI. If not provided, uses environment or interactive login.
    """
    token = os.environ.get("DATABRICKS_TOKEN")
    if token:
        # CI environment - set both host and token for Databricks auth
        host = os.environ.get(
            "DATABRICKS_HOST", "https://dbc-6756e917-e5fc.cloud.databricks.com"
        )
        os.environ["DATABRICKS_HOST"] = host
        mlflow.set_tracking_uri("databricks")
    elif tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        # Local environment - interactive login
        mlflow.login()


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
        Only download artifacts matching these extensions (e.g., [".h5", ".png"]).

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

    client = mlflow.tracking.MlflowClient()
    downloaded = []

    for _, row in df.iterrows():
        run_id = row["run_id"]
        artifacts = client.list_artifacts(run_id)

        for artifact in artifacts:
            if artifact_filter:
                if not any(artifact.path.endswith(ext) for ext in artifact_filter):
                    continue

            local_path = client.download_artifacts(run_id, artifact.path, output_dir)
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

    client = mlflow.tracking.MlflowClient()
    downloaded = []

    for _, row in df.iterrows():
        run_id = row["run_id"]

        # Extract parameters for naming
        n = row.get("params.n", row.get("params.N", "unknown"))

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

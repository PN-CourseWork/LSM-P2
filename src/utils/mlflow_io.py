"""MLflow I/O utilities for fetching runs and artifacts.

This module provides helpers for retrieving experiment data from
MLflow/Databricks and downloading artifacts to the local data directory.
"""

import os
from pathlib import Path
from typing import List, Optional
import mlflow
import pandas as pd


def setup_mlflow_auth():
    """Configure MLflow authentication.

    Uses DATABRICKS_TOKEN environment variable if available (for CI),
    otherwise falls back to interactive login.
    """
    token = os.environ.get("DATABRICKS_TOKEN")
    if token:
        # CI environment - set both host and token for Databricks auth
        host = "https://dbc-6756e917-e5fc.cloud.databricks.com"
        os.environ["DATABRICKS_HOST"] = host
        mlflow.set_tracking_uri("databricks")
    else:
        # Local environment - interactive login
        mlflow.login()


def load_runs(
    experiment: str,
    converged_only: bool = True,
    exclude_parent_runs: bool = True,
) -> pd.DataFrame:
    """Load runs from an MLflow experiment.

    Parameters
    ----------
    experiment : str
        Experiment name (e.g., "HPC-FV-Solver" or full path "/Shared/ANA-P3/HPC-FV-Solver").
    converged_only : bool, default True
        Only return runs where metrics.converged = 1.
    exclude_parent_runs : bool, default True
        Exclude parent runs (nested run containers).

    Returns
    -------
    pd.DataFrame
        DataFrame with run info, parameters (params.*), and metrics (metrics.*).

    Examples
    --------
    >>> df = load_runs("HPC-FV-Solver")
    >>> df[["run_id", "params.nx", "metrics.wall_time_seconds"]]
    """
    # Normalize experiment name
    if not experiment.startswith("/"):
        # Note: Adapted for LSM Project 2, assuming same shared folder structure or user should adjust
        experiment = f"/Shared/LSM-Project-2/{experiment}"

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
        Experiment name (e.g., "HPC-FV-Solver").
    output_dir : Path
        Directory to save artifacts. Files are named based on run parameters.
    converged_only : bool, default True
        Only download from converged runs.
    artifact_filter : list of str, optional
        Only download artifacts matching these patterns (e.g., ["*.h5", "*.png"]).
        If None, downloads all artifacts.

    Returns
    -------
    list of Path
        Paths to downloaded files.

    Examples
    --------
    >>> paths = download_artifacts("HPC-FV-Solver", Path("data/FV-Solver"))
    >>> print(paths)
    [Path('data/FV-Solver/LDC_N32_Re100.h5'), ...]
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get runs
    df = load_runs(experiment, converged_only=converged_only)
    if df.empty:
        print(f"No runs found for {experiment}")
        return []

    client = mlflow.tracking.MlflowClient()
    downloaded = []

    for _, row in df.iterrows():
        run_id = row["run_id"]

        # List artifacts
        artifacts = client.list_artifacts(run_id)

        for artifact in artifacts:
            # Apply filter if specified
            if artifact_filter:
                if not any(artifact.path.endswith(f) for f in artifact_filter):
                    continue

            # Download to output directory
            local_path = client.download_artifacts(run_id, artifact.path, output_dir)
            downloaded.append(Path(local_path))
            print(f"  Downloaded: {artifact.path}")

    return downloaded


def download_artifacts_with_naming(
    experiment: str,
    output_dir: Path,
    converged_only: bool = True,
) -> List[Path]:
    """Download HDF5 artifacts with standardized naming.

    Names files as: POISSON_N{n}_Iter{iter}.h5 (Adapted for LSM)

    Parameters
    ----------
    experiment : str
        Experiment name.
    output_dir : Path
        Directory to save artifacts.
    converged_only : bool, default True
        Only download from converged runs.

    Returns
    -------
    list of Path
        Paths to downloaded files.
    """
    import tempfile
    import shutil

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

        # Extract parameters for naming - Adapting to typical Poisson params
        # Assuming 'n' is grid size, 'max_iter' or 'iterations' might be useful
        n = row.get("params.n", row.get("params.N", "unknown"))

        # List artifacts and find HDF5 files
        artifacts = client.list_artifacts(run_id)

        for artifact in artifacts:
            if artifact.path.endswith(".h5"):
                # Download to temp location first
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = client.download_artifacts(run_id, artifact.path, tmpdir)

                    # Rename with standardized naming
                    new_name = f"Poisson_N{n}_{artifact.path.split('/')[-1]}"
                    final_path = output_dir / new_name

                    shutil.copy(tmp_path, final_path)
                    downloaded.append(final_path)
                    print(f"  {artifact.path} -> {new_name}")

    return downloaded

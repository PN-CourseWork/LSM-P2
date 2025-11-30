"""MLflow utilities for experiment tracking and artifact management.

Provides:
- Authentication setup for MLflow/Databricks
- Context manager for MLflow run orchestration
- Granular logging functions for parameters, metrics, time-series, and artifacts
- Run fetching and filtering
- Artifact downloading with naming conventions
- Log uploading for HPC jobs
"""

from .io import (
    setup_mlflow_tracking,
    start_mlflow_run_context,
    log_parameters,
    log_metrics_dict,
    log_timeseries_metrics,
    log_artifact_file,
    fetch_project_artifacts,
    load_runs,
    download_artifacts,
    download_artifacts_with_naming,
)
from .logs import upload_logs

__all__ = [
    "setup_mlflow_tracking",
    "start_mlflow_run_context",
    "log_parameters",
    "log_metrics_dict",
    "log_timeseries_metrics",
    "log_artifact_file",
    "fetch_project_artifacts",
    "load_runs",
    "download_artifacts",
    "download_artifacts_with_naming",
    "upload_logs",
]

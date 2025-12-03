"""Hydra callbacks for MLflow integration.

This module provides callbacks to integrate Hydra's job logging with MLflow.
The MLflowLogCallback uploads the Hydra job log file as an MLflow artifact
after each job completes, allowing you to view job output (including MPI
report-bindings) in the MLflow UI.
"""

import logging
from pathlib import Path
from typing import Any

from hydra.core.utils import JobReturn
from hydra.experimental.callback import Callback
from omegaconf import DictConfig

log = logging.getLogger(__name__)


class MLflowLogCallback(Callback):
    """Callback to log Hydra job output to MLflow as an artifact.

    This callback runs after each Hydra job completes and uploads the
    job log file to the active MLflow run as an artifact.

    Configuration (in hydra/callbacks/mlflow_log.yaml):

    .. code-block:: yaml

        # @package _global_
        hydra:
          callbacks:
            mlflow_log:
              _target_: utils.hydra.callbacks.MLflowLogCallback
              artifact_path: logs

    The log file will be uploaded to the "logs" artifact path in MLflow.
    """

    def __init__(self, artifact_path: str = "logs") -> None:
        """Initialize the callback.

        Parameters
        ----------
        artifact_path : str
            MLflow artifact subdirectory for log files (default: "logs")
        """
        self.artifact_path = artifact_path

    def on_job_end(
        self, config: DictConfig, job_return: JobReturn, **kwargs: Any
    ) -> None:
        """Upload job log to MLflow after job completes.

        Parameters
        ----------
        config : DictConfig
            The job config
        job_return : JobReturn
            Return value from the job (contains status, return value, etc.)
        kwargs : Any
            Additional keyword arguments
        """
        try:
            import mlflow
            from hydra.core.hydra_config import HydraConfig

            # Check if MLflow run is active
            if not mlflow.active_run():
                log.debug("No active MLflow run, skipping log upload")
                return

            # Get the job log file path from Hydra
            hc = HydraConfig.get()
            output_dir = Path(hc.runtime.output_dir)
            job_name = hc.job.name
            log_file = output_dir / f"{job_name}.log"

            if log_file.exists():
                mlflow.log_artifact(str(log_file), artifact_path=self.artifact_path)
                log.info(f"Uploaded job log to MLflow: {log_file.name}")
            else:
                log.debug(f"Job log not found: {log_file}")

        except Exception as e:
            # Don't fail the job if logging fails
            log.warning(f"Failed to upload job log to MLflow: {e}")

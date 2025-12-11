"""
Poisson Solver Runner - Clean Hydra + MLflow integration.

Architecture:
    - Orchestrator (main): Hydra entry, saves config, spawns MPI
    - Worker: MPI computation, MLflow logging

Usage:
    uv run python run_solver.py N=65 solver=jacobi n_ranks=1
    uv run python run_solver.py N=129 solver=fmg n_ranks=4

    # With custom MPI options (for HPC binding):
    uv run python run_solver.py --mpi-options "--map-by ppr:12:package --bind-to core" ...
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, fields
from typing import List, Optional

import hydra
import mlflow
from hydra.core.hydra_config import HydraConfig
from mlflow.tracking import MlflowClient
from mpi4py import MPI
from omegaconf import DictConfig, OmegaConf

from Poisson.datastructures import GlobalParams, LocalParams
from Poisson.solvers import FMGMPISolver, FMGSolver, JacobiMPISolver, JacobiSolver

log = logging.getLogger(__name__)


# =============================================================================
# Orchestrator (Hydra entry point)
# =============================================================================


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra orchestrator - saves config and spawns MPI workers."""
    output_dir = HydraConfig.get().runtime.output_dir
    cfg_path = os.path.join(output_dir, "config.yaml")
    OmegaConf.save(cfg, cfg_path)

    log.info(f"{cfg.solver}, N={cfg.N}, n_ranks={cfg.n_ranks}")

    # Build MPI command with optional binding options
    # Use sys.executable to ensure workers use same Python as orchestrator
    mpi_options = os.environ.get("MPI_OPTIONS", "").split()
    cmd = ["mpiexec"] + mpi_options + [
        "-n",
        str(cfg.n_ranks),
        sys.executable,
        __file__,
        cfg_path,
    ]
    log.info(f"Spawning: {' '.join(cmd)}")

    env = os.environ.copy()
    env["MPI_WORKER"] = "1"
    subprocess.run(cmd, env=env, timeout=600, check=True)


# =============================================================================
# Worker (MPI computation)
# =============================================================================


def worker(cfg_path: str) -> None:
    """MPI worker - runs solver and logs to MLflow."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    cfg = OmegaConf.load(cfg_path)
    params = config_to_params(cfg)

    # Setup MLflow (all ranks, but only rank 0 logs)
    mlflow.set_tracking_uri(cfg.mlflow.get("tracking_uri", "./mlruns"))

    # Apply project prefix for Databricks (experiment names must be absolute paths)
    experiment_name = cfg.experiment_name
    project_prefix = cfg.mlflow.get("project_prefix", "")
    if project_prefix and not experiment_name.startswith("/"):
        experiment_name = f"{project_prefix}/{experiment_name}"
    mlflow.set_experiment(experiment_name)

    if rank == 0:
        log.info(
            f"{params.solver}, N={params.N}, ranks={params.n_ranks}, "
            f"{params.strategy}/{params.communicator}"
        )

    # Run solver
    solver = create_solver(params, comm)
    solver.warmup()
    solver.solve()
    solver.post_solve()

    # Log results (rank 0 only)
    if rank == 0:
        log_to_mlflow(params, solver, solver.rank_topology)

    log.info(
        f"Done: {solver.metrics.iterations} iter, error={solver.metrics.final_error:.2e}, "
        f"time={solver.metrics.wall_time:.3f}s"
        + (f", {solver.metrics.mlups:.1f} MLup/s" if solver.metrics.mlups else "")
    )


# =============================================================================
# Shared utilities
# =============================================================================


def config_to_params(cfg: DictConfig) -> GlobalParams:
    """Convert Hydra config to GlobalParams."""
    param_fields = {f.name for f in fields(GlobalParams) if f.init}
    return GlobalParams(**{k: cfg[k] for k in param_fields})


def create_solver(params: GlobalParams, comm):
    """Create solver instance."""
    common = {
        "N": params.N,
        "omega": params.omega,
        "tolerance": params.tolerance,
        "max_iter": params.max_iter,
        "use_numba": params.use_numba,
        "specified_numba_threads": params.specified_numba_threads,
    }

    # Use MPI solvers when n_ranks > 1, sequential otherwise
    use_mpi = params.n_ranks > 1

    if params.solver == "jacobi":
        if use_mpi:
            return JacobiMPISolver(
                **common, strategy=params.strategy, communicator=params.communicator
            )
        return JacobiSolver(**common)
    else:  # fmg
        common.update(
            n_smooth=params.n_smooth, fmg_post_vcycles=params.fmg_post_vcycles
        )
        if use_mpi:
            return FMGMPISolver(
                **common, strategy=params.strategy, communicator=params.communicator
            )
        return FMGSolver(**common)


def log_to_mlflow(params: GlobalParams, solver, rank_topology: Optional[List[LocalParams]] = None) -> None:
    """Log solver run to MLflow."""
    with mlflow.start_run(run_name=f"{params.solver}_N{params.N}_p{params.n_ranks}") as run:
        # Params and metrics from dataclasses
        mlflow.log_params(params.to_mlflow())
        mlflow.log_metrics(solver.metrics.to_mlflow())

        # Timeseries: batch log step-based metrics
        timeseries = solver.timeseries.to_mlflow_batch()
        if timeseries:
            MlflowClient().log_batch(run_id=run.info.run_id, metrics=timeseries)

        # Rank topology artifact (for networkx visualizations)
        if rank_topology:
            topology_data = [asdict(rp) for rp in rank_topology]
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump(topology_data, f, indent=2)
                temp_path = f.name
            mlflow.log_artifact(temp_path, artifact_path="topology")
            os.unlink(temp_path)


# =============================================================================
# Entry point
# =============================================================================


if __name__ == "__main__":
    if os.environ.get("MPI_WORKER"):
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
        if len(sys.argv) < 2:
            sys.exit("Error: Worker requires config path argument")
        worker(sys.argv[1])
    else:
        main()

"""
Unified Solver Runner - runs sequential or MPI solvers based on n_ranks.

Usage:
    uv run python run_solver.py -cn experiment/validation
    uv run python run_solver.py -cn experiment/scaling --multirun
"""

import logging
import os
import subprocess
import sys
import tempfile

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def _get_param(cfg, *keys, default=None):
    """Helper to get param from nested config or global fallback."""
    val = cfg
    try:
        for k in keys:
            val = val[k]
        return val
    except (KeyError, AttributeError, TypeError):
        pass
    return default


def _get_hardware_info() -> dict:
    """Get hostname, socket ID, and CPU model for current process."""
    import platform
    import socket as sock

    info = {"hostname": sock.gethostname(), "cpu_model": platform.processor() or "unknown", "socket_id": -1}

    return info


def _create_solver(cfg: DictConfig, is_mpi: bool = False, **mpi_kwargs):
    """Create solver instance from config."""
    from Poisson.solvers import JacobiSolver, FMGSolver, JacobiMPISolver, FMGMPISolver

    N = _get_param(cfg, "problem", "N") or cfg.get("N")
    solver_name = _get_param(cfg, "solver", "name") or cfg.get("solver_type", "jacobi")
    
    # Gather params from structured location or global fallback
    solver_params = _get_param(cfg, "solver", "params") or {}
    # Merge legacy global keys if present
    for k in ["omega", "tolerance", "max_iter", "numba_threads", "n_smooth", "fmg_post_vcycles", "use_numba"]:
        if k in cfg:
            solver_params[k] = cfg[k]

    common = {"use_numba": solver_params.get("use_numba", False)}

    if solver_name == "jacobi":
        cls = JacobiMPISolver if is_mpi else JacobiSolver
        params = {
            "N": N, 
            "omega": solver_params.get("omega", 0.8), 
            "tolerance": solver_params.get("tolerance", 1e-6),
            "max_iter": solver_params.get("max_iter", 10000), 
            "numba_threads": solver_params.get("numba_threads", 1), 
            **common
        }
    else:  # multigrid/fmg
        cls = FMGMPISolver if is_mpi else FMGSolver
        params = {
            "N": N, 
            "n_smooth": solver_params.get("n_smooth", 3), 
            "omega": solver_params.get("omega", 2/3),
            "tolerance": solver_params.get("tolerance", 1e-16), 
            "max_iter": solver_params.get("max_iter", 100),
            "fmg_post_vcycles": solver_params.get("fmg_post_vcycles", 1), 
            **common
        }
        if is_mpi:
            params["numba_threads"] = solver_params.get("numba_threads", 1)

    if is_mpi:
        params.update(mpi_kwargs)

    return cls(**params)


def _log_results(cfg, solver, n_ranks: int, hw_info: list = None):
    """Log solver results to MLflow."""
    import numpy as np
    from dataclasses import asdict
    import mlflow
    from utils.mlflow.io import (
        start_mlflow_run_context, log_parameters, log_metrics_dict, log_timeseries_metrics
    )

    solver_name = _get_param(cfg, "solver", "name") or cfg.get("solver_type", "jacobi")
    strategy = _get_param(cfg, "solver", "strategy") or cfg.get("strategy")
    communicator = _get_param(cfg, "solver", "communicator") or cfg.get("communicator")
    N = _get_param(cfg, "problem", "N") or cfg.get("N")
    
    solver_params = _get_param(cfg, "solver", "params") or {}
    # Merge legacy global keys
    for k in ["omega", "tolerance", "max_iter", "use_numba", "numba_threads"]:
        if k in cfg:
            solver_params[k] = cfg[k]

    experiment_name = cfg.get("experiment_name") or "default"
    run_name = f"{solver_name}_N{N}_p{n_ranks}" + (f"_{strategy}" if strategy else "")

    with start_mlflow_run_context(experiment_name=experiment_name, parent_run_name=f"N{N}", child_run_name=run_name):
        params = {
            "N": N, "n_ranks": n_ranks, "solver": solver_name,
            "omega": solver_params.get("omega"), 
            "max_iter": solver_params.get("max_iter"), 
            "tolerance": solver_params.get("tolerance")
        }
        if strategy:
            params.update({"strategy": strategy, "communicator": communicator,
                          "label": f"{communicator.title()} ({strategy})"})
        if hasattr(solver, "local_shape"):
            params.update({"local_volume": int(np.prod(solver.local_shape)), "halo_size_mb": solver.halo_size_mb})
        if solver_params.get("use_numba"):
            params.update({"use_numba": True, "numba_threads": solver_params.get("numba_threads", 1)})

        log_parameters(params)
        
        metrics_to_log = asdict(solver.results)
        
        log_metrics_dict(metrics_to_log)

        if hw_info:
            import pandas as pd
            hw_df = pd.DataFrame(hw_info)
            mlflow.log_table(hw_df, artifact_file="hardware.json")
            log_parameters({"nodes": hw_df["hostname"].nunique()})

        if solver.timeseries.halo_exchange_times:
            halo = np.array(solver.timeseries.halo_exchange_times) * 1e6
            log_metrics_dict({f"halo_time_{k}_us": float(fn(halo)) for k, fn in
                             [("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max)]})

        log_timeseries_metrics(solver.timeseries)

    log.info(f"Done: {solver.results.iterations} iter, error={solver.results.final_error:.2e}, "
             f"time={solver.results.wall_time:.3f}s" + (f", {solver.results.mlups:.1f} Mlup/s" if solver.results.mlups else ""))


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Entry point - runs sequential or spawns MPI based on n_ranks."""
    n_ranks = cfg.get("n_ranks", 1)
    N = _get_param(cfg, "problem", "N") or cfg.get("N")
    solver_name = _get_param(cfg, "solver", "name") or cfg.get("solver_type", "jacobi")
    
    log.info(f"{solver_name}, N={N}, n_ranks={n_ranks}")

    if n_ranks == 1:
        _run_sequential(cfg)
    else:
        _spawn_mpi(cfg, n_ranks)


def _run_sequential(cfg: DictConfig):
    """Run sequential solver."""
    from utils.mlflow.io import setup_mlflow_tracking

    setup_mlflow_tracking(mode=cfg.mlflow.mode)

    solver_name = _get_param(cfg, "solver", "name") or cfg.get("solver_type", "jacobi")
    if solver_name not in ["jacobi", "multigrid", "fmg"]:
        log.error(f"Unknown solver: {solver_name}")
        sys.exit(1)

    solver = _create_solver(cfg)
    solver.warmup()
    solver.fmg_solve() if solver_name in ["multigrid", "fmg"] else solver.solve()
    solver.compute_l2_error()
    _log_results(cfg, solver, n_ranks=1)

    # Log artifacts (Hydra config and logs)
    import mlflow
    try:
        hydra_cfg = HydraConfig.get()
        output_dir = hydra_cfg.runtime.output_dir
        mlflow.log_artifact(os.path.join(output_dir, ".hydra", "config.yaml"), artifact_path="hydra")
        mlflow.log_artifact(os.path.join(output_dir, "run_solver.log"), artifact_path="hydra")
    except Exception as e:
        log.warning(f"Could not log Hydra artifacts: {e}")


def _spawn_mpi(cfg: DictConfig, n_ranks: int):
    """Spawn MPI subprocess with full config dump."""
    # Determine output dir
    try:
        hydra_cfg = HydraConfig.get()
        output_dir = hydra_cfg.runtime.output_dir
    except Exception:
        output_dir = tempfile.gettempdir()
        
    # Save config for workers
    cfg_path = os.path.join(output_dir, "mpi_config.yaml")
    OmegaConf.save(cfg, cfg_path)
    
    mpi = _get_param(cfg, "machine", "mpi") or cfg.get("mpi", {})
    env = os.environ.copy()
    env["MPI_SUBPROCESS"] = "1"

    cmd = ["mpiexec", "-n", str(n_ranks)]
    if mpi.get("bind_to"):
        cores_per_node = mpi.get("cores_per_node", 24)
        alloc = int(os.environ.get("LSB_DJOB_NUMPROC", cores_per_node))
        n_nodes = max(1, alloc // cores_per_node)
        rps = max(1, (n_ranks + n_nodes - 1) // n_nodes // mpi.get("sockets_per_node", 2))
        cmd.extend(["--report-bindings", "--map-by", f"ppr:{rps}:package", "--bind-to", str(mpi.bind_to)])

    cmd.extend(["uv", "run", "python", os.path.abspath(__file__), cfg_path])

    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    for line in (result.stdout or "").strip().split("\n"):
        if line:
            log.info(line)
    for line in (result.stderr or "").strip().split("\n"):
        if line:
            log.warning(line) if "error" in line.lower() else log.info(line)
            
    # Log artifacts (Hydra config and logs) from the spawner
    import mlflow
    try:
        if mlflow.active_run():
            mlflow.log_artifact(os.path.join(output_dir, ".hydra", "config.yaml"), artifact_path="hydra")
            mlflow.log_artifact(os.path.join(output_dir, "run_solver.log"), artifact_path="hydra")
    except Exception as e:
        log.warning(f"Could not log Hydra artifacts: {e}")


def _run_mpi_solver(cfg: DictConfig, comm):
    """Run MPI solver (called within mpiexec subprocess)."""
    from utils.mlflow.io import setup_mlflow_tracking

    rank, n_ranks = comm.Get_rank(), comm.Get_size()
    hw_info = _get_hardware_info()
    hw_info["rank"] = rank
    all_hw = comm.gather(hw_info, root=0)

    solver_name = _get_param(cfg, "solver", "name") or cfg.get("solver_type", "jacobi")
    strategy = _get_param(cfg, "solver", "strategy") or cfg.get("strategy", "sliced")
    communicator = _get_param(cfg, "solver", "communicator") or cfg.get("communicator", "custom")
    N = _get_param(cfg, "problem", "N") or cfg.get("N")

    if rank == 0:
        setup_mlflow_tracking(mode=cfg.mlflow.mode)
        log.info(f"{solver_name}, N={N}, ranks={n_ranks}, {strategy}/{communicator}")

    if solver_name not in ["jacobi", "multigrid", "fmg"]:
        if rank == 0:
            log.error(f"Unknown solver: {solver_name}")
        sys.exit(1)

    solver = _create_solver(cfg, is_mpi=True, strategy=strategy, communicator=communicator)
    solver.warmup()
    solver.fmg_solve() if solver_name in ["multigrid", "fmg"] else solver.solve()
    solver.compute_l2_error()

    if rank == 0:
        _log_results(cfg, solver, n_ranks, all_hw)


if __name__ == "__main__":
    if os.environ.get("MPI_SUBPROCESS"):
        # Worker mode: Load config from file passed as arg
        from mpi4py import MPI
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
        
        if len(sys.argv) > 1:
            cfg_path = sys.argv[1]
            cfg = OmegaConf.load(cfg_path)
            _run_mpi_solver(cfg, MPI.COMM_WORLD)
        else:
            print("Error: Worker started without config path.")
            sys.exit(1)
    else:
        # Hydra mode
        main()
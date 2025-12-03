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

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def _get_hardware_info() -> dict:
    """Get hostname, socket ID, and CPU model for current process."""
    import platform
    import socket as sock

    info = {"hostname": sock.gethostname(), "cpu_model": platform.processor() or "unknown", "socket_id": -1}

    return info


def _create_solver(cfg: DictConfig, solver_type: str, is_mpi: bool = False, **mpi_kwargs):
    """Create solver instance from config."""
    from Poisson.solvers import JacobiSolver, FMGSolver, JacobiMPISolver, FMGMPISolver

    N = cfg.N
    common = {"use_numba": cfg.get("use_numba", False)}

    if solver_type == "jacobi":
        cls = JacobiMPISolver if is_mpi else JacobiSolver
        params = {
            "N": N, "omega": cfg.get("omega", 0.8), "tolerance": cfg.get("tolerance", 1e-6),
            "max_iter": cfg.get("max_iter", 10000), "numba_threads": cfg.get("numba_threads", 1), **common
        }
    else:  # multigrid/fmg
        cls = FMGMPISolver if is_mpi else FMGSolver
        params = {
            "N": N, "n_smooth": cfg.get("n_smooth", 3), "omega": cfg.get("omega", 2/3),
            "tolerance": cfg.get("tolerance", 1e-16), "max_iter": cfg.get("max_iter", 100),
            "fmg_post_vcycles": cfg.get("fmg_post_vcycles", 1), **common
        }
        if is_mpi:
            params["numba_threads"] = cfg.get("numba_threads", 1)

    if is_mpi:
        params.update(mpi_kwargs)

    return cls(**params)


def _log_results(cfg, solver, solver_type: str, n_ranks: int, strategy: str = None,
                 communicator: str = None, hw_info: list = None):
    """Log solver results to MLflow."""
    import numpy as np
    from dataclasses import asdict
    import mlflow
    from utils.mlflow.io import (start_mlflow_run_context, log_parameters, log_metrics_dict, log_timeseries_metrics)

    experiment_name = cfg.get("experiment_name") or "default"
    run_name = f"{solver_type}_N{cfg.N}_p{n_ranks}" + (f"_{strategy}" if strategy else "")

    with start_mlflow_run_context(experiment_name=experiment_name, parent_run_name=f"N{cfg.N}", child_run_name=run_name):
        params = {"N": cfg.N, "n_ranks": n_ranks, "solver": solver_type,
                  "omega": cfg.get("omega"), "max_iter": cfg.get("max_iter"), "tolerance": cfg.get("tolerance")}
        if strategy:
            params.update({"strategy": strategy, "communicator": communicator,
                          "label": f"{communicator.title()} ({strategy})"})
        if hasattr(solver, "local_shape"):
            params.update({"local_volume": int(np.prod(solver.local_shape)), "halo_size_mb": solver.halo_size_mb})
        if cfg.get("use_numba"):
            params.update({"use_numba": True, "numba_threads": cfg.get("numba_threads", 1)})

        log_parameters(params)
        log_metrics_dict(asdict(solver.results))

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


@hydra.main(config_path="Experiments/hydra-conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Entry point - runs sequential or spawns MPI based on n_ranks."""
    n_ranks = cfg.get("n_ranks", 1)
    solver_type = cfg.get("solver_type", "jacobi")
    log.info(f"{solver_type}, N={cfg.N}, n_ranks={n_ranks}")

    if n_ranks == 1:
        _run_sequential(cfg, solver_type)
    else:
        _spawn_mpi(cfg, n_ranks)


def _run_sequential(cfg: DictConfig, solver_type: str):
    """Run sequential solver."""
    from utils.mlflow.io import setup_mlflow_tracking

    setup_mlflow_tracking(mode=cfg.mlflow.mode)

    if solver_type not in ["jacobi", "multigrid", "fmg"]:
        log.error(f"Unknown solver: {solver_type}")
        sys.exit(1)

    solver = _create_solver(cfg, solver_type)
    solver.warmup()
    solver.fmg_solve() if solver_type in ["multigrid", "fmg"] else solver.solve()
    solver.compute_l2_error()
    _log_results(cfg, solver, solver_type, n_ranks=1)


def _spawn_mpi(cfg: DictConfig, n_ranks: int):
    """Spawn MPI subprocess."""
    mpi = cfg.get("mpi", {})
    env = os.environ.copy()
    env["MPI_SUBPROCESS"] = "1"

    cmd = ["mpiexec", "-n", str(n_ranks)]
    if mpi.get("bind_to"):
        cores_per_node = mpi.get("cores_per_node", 24)
        alloc = int(os.environ.get("LSB_DJOB_NUMPROC", cores_per_node))
        n_nodes = max(1, alloc // cores_per_node)
        rps = max(1, (n_ranks + n_nodes - 1) // n_nodes // mpi.get("sockets_per_node", 2))
        cmd.extend(["--report-bindings", "--map-by", f"ppr:{rps}:package", "--bind-to", str(mpi.bind_to)])

    cmd.extend(["uv", "run", "python", os.path.abspath(__file__)])

    # Pass config as args
    for key in ["n_ranks", "N", "strategy", "solver_type", "communicator", "omega", "max_iter",
                "tolerance", "experiment_name", "use_numba", "numba_threads", "n_smooth", "fmg_post_vcycles"]:
        if "." in key:
            val = cfg
            for k in key.split("."):
                val = val.get(k) if val else None
        else:
            val = cfg.get(key)
        if val is not None:
            cmd.append(f"{key}={val}")
    cmd.append(f"mlflow.mode={cfg.mlflow.mode}")

    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    for line in (result.stdout or "").strip().split("\n"):
        if line:
            log.info(line)
    for line in (result.stderr or "").strip().split("\n"):
        if line:
            log.warning(line) if "error" in line.lower() else log.info(line)


def _run_mpi_solver(cfg: DictConfig, comm):
    """Run MPI solver (called within mpiexec subprocess)."""
    from utils.mlflow.io import setup_mlflow_tracking

    rank, n_ranks = comm.Get_rank(), comm.Get_size()
    hw_info = _get_hardware_info()
    hw_info["rank"] = rank
    all_hw = comm.gather(hw_info, root=0)

    solver_type = cfg.get("solver_type", "jacobi")
    strategy = cfg.get("strategy", "sliced")
    communicator = cfg.get("communicator", "custom")

    if rank == 0:
        setup_mlflow_tracking(mode=cfg.mlflow.mode)
        log.info(f"{solver_type}, N={cfg.N}, ranks={n_ranks}, {strategy}/{communicator}")

    if solver_type not in ["jacobi", "multigrid", "fmg"]:
        if rank == 0:
            log.error(f"Unknown solver: {solver_type}")
        sys.exit(1)

    solver = _create_solver(cfg, solver_type, is_mpi=True, strategy=strategy, communicator=communicator)
    solver.warmup()
    solver.fmg_solve() if solver_type in ["multigrid", "fmg"] else solver.solve()
    solver.compute_l2_error()

    if rank == 0:
        _log_results(cfg, solver, solver_type, n_ranks, strategy, communicator, all_hw)


if __name__ == "__main__":
    if os.environ.get("MPI_SUBPROCESS"):
        from mpi4py import MPI

        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

        # Parse key=value args
        cfg_dict = {}
        for arg in sys.argv[1:]:
            if "=" in arg and not arg.startswith("-"):
                key, val = arg.split("=", 1)
                d = cfg_dict
                for k in key.split(".")[:-1]:
                    d = d.setdefault(k, {})
                try:
                    d[key.split(".")[-1]] = {"true": True, "false": False}.get(val.lower()) if val.lower() in ("true", "false") \
                        else float(val) if ("." in val or "e" in val.lower()) else int(val)
                except ValueError:
                    d[key.split(".")[-1]] = val

        _run_mpi_solver(OmegaConf.create(cfg_dict), MPI.COMM_WORLD)
    else:
        main()

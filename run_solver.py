"""
Unified Solver Runner (Hydra v2)
================================

Single runner for all solver experiments using Hydra's native multirun.

Solver selection:
- n_ranks=1: JacobiSolver or FMGSolver (no MPI overhead)
- n_ranks>1: JacobiMPISolver or FMGMPISolver (with MPI)

Usage
-----

.. code-block:: bash

    # Single run (local)
    uv run python run_solver.py -cn experiment/validation

    # Multirun sweep (local - runs all sequentially)
    uv run python run_solver.py -cn experiment/scaling --multirun
"""

import logging
import os
import subprocess
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@hydra.main(config_path="Experiments/hydra-conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Entry point - runs sequential or spawns MPI based on n_ranks.

    In multirun mode, Hydra calls this once per sweep combination.
    """
    mpi = cfg.get("mpi", {})
    cores_per_node = mpi.get("cores_per_node", 24)
    alloc_cores = int(os.environ.get("LSB_DJOB_NUMPROC", cores_per_node))

    n_ranks = cfg.get("n_ranks", 1)
    solver_type = cfg.get("solver_type", "jacobi")
    N = cfg.get("N", 64)

    log.info(f"{solver_type}, N={N}, n_ranks={n_ranks}")

    # Sequential vs MPI
    if n_ranks == 1:
        _run_sequential_solver(cfg)
    else:
        _spawn_mpi(cfg, n_ranks, alloc_cores, cores_per_node, mpi)


def _spawn_mpi(cfg: DictConfig, n_ranks: int, alloc_cores: int, cores_per_node: int, mpi: dict):
    """Spawn MPI subprocess for parallel execution."""
    script = os.path.abspath(__file__)

    env = os.environ.copy()
    env["MPI_SUBPROCESS"] = "1"

    # Build MPI command
    cmd = ["mpiexec", "-n", str(n_ranks)]

    # Add binding options if configured
    if mpi.get("bind_to"):
        cmd.append("--report-bindings")
        sockets_per_node = mpi.get("sockets_per_node", 2)
        n_nodes = max(1, alloc_cores // cores_per_node)
        ranks_per_node = max(1, (n_ranks + n_nodes - 1) // n_nodes)  # ceiling div
        ranks_per_socket = max(1, (ranks_per_node + sockets_per_node - 1) // sockets_per_node)  # ceiling div
        # Map ranks across sockets (packages) for NUMA spreading
        cmd.extend(["--map-by", f"ppr:{ranks_per_socket}:package"])
        cmd.extend(["--bind-to", str(mpi.bind_to)])

    cmd.extend(["uv", "run", "python", script])

    # Pass config as command line args (MPI subprocess bypasses Hydra)
    overrides = [
        f"n_ranks={n_ranks}",
        f"N={cfg.N}",
        f"strategy={cfg.get('strategy', 'sliced')}",
        f"solver_type={cfg.get('solver_type', 'jacobi')}",
        f"communicator={cfg.get('communicator', 'custom')}",
        f"omega={cfg.get('omega', 0.8)}",
        f"max_iter={cfg.get('max_iter', 1000)}",
        f"tolerance={cfg.get('tolerance', 1e-6)}",
        f"mlflow.mode={cfg.mlflow.mode}",
    ]
    if cfg.get("experiment_name"):
        overrides.append(f"experiment_name={cfg.experiment_name}")
    if cfg.get("use_numba"):
        overrides.append(f"use_numba={cfg.use_numba}")
        overrides.append(f"numba_threads={cfg.get('numba_threads', 1)}")
    if cfg.get("n_smooth"):
        overrides.append(f"n_smooth={cfg.n_smooth}")
    if cfg.get("fmg_post_vcycles"):
        overrides.append(f"fmg_post_vcycles={cfg.fmg_post_vcycles}")

    cmd.extend(overrides)

    log.debug(f"MPI command: {' '.join(cmd)}")

    # Run MPI subprocess and capture output for logging
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)

    # Log output using Python logging so Hydra captures it
    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            log.info(line)
    if result.stderr:
        for line in result.stderr.strip().split("\n"):
            # MPI report-bindings goes to stderr, log as info not warning
            if "binding" in line.lower() or "[" in line:
                log.info(line)
            else:
                log.warning(line)


def _run_sequential_solver(cfg: DictConfig):
    """Run sequential solver (no MPI)."""
    from Poisson.solvers import JacobiSolver, FMGSolver
    from utils.mlflow.io import setup_mlflow_tracking

    N = cfg.N
    solver_type = cfg.get("solver_type", "jacobi")
    experiment_name = cfg.get("experiment_name", "experiment")

    setup_mlflow_tracking(mode=cfg.mlflow.mode)
    log.info("=" * 60)
    log.info(f"Experiment: {experiment_name}")
    log.info(f"Solver: {solver_type} (sequential), N={N}")
    log.info("=" * 60)

    # Create solver
    if solver_type == "jacobi":
        solver = JacobiSolver(
            N=N,
            omega=cfg.get("omega", 0.8),
            tolerance=cfg.get("tolerance", 1e-6),
            max_iter=cfg.get("max_iter", 10000),
            use_numba=cfg.get("use_numba", False),
            numba_threads=cfg.get("numba_threads", 1),
        )
    elif solver_type in ["multigrid", "fmg"]:
        solver = FMGSolver(
            N=N,
            n_smooth=cfg.get("n_smooth", 3),
            omega=cfg.get("omega", 2/3),
            tolerance=cfg.get("tolerance", 1e-16),
            max_iter=cfg.get("max_iter", 100),
            fmg_post_vcycles=cfg.get("fmg_post_vcycles", 1),
            use_numba=cfg.get("use_numba", False),
        )
    else:
        log.error(f"Unknown solver: {solver_type}")
        sys.exit(1)

    # Warmup and solve
    solver.warmup()
    if solver_type in ["multigrid", "fmg"]:
        solver.fmg_solve()
    else:
        solver.solve()

    solver.compute_l2_error()

    # Log to MLflow
    _log_results(cfg, solver, solver_type, N, n_ranks=1)


def _get_hardware_info() -> dict:
    """Get detailed hardware info for the current process."""
    import platform
    import socket as sock

    info = {
        "hostname": sock.gethostname(),
        "processor": platform.processor() or "unknown",
        "platform": platform.platform(),
        "cpu_model": "unknown",
        "core_id": -1,
        "socket_id": -1,
    }

    # Get CPU affinity (which core(s) this process can run on)
    try:
        affinity = os.sched_getaffinity(0)
        info["cpu_affinity"] = list(affinity)
        if len(affinity) == 1:
            info["core_id"] = list(affinity)[0]
    except (AttributeError, OSError):
        # sched_getaffinity not available (e.g., macOS)
        info["cpu_affinity"] = []

    # Try to get CPU model from /proc/cpuinfo (Linux)
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":")[1].strip()
                    break
    except (IOError, OSError):
        pass

    # Try to get socket/NUMA info for the bound core
    if info["core_id"] >= 0:
        try:
            # Physical package ID = socket
            pkg_path = f"/sys/devices/system/cpu/cpu{info['core_id']}/topology/physical_package_id"
            with open(pkg_path, "r") as f:
                info["socket_id"] = int(f.read().strip())
        except (IOError, OSError, ValueError):
            pass

        try:
            # NUMA node
            import glob
            numa_paths = glob.glob(f"/sys/devices/system/cpu/cpu{info['core_id']}/node*")
            if numa_paths:
                # Extract node number from path like /sys/.../node0
                info["numa_node"] = int(numa_paths[0].split("node")[-1])
        except (IOError, OSError, ValueError, IndexError):
            pass

    return info


def _run_mpi_solver(cfg: DictConfig, comm):
    """Run MPI solver (called within mpiexec subprocess)."""
    from mpi4py import MPI
    from Poisson.solvers import JacobiMPISolver, FMGMPISolver
    from utils.mlflow.io import setup_mlflow_tracking

    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    # Gather hardware info from all ranks
    hw_info = _get_hardware_info()
    hw_info["rank"] = rank
    all_hw_info = comm.gather(hw_info, root=0)

    N = cfg.N
    solver_type = cfg.get("solver_type", "jacobi")
    strategy = cfg.get("strategy", "sliced")
    communicator = cfg.get("communicator", "custom")
    experiment_name = cfg.get("experiment_name", "experiment")

    if rank == 0:
        setup_mlflow_tracking(mode=cfg.mlflow.mode)
        log.info("=" * 60)
        log.info(f"Experiment: {experiment_name}")
        log.info(f"Solver: {solver_type}, N={N}, ranks={n_ranks}")
        log.info(f"Strategy: {strategy}, Communicator: {communicator}")
        log.info("=" * 60)

    # Create solver
    if solver_type == "jacobi":
        solver = JacobiMPISolver(
            N=N,
            strategy=strategy,
            communicator=communicator,
            omega=cfg.get("omega", 0.8),
            tolerance=cfg.get("tolerance", 1e-6),
            max_iter=cfg.get("max_iter", 10000),
            use_numba=cfg.get("use_numba", False),
            numba_threads=cfg.get("numba_threads", 1),
        )
    elif solver_type in ["multigrid", "fmg"]:
        solver = FMGMPISolver(
            N=N,
            strategy=strategy,
            communicator=communicator,
            n_smooth=cfg.get("n_smooth", 3),
            omega=cfg.get("omega", 2/3),
            tolerance=cfg.get("tolerance", 1e-16),
            max_iter=cfg.get("max_iter", 100),
            fmg_post_vcycles=cfg.get("fmg_post_vcycles", 1),
            use_numba=cfg.get("use_numba", False),
            numba_threads=cfg.get("numba_threads", 1),
        )
    else:
        if rank == 0:
            log.error(f"Unknown solver: {solver_type}")
        sys.exit(1)

    # Warmup and solve
    solver.warmup()
    if solver_type in ["multigrid", "fmg"]:
        solver.fmg_solve()
    else:
        solver.solve()

    solver.compute_l2_error()

    # Log to MLflow (rank 0 only)
    if rank == 0:
        _log_results(cfg, solver, solver_type, N, n_ranks, strategy, communicator, all_hw_info)


def _log_results(cfg, solver, solver_type, N, n_ranks, strategy=None, communicator=None, hw_info=None):
    """Log solver results to MLflow."""
    import numpy as np
    from dataclasses import asdict
    from pathlib import Path
    import mlflow
    from utils.mlflow.io import (
        start_mlflow_run_context,
        log_parameters,
        log_metrics_dict,
        log_timeseries_metrics,
    )

    experiment_name = cfg.get("experiment_name") or "default"
    run_name = f"{solver_type}_N{N}_p{n_ranks}"
    if strategy:
        run_name += f"_{strategy}"

    # Get local volume info if available
    local_volume = None
    halo_size_mb = None
    if hasattr(solver, "local_shape"):
        local_volume = int(np.prod(solver.local_shape))
        halo_size_mb = solver.halo_size_mb

    # Create label for plots
    if strategy and communicator:
        comm_type = "contiguous" if strategy == "sliced" else "mixed"
        label = f"{communicator.title()} ({strategy}, {comm_type})"
    else:
        label = "sequential"

    with start_mlflow_run_context(
        experiment_name=experiment_name,
        parent_run_name=f"N{N}",
        child_run_name=run_name,
    ):
        params = {
            "N": N,
            "n_ranks": n_ranks,
            "solver": solver_type,
            "label": label,
            "omega": cfg.get("omega"),
            "max_iter": cfg.get("max_iter"),
            "tolerance": cfg.get("tolerance"),
        }
        if strategy:
            params["strategy"] = strategy
            params["communicator"] = communicator
        if local_volume:
            params["local_volume"] = local_volume
            params["halo_size_mb"] = halo_size_mb
        if cfg.get("use_numba"):
            params["use_numba"] = True
            params["numba_threads"] = cfg.get("numba_threads", 1)
        if cfg.get("scaling_type"):
            params["scaling_type"] = cfg.get("scaling_type")

        log_parameters(params)
        log_metrics_dict(asdict(solver.results))

        # Log hardware info per rank as table
        if hw_info:
            import pandas as pd
            hw_df = pd.DataFrame(hw_info)
            mlflow.log_table(hw_df, artifact_file="hardware.json")
            n_nodes = hw_df["hostname"].nunique()
            log_parameters({"nodes": n_nodes})
            log.info(f"Logged hardware info: {n_nodes} nodes, {len(hw_info)} ranks")

        # Log halo timing stats
        if solver.timeseries.halo_exchange_times:
            halo_times = np.array(solver.timeseries.halo_exchange_times)
            log_metrics_dict({
                "halo_time_mean_us": float(np.mean(halo_times) * 1e6),
                "halo_time_std_us": float(np.std(halo_times) * 1e6),
                "halo_time_min_us": float(np.min(halo_times) * 1e6),
                "halo_time_max_us": float(np.max(halo_times) * 1e6),
            })

        log_timeseries_metrics(solver.timeseries)

        # Log Hydra job log file as artifact (contains MPI report-bindings)
        try:
            from hydra.core.hydra_config import HydraConfig
            hc = HydraConfig.get()
            output_dir = Path(hc.runtime.output_dir)
            job_name = hc.job.name
            log_file = output_dir / f"{job_name}.log"
            if log_file.exists():
                mlflow.log_artifact(str(log_file), artifact_path="logs")
                log.info(f"Uploaded job log to MLflow: {log_file.name}")
        except Exception as e:
            log.debug(f"Could not upload job log: {e}")

    log.info(f"--- {solver_type.upper()} Complete ---")
    log.info(f"  Iterations: {solver.results.iterations}")
    log.info(f"  L2 error: {solver.results.final_error:.6e}")
    log.info(f"  Wall time: {solver.results.wall_time:.4f}s")
    if solver.results.mlups:
        log.info(f"  Performance: {solver.results.mlups:.2f} Mlup/s")
    if solver.results.bandwidth_gb_s:
        log.info(f"  Bandwidth: {solver.results.bandwidth_gb_s:.2f} GB/s")


if __name__ == "__main__":
    # Skip Hydra for MPI subprocesses - parse args directly
    if os.environ.get("MPI_SUBPROCESS"):
        from mpi4py import MPI
        from omegaconf import OmegaConf

        # Set up basic logging since we bypass Hydra
        logging.basicConfig(
            level=logging.INFO,
            format="[%(levelname)s] %(message)s",
        )

        # Parse command line args as key=value pairs
        cfg_dict = {}
        for arg in sys.argv[1:]:
            if "=" in arg and not arg.startswith("-"):
                key, value = arg.split("=", 1)
                # Handle nested keys
                keys = key.split(".")
                d = cfg_dict
                for k in keys[:-1]:
                    d = d.setdefault(k, {})
                # Try to parse as int/float/bool
                try:
                    if value.lower() in ("true", "false"):
                        d[keys[-1]] = value.lower() == "true"
                    elif "." in value or "e" in value.lower():
                        d[keys[-1]] = float(value)
                    else:
                        d[keys[-1]] = int(value)
                except ValueError:
                    d[keys[-1]] = value

        cfg = OmegaConf.create(cfg_dict)
        _run_mpi_solver(cfg, MPI.COMM_WORLD)
    else:
        main()

"""
Unified Solver Runner
=====================

Single runner for all solver-based experiments. Spawns MPI subprocess
with rank count from Hydra config. Results logged to MLflow.

Supports LSF job arrays for parallel experiment execution:
- LSB_DJOB_NUMPROC: allocated cores (determines valid rank counts)
- LSB_JOBINDEX: job array index (selects which config to run)

Usage
-----

.. code-block:: bash

    # Single run (local)
    uv run python run_solver.py -cn experiment/validation

    # HPC with job array (each element runs one config)
    #BSUB -J scaling[1-66]
    #BSUB -n 48
    uv run python run_solver.py -cn experiment/scaling
"""

# %%
# Setup
# -----

import os
import subprocess
import sys
from dataclasses import asdict
from itertools import product

import hydra
from omegaconf import DictConfig, OmegaConf


def _get_sweep_configs(cfg: DictConfig, alloc_cores: int, cores_per_node: int = 24):
    """Generate valid experiment configs for this allocation tier.

    Filters sweep parameters to only include configs that:
    1. Fit within the allocated cores (n_ranks <= alloc_cores)
    2. Require this tier (n_ranks > prev_tier, avoiding duplicates)
    """
    sweep = cfg.get("sweep", {})
    if not sweep:
        # No sweep defined - return single config from cfg
        return [{"n_ranks": cfg.get("n_ranks", 1),
                 "N": cfg.get("N", 64),
                 "strategy": cfg.get("strategy", "sliced")}]

    prev_tier = alloc_cores - cores_per_node  # Previous allocation tier

    # Get sweep parameters with defaults
    rank_values = list(sweep.get("n_ranks", [cfg.get("n_ranks", 1)]))
    N_values = list(sweep.get("N", [cfg.get("N", 64)]))
    strategy_values = list(sweep.get("strategy", [cfg.get("strategy", "sliced")]))

    # Filter ranks: must fit in allocation AND require this tier
    valid_ranks = [r for r in rank_values if prev_tier < r <= alloc_cores]

    if not valid_ranks:
        return []

    # Generate all combinations
    configs = []
    for n_ranks, N, strategy in product(valid_ranks, N_values, strategy_values):
        configs.append({"n_ranks": n_ranks, "N": N, "strategy": strategy})

    return configs


@hydra.main(config_path="Experiments/hydra-conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Entry point - spawns MPI if needed, otherwise runs solver."""
    # Note: NUMBA_NUM_THREADS is set via hydra.job.env_set in config.yaml

    # Check if we're in an MPI subprocess (spawned by mpiexec)
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        if comm.Get_size() > 1 or os.environ.get("MPI_SUBPROCESS"):
            _run_solver(cfg, comm)
            return
    except ImportError:
        pass

    # %%
    # Job Array Configuration
    # -----------------------

    mpi = cfg.get("mpi", {})
    cores_per_node = mpi.get("cores_per_node", 24)
    alloc_cores = int(os.environ.get("LSB_DJOB_NUMPROC", cores_per_node))
    job_index = int(os.environ.get("LSB_JOBINDEX", 0))

    # Get valid configs for this allocation tier
    configs = _get_sweep_configs(cfg, alloc_cores, cores_per_node)

    if not configs:
        print(f"No valid configs for allocation={alloc_cores} cores")
        return

    if job_index > 0:
        # Job array mode: run config at this index
        if job_index > len(configs):
            print(f"Job index {job_index} > num configs {len(configs)}, exiting")
            return
        run_cfg = configs[job_index - 1]  # LSB_JOBINDEX is 1-based
    else:
        # Local/single mode: should not happen, but fallback to first
        print(f"Warning: No LSB_JOBINDEX set, running first config")
        run_cfg = configs[0]

    # Override cfg with selected config
    n_ranks = run_cfg["n_ranks"]
    cfg_override = OmegaConf.create(run_cfg)
    cfg = OmegaConf.merge(cfg, cfg_override)

    print(f"[Job {job_index}/{len(configs)}] n_ranks={n_ranks}, N={run_cfg['N']}, strategy={run_cfg['strategy']}")

    # %%
    # Spawn MPI Subprocess
    # --------------------

    script = os.path.abspath(__file__)

    # Set env var to prevent recursive spawning
    env = os.environ.copy()
    env["MPI_SUBPROCESS"] = "1"

    # Build MPI command with optional args
    cmd = ["mpiexec", "-n", str(n_ranks)]
    cmd.append("--report-binding")

    # Calculate NPS from allocation
    if mpi.get("bind_to"):
        n_nodes = alloc_cores // cores_per_node
        ranks_per_node = n_ranks // max(n_nodes, 1)
        ranks_per_socket = ranks_per_node // 2  # 2 sockets per node
        if ranks_per_socket > 0:
            cmd.extend(["--map-by", f"ppr:{ranks_per_socket}:package"])
        cmd.extend(["--bind-to", str(mpi.bind_to)])

    cmd.extend(["uv", "run", "python", script])
    # Pass the resolved config, not original argv
    cmd.extend([f"n_ranks={n_ranks}", f"N={run_cfg['N']}", f"strategy={run_cfg['strategy']}"])
    # Keep the config-name from original args
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg.startswith("-c") or arg.startswith("--config"):
            cmd.append(arg)
            # If flag and value are separate args, also append value
            if "=" not in arg and i + 1 < len(args):
                cmd.append(args[i + 1])
            break

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)


def _run_solver(cfg: DictConfig, comm):
    """MPI worker - runs solver and logs to MLflow."""

    # %%
    # Initialize
    # ----------

    from mpi4py import MPI
    from Poisson import JacobiPoisson, MultigridPoisson, get_project_root
    from utils.mlflow.io import (
        setup_mlflow_tracking,
        start_mlflow_run_context,
        log_parameters,
        log_metrics_dict,
        log_timeseries_metrics,
        log_artifact_file,
    )

    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    N = cfg.N
    solver_type = cfg.get("solver", "jacobi")
    strategy = cfg.get("strategy", "sliced")
    communicator = cfg.get("communicator", "custom")
    experiment_name = cfg.get("experiment_name", "experiment")

    if rank == 0:
        setup_mlflow_tracking(mode=cfg.mlflow.mode)
        print(f"\n{'='*60}")
        print(f"Experiment: {experiment_name}")
        print(f"Solver: {solver_type}, N={N}, ranks={n_ranks}")
        print(f"Strategy: {strategy}, Communicator: {communicator}")
        print(f"{'='*60}")

    # %%
    # Create Solver
    # -------------

    if solver_type == "jacobi":
        solver = JacobiPoisson(
            N=N,
            strategy=strategy,
            communicator=communicator,
            omega=cfg.get("omega", 0.8),
            tolerance=cfg.get("tolerance", 1e-6),
            max_iter=cfg.get("max_iter", 10000),
            use_numba=cfg.get("use_numba", False),
        )
    elif solver_type in ["multigrid", "fmg"]:
        solver = MultigridPoisson(
            N=N,
            n_smooth=cfg.get("n_smooth", 3),
            omega=cfg.get("omega", 2/3),
            decomposition_strategy=strategy,
            communicator=communicator,
            tolerance=cfg.get("tolerance", 1e-16),
            max_iter=cfg.get("max_iter", 100),
            fmg_post_cycles=cfg.get("fmg_post_vcycles", 1),
        )
    else:
        if rank == 0:
            print(f"Unknown solver: {solver_type}")
        sys.exit(1)

    # %%
    # Run Solver
    # ----------

    if solver_type == "fmg":
        solver.fmg_solve()  # FMG is 1 cycle by definition
    else:
        solver.solve()

    solver.compute_l2_error()

    # %%
    # Log to MLflow
    # -------------

    if rank == 0:
        run_name = f"{solver_type}_N{N}_p{n_ranks}_{strategy}"

        # Get local grid shape from solver config (set during grid creation)
        local_shape = solver.config.local_N
        # Use local_volume (product of dimensions) - shape-independent scalar
        import numpy as np
        local_volume = int(np.prod(local_shape)) if local_shape else (N // n_ranks) ** 3

        # Create label for communication plots (strategy + communicator)
        comm_type = "contiguous" if strategy == "sliced" else "mixed"
        label = f"{communicator.title()} ({strategy}, {comm_type})"

        with start_mlflow_run_context(
            experiment_name=experiment_name,
            parent_run_name=f"N{N}",
            child_run_name=run_name,
        ):
            log_parameters({
                "N": N,
                "local_volume": local_volume,
                "halo_size_mb": solver.config.halo_size_mb,
                "n_ranks": n_ranks,
                "solver": solver_type,
                "strategy": strategy,
                "communicator": communicator,
                "label": label,
                "omega": cfg.get("omega"),
                "max_iter": cfg.get("max_iter"),
                "tolerance": cfg.get("tolerance"),
            })

            # Log solver results
            log_metrics_dict(asdict(solver.results))

            # Log halo exchange timing statistics for communication analysis
            if hasattr(solver, "timeseries") and solver.timeseries.halo_exchange_times:
                import numpy as np
                halo_times = np.array(solver.timeseries.halo_exchange_times)
                log_metrics_dict({
                    "halo_time_mean_us": float(np.mean(halo_times) * 1e6),
                    "halo_time_std_us": float(np.std(halo_times) * 1e6),
                    "halo_time_min_us": float(np.min(halo_times) * 1e6),
                    "halo_time_max_us": float(np.max(halo_times) * 1e6),
                })

            if hasattr(solver, "timeseries"):
                log_timeseries_metrics(solver.timeseries)

        print(f"\n--- {solver_type.upper()} Complete ---")
        print(f"  Iterations: {solver.results.iterations}")
        print(f"  L2 error: {solver.results.final_error:.6e}")
        print(f"  Wall time: {solver.results.wall_time:.4f}s")
        if solver.results.mlups:
            print(f"  Performance: {solver.results.mlups:.2f} Mlup/s")
        if solver.results.bandwidth_gb_s:
            print(f"  Bandwidth: {solver.results.bandwidth_gb_s:.2f} GB/s")


if __name__ == "__main__":
    main()

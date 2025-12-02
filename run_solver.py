"""
Unified Solver Runner
=====================

Single runner for all solver experiments. Uses sequential solvers for
single-process runs, spawns MPI for parallel runs.

Solver selection:
- n_ranks=1: JacobiSolver or FMGSolver (no MPI overhead)
- n_ranks>1: JacobiMPISolver or FMGMPISolver (with MPI)

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

import os
import subprocess
import sys
from dataclasses import asdict
from itertools import product

import hydra
from omegaconf import DictConfig, OmegaConf


def _get_sweep_configs(cfg: DictConfig, alloc_cores: int, cores_per_node: int = 24):
    """Generate valid experiment configs for this allocation tier.

    Supports two formats:
    1. sweep: {n_ranks: [...], N: [...], strategy: [...]} - all combinations
    2. sweep_groups: {name: {solver_type: x, N: [...]}} - group-specific sweeps
    """
    prev_tier = alloc_cores - cores_per_node

    # Check for sweep_groups first (new format)
    sweep_groups = cfg.get("sweep_groups", {})
    if sweep_groups:
        configs = []
        base_n_ranks = cfg.get("n_ranks", 1)
        base_strategy = cfg.get("strategy", "sliced")

        base_communicator = cfg.get("communicator", "custom")

        for group_name, group_cfg in sweep_groups.items():
            solver_type = group_cfg.get("solver_type", "jacobi")
            N_values = list(group_cfg.get("N", [65]))
            n_ranks_values = list(group_cfg.get("n_ranks", [base_n_ranks]))
            strategy_values = list(group_cfg.get("strategy", [base_strategy]))
            communicator_values = list(group_cfg.get("communicator", [base_communicator]))

            # Filter ranks for this allocation tier
            valid_ranks = [r for r in n_ranks_values if prev_tier < r <= alloc_cores]
            if not valid_ranks:
                valid_ranks = [base_n_ranks] if prev_tier < base_n_ranks <= alloc_cores else []

            for n_ranks in valid_ranks:
                for N in N_values:
                    for strategy in strategy_values:
                        for communicator in communicator_values:
                            configs.append({
                                "n_ranks": n_ranks,
                                "N": N,
                                "strategy": strategy,
                                "solver_type": solver_type,
                                "communicator": communicator,
                            })
        return configs

    # Fall back to sweep format
    sweep = cfg.get("sweep", {})
    if not sweep:
        # No sweep defined - return single config from cfg
        return [{"n_ranks": cfg.get("n_ranks", 1),
                 "N": cfg.get("N", 64),
                 "strategy": cfg.get("strategy", "sliced"),
                 "solver_type": cfg.get("solver_type", "jacobi")}]

    # Get sweep parameters with defaults
    rank_values = list(sweep.get("n_ranks", [cfg.get("n_ranks", 1)]))
    N_values = list(sweep.get("N", [cfg.get("N", 64)]))
    strategy_values = list(sweep.get("strategy", [cfg.get("strategy", "sliced")]))
    solver_values = list(sweep.get("solver_type", [cfg.get("solver_type", "jacobi")]))

    # Filter ranks: must fit in allocation AND require this tier
    valid_ranks = [r for r in rank_values if prev_tier < r <= alloc_cores]

    if not valid_ranks:
        return []

    # Generate all combinations
    configs = []
    for n_ranks, N, strategy, solver_type in product(valid_ranks, N_values, strategy_values, solver_values):
        configs.append({"n_ranks": n_ranks, "N": N, "strategy": strategy, "solver_type": solver_type})

    return configs


@hydra.main(config_path="Experiments/hydra-conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Entry point - runs sequential or spawns MPI based on n_ranks."""

    # Check if we're in an MPI subprocess (spawned by mpiexec)
    if os.environ.get("MPI_SUBPROCESS"):
        from mpi4py import MPI
        _run_mpi_solver(cfg, MPI.COMM_WORLD)
        return

    # Job Array Configuration
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
        configs_to_run = [configs[job_index - 1]]  # LSB_JOBINDEX is 1-based
    else:
        # Local mode: run all configs
        print(f"No LSB_JOBINDEX set, running all {len(configs)} configs")
        configs_to_run = configs

    for i, run_cfg in enumerate(configs_to_run):
        n_ranks = run_cfg["n_ranks"]
        merged_cfg = OmegaConf.merge(cfg, OmegaConf.create(run_cfg))

        solver = run_cfg.get('solver_type', 'jacobi')
        print(f"\n[Run {i+1}/{len(configs_to_run)}] {solver}, N={run_cfg['N']}, n_ranks={n_ranks}")

        # Sequential vs MPI
        if n_ranks == 1:
            _run_sequential_solver(merged_cfg)
        else:
            _spawn_mpi(merged_cfg, n_ranks, alloc_cores, cores_per_node, mpi)


def _spawn_mpi(cfg: DictConfig, n_ranks: int, alloc_cores: int, cores_per_node: int, mpi: dict):
    """Spawn MPI subprocess for parallel execution."""
    script = os.path.abspath(__file__)

    env = os.environ.copy()
    env["MPI_SUBPROCESS"] = "1"

    # Build MPI command
    cmd = ["mpiexec", "-n", str(n_ranks), "--report-binding"]

    # Calculate process mapping from allocation
    if mpi.get("bind_to"):
        n_nodes = alloc_cores // cores_per_node
        ranks_per_node = n_ranks // max(n_nodes, 1)
        ranks_per_socket = ranks_per_node // 2  # 2 sockets per node
        if ranks_per_socket > 0:
            cmd.extend(["--map-by", f"ppr:{ranks_per_socket}:package"])
        cmd.extend(["--bind-to", str(mpi.bind_to)])

    cmd.extend(["uv", "run", "python", script])

    # Pass all relevant config values to subprocess
    overrides = [
        f"n_ranks={n_ranks}",
        f"N={cfg.N}",
        f"strategy={cfg.strategy}",
        f"solver_type={cfg.get('solver_type', 'jacobi')}",
        f"communicator={cfg.get('communicator', 'custom')}",
        f"omega={cfg.get('omega', 0.8)}",
        f"max_iter={cfg.get('max_iter', 1000)}",
        f"tolerance={cfg.get('tolerance', 1e-6)}",
    ]
    if cfg.get("experiment_name"):
        overrides.append(f"experiment_name={cfg.experiment_name}")
    if cfg.get("use_numba"):
        overrides.append(f"use_numba={cfg.use_numba}")
    if cfg.get("n_smooth"):
        overrides.append(f"n_smooth={cfg.n_smooth}")
    if cfg.get("fmg_post_vcycles"):
        overrides.append(f"fmg_post_vcycles={cfg.fmg_post_vcycles}")

    cmd.extend(overrides)

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)


def _run_sequential_solver(cfg: DictConfig):
    """Run sequential solver (no MPI)."""
    from Poisson.solvers import JacobiSolver, FMGSolver
    from utils.mlflow.io import (
        setup_mlflow_tracking,
        start_mlflow_run_context,
        log_parameters,
        log_metrics_dict,
        log_timeseries_metrics,
    )

    N = cfg.N
    solver_type = cfg.get("solver_type", "jacobi")
    experiment_name = cfg.get("experiment_name", "experiment")

    setup_mlflow_tracking(mode=cfg.mlflow.mode)
    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"Solver: {solver_type} (sequential), N={N}")
    print(f"{'='*60}")

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
        print(f"Unknown solver: {solver_type}")
        sys.exit(1)

    # Warmup and solve
    solver.warmup()
    if solver_type == "fmg":
        solver.fmg_solve()
    else:
        solver.solve()

    solver.compute_l2_error()

    # Log to MLflow
    _log_results(cfg, solver, solver_type, N, n_ranks=1)


def _run_mpi_solver(cfg: DictConfig, comm):
    """Run MPI solver (called within mpiexec subprocess)."""
    from mpi4py import MPI
    from Poisson.solvers import JacobiMPISolver, FMGMPISolver
    from utils.mlflow.io import (
        setup_mlflow_tracking,
        start_mlflow_run_context,
        log_parameters,
        log_metrics_dict,
        log_timeseries_metrics,
    )

    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    N = cfg.N
    solver_type = cfg.get("solver_type", "jacobi")
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
        )
    else:
        if rank == 0:
            print(f"Unknown solver: {solver_type}")
        sys.exit(1)

    # Solve
    if solver_type == "fmg":
        solver.fmg_solve()
    else:
        solver.solve()

    solver.compute_l2_error()

    # Log to MLflow (rank 0 only)
    if rank == 0:
        _log_results(cfg, solver, solver_type, N, n_ranks, strategy, communicator)


def _log_results(cfg, solver, solver_type, N, n_ranks, strategy=None, communicator=None):
    """Log solver results to MLflow."""
    import numpy as np
    from dataclasses import asdict
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

        log_parameters(params)
        log_metrics_dict(asdict(solver.results))

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

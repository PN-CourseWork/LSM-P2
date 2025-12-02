"""
Unified Solver Runner
=====================

Single runner for all solver-based experiments. Spawns MPI subprocess
with rank count from Hydra config. Results logged to MLflow.

Usage
-----

.. code-block:: bash

    # Single run
    uv run python run_solver.py --config-name=experiment/04-validation

    # Override parameters
    uv run python run_solver.py --config-name=experiment/04-validation N=64 n_ranks=8

    # Hydra multirun sweep
    uv run python run_solver.py --config-name=experiment/04-validation \\
        --multirun N=16,32,48 strategy=sliced,cubic
"""

# %%
# Setup
# -----

import os
import subprocess
import sys
from dataclasses import asdict

import hydra
from omegaconf import DictConfig, OmegaConf


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
    # Spawn MPI Subprocess
    # --------------------

    n_ranks = cfg.get("n_ranks", 1)
    script = os.path.abspath(__file__)

    # Set env var to prevent recursive spawning
    env = os.environ.copy()
    env["MPI_SUBPROCESS"] = "1"

    # Build MPI command with optional args
    cmd = ["mpiexec", "-n", str(n_ranks)]

    mpi = cfg.get("mpi", {})
    cmd.append("--report-binding")

    # Calculate NPS from LSF allocation if available
    alloc_cores = os.environ.get("LSB_DJOB_NUMPROC")
    if alloc_cores and mpi.get("bind_to"):
        cores_per_node = mpi.get("cores_per_node", 24)
        n_nodes = int(alloc_cores) // cores_per_node
        ranks_per_node = n_ranks // max(n_nodes, 1)
        ranks_per_socket = ranks_per_node // 2  # 2 sockets per node
        if ranks_per_socket > 0:
            cmd.extend(["--map-by", f"ppr:{ranks_per_socket}:package"])

    if mpi.get("bind_to"):
        cmd.extend(["--bind-to", str(mpi.bind_to)])

    cmd.extend(["uv", "run", "python", script])
    cmd.extend(sys.argv[1:])

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    # Log output as MLflow artifact
    if cfg.get("experiment_name") and (result.stdout or result.stderr):
        from Poisson import get_project_root
        from utils.mlflow.io import setup_mlflow_tracking
        import mlflow

        setup_mlflow_tracking(mode=cfg.mlflow.mode)

        log_dir = get_project_root() / "logs" / "runs"
        log_dir.mkdir(parents=True, exist_ok=True)

        run_id = f"{cfg.experiment_name}_N{cfg.N}_p{n_ranks}"
        log_file = log_dir / f"{run_id}.log"

        with open(log_file, "w") as f:
            if result.stdout:
                f.write(result.stdout)
            if result.stderr:
                f.write("\n--- STDERR ---\n")
                f.write(result.stderr)

        try:
            mlflow.set_experiment(cfg.experiment_name)
            runs = mlflow.search_runs(max_results=1, order_by=["start_time DESC"])
            if not runs.empty:
                run_id = runs.iloc[0]["run_id"]
                with mlflow.start_run(run_id=run_id):
                    mlflow.log_artifact(str(log_file), artifact_path="logs")
        except Exception:
            pass


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

"""
Unified Solver Runner
=====================

Single runner for all solver-based experiments. Spawns MPI subprocess
with rank count from Hydra config. Results logged to MLflow.

Usage
-----

.. code-block:: bash

    # Single run
    uv run python run_solver.py --config-name=04-validation

    # Override parameters
    uv run python run_solver.py --config-name=04-validation N=64 n_ranks=8

    # Hydra multirun sweep
    uv run python run_solver.py --config-name=04-validation \\
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

    # Set numba threads from config
    if cfg.get("numba_threads"):
        os.environ["NUMBA_NUM_THREADS"] = str(cfg.numba_threads)

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        if comm.Get_size() > 1:
            _run_solver(cfg, comm)
            return
    except ImportError:
        pass

    # %%
    # Spawn MPI Subprocess
    # --------------------

    n_ranks = cfg.get("n_ranks", 1)
    script = os.path.abspath(__file__)

    cmd = [
        "mpiexec", "-n", str(n_ranks),
        "--report-bindings",
        "uv", "run", "python", script,
    ]
    cmd.extend(sys.argv[1:])

    result = subprocess.run(cmd, capture_output=True, text=True)

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
        )
    else:
        if rank == 0:
            print(f"Unknown solver: {solver_type}")
        sys.exit(1)

    # %%
    # Run Solver
    # ----------

    comm.Barrier()
    t0 = MPI.Wtime()

    if solver_type == "fmg":
        cycles = cfg.get("fmg_cycles", 1)
        solver.fmg_solve(cycles=cycles)
    else:
        solver.solve()

    wall_time = MPI.Wtime() - t0

    if rank == 0:
        solver.results.wall_time = wall_time
        n_interior = (N - 2) ** 3
        if solver.results.iterations > 0:
            solver.results.mlups = n_interior * solver.results.iterations / (wall_time * 1e6)

    solver.compute_l2_error()

    # %%
    # Log to MLflow
    # -------------

    if rank == 0:
        run_name = f"{solver_type}_N{N}_p{n_ranks}_{strategy}"

        with start_mlflow_run_context(
            experiment_name=experiment_name,
            parent_run_name=f"N{N}",
            child_run_name=run_name,
        ):
            log_parameters({
                "N": N,
                "n_ranks": n_ranks,
                "solver": solver_type,
                "strategy": strategy,
                "communicator": communicator,
                "omega": cfg.get("omega"),
                "max_iter": cfg.get("max_iter"),
                "tolerance": cfg.get("tolerance"),
            })
            log_metrics_dict(asdict(solver.results))

            if hasattr(solver, "timeseries"):
                log_timeseries_metrics(solver.timeseries)

        print(f"\n--- {solver_type.upper()} Complete ---")
        print(f"  Iterations: {solver.results.iterations}")
        print(f"  L2 error: {solver.results.final_error:.6e}")
        print(f"  Wall time: {wall_time:.4f}s")
        if hasattr(solver.results, "mlups") and solver.results.mlups:
            print(f"  Performance: {solver.results.mlups:.2f} Mlup/s")


if __name__ == "__main__":
    main()

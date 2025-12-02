"""
Communication Method Benchmark
==============================

Compare NumPy vs Custom MPI datatype halo exchange for contiguous (Z-axis)
and non-contiguous (X-axis) memory layouts.

Uses per-iteration timeseries data for statistical analysis.
All data is logged to MLflow for retrieval by plotting scripts.
"""

import subprocess
import numpy as np
import hydra
from omegaconf import DictConfig

from Poisson import get_project_root
from utils.mlflow.io import (
    setup_mlflow_tracking,
    start_mlflow_run_context,
    log_parameters,
    log_metrics_dict,
)


@hydra.main(config_path="../hydra-conf", config_name="03-communication", version_base=None)
def main(cfg: DictConfig):
    """Entry point - spawns MPI if needed."""
    try:
        from mpi4py import MPI

        if MPI.COMM_WORLD.Get_size() > 1:
            _run_benchmark(cfg)
            return
    except ImportError:
        pass

    # Spawn MPI
    script = (
        get_project_root()
        / "Experiments"
        / "03-communication"
        / "compute_communication.py"
    )
    subprocess.run(["mpiexec", "-n", "4", "uv", "run", "python", str(script)])


def _run_benchmark(cfg: DictConfig):
    """MPI worker - collects per-iteration timings and logs to MLflow."""
    from mpi4py import MPI
    from Poisson import JacobiPoisson

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Config from Hydra
    PROBLEM_SIZES = list(cfg.problem_sizes)
    ITERATIONS = cfg.iterations
    WARMUP = cfg.warmup

    # Test configurations: strategy x communicator
    CONFIGS = [
        ("sliced", "numpy", "NumPy (sliced, contiguous)"),
        ("sliced", "custom", "Custom (sliced, contiguous)"),
        ("cubic", "numpy", "NumPy (cubic, mixed)"),
        ("cubic", "custom", "Custom (cubic, mixed)"),
    ]

    if rank == 0:
        setup_mlflow_tracking(mode=cfg.mlflow.mode)
        print("Communication Benchmark: Per-Iteration Timings")
        print(f"Ranks: {size}, Iterations: {ITERATIONS}, Warmup: {WARMUP}")
        print("=" * 60)

    for N in PROBLEM_SIZES:
        if rank == 0:
            print(f"\nN={N}")

        for strategy, comm_type, label in CONFIGS:
            if rank == 0:
                print(f"  {label}...", end=" ", flush=True)

            # Create and run solver
            solver = JacobiPoisson(
                N=N,
                strategy=strategy,
                communicator=comm_type,
                max_iter=WARMUP + ITERATIONS,
                tolerance=0,
            )
            solver.solve()

            # Get max halo time across ranks per iteration (skip warmup)
            local_times = solver.timeseries.halo_exchange_times[WARMUP:]
            max_times = comm.allreduce(local_times, op=MPI.MAX)

            if rank == 0:
                local_N = N // size
                halo_times_us = [t * 1e6 for t in max_times]
                mean_time = np.mean(halo_times_us)
                std_time = np.std(halo_times_us)

                print(f"mean={mean_time:.1f} μs/iter")

                # Log to MLflow
                run_name = f"N{N}_{strategy}_{comm_type}"
                with start_mlflow_run_context(
                    experiment_name=cfg.experiment_name or "03-communication",
                    parent_run_name=f"np{size}",
                    child_run_name=run_name,
                ):
                    log_parameters({
                        "N": N,
                        "local_N": local_N,
                        "strategy": strategy,
                        "communicator": comm_type,
                        "label": label,
                        "n_ranks": size,
                        "iterations": ITERATIONS,
                        "warmup": WARMUP,
                    })
                    log_metrics_dict({
                        "halo_time_mean_us": mean_time,
                        "halo_time_std_us": std_time,
                        "halo_time_min_us": np.min(halo_times_us),
                        "halo_time_max_us": np.max(halo_times_us),
                    })

    if rank == 0:
        print("\nAll results logged to MLflow.")


if __name__ == "__main__":
    main()

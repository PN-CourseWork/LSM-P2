"""
Communication Method Benchmark
==============================

Compare NumPy vs Custom MPI datatype halo exchange for contiguous (Z-axis)
and non-contiguous (X-axis) memory layouts.

Uses per-iteration timeseries data for statistical analysis.
"""

import subprocess
import mlflow

from Poisson import get_project_root
from utils.mlflow.io import setup_mlflow_tracking


def main():
    """Entry point - spawns MPI if needed."""
    try:
        from mpi4py import MPI

        if MPI.COMM_WORLD.Get_size() > 1:
            _run_benchmark()
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


def _run_benchmark():
    """MPI worker - collects per-iteration timings."""
    import pandas as pd
    from mpi4py import MPI
    from Poisson import JacobiPoisson, get_project_root

    # --- MLflow Setup ---
    # Initialize MLflow tracking from environment variables
    # Must be called by all ranks to ensure proper MPI barrier synchronization if needed
    #setup_mlflow_tracking()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Config
    PROBLEM_SIZES = [32, 48, 64, 80, 100]
    ITERATIONS = 500  # Per-iteration data gives us 500 samples each
    WARMUP = 50

    # Test configurations: strategy x communicator
    # - sliced: 1D decomposition along z-axis (contiguous memory access)
    # - cubic: 3D decomposition (includes non-contiguous x/y communication)
    CONFIGS = [
        ("sliced", "numpy", "NumPy (sliced, contiguous)"),
        ("sliced", "custom", "Custom (sliced, contiguous)"),
        ("cubic", "numpy", "NumPy (cubic, mixed)"),
        ("cubic", "custom", "Custom (cubic, mixed)"),
    ]

    repo_root = get_project_root()
    data_dir = repo_root / "data" / "03-communication"
    data_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        print("Communication Benchmark: Per-Iteration Timings")
        print(f"Ranks: {size}, Iterations: {ITERATIONS}, Warmup: {WARMUP}")
        print("=" * 60)

    dfs = []

    for N in PROBLEM_SIZES:
        if rank == 0:
            print(f"\nN={N}")

        for strategy, comm_type, label in CONFIGS:
            if rank == 0:
                print(f"  {label}...", end=" ", flush=True)

            # Create and run solver using unified DistributedGrid API
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
                local_N = N // size  # Local subdomain size along decomposed axis
                print(f"mean={sum(max_times) / len(max_times) * 1e6:.1f} μs/iter")
                dfs.append(
                    pd.DataFrame(
                        {
                            "N": N,
                            "local_N": local_N,
                            "strategy": strategy,
                            "communicator": comm_type,
                            "label": label,
                            "iteration": range(len(max_times)),
                            "halo_time_us": [t * 1e6 for t in max_times],
                        }
                    )
                )
"""
    if rank == 0:
        df = pd.concat(dfs, ignore_index=True)
        output_file = data_dir / f"communication_np{size}.parquet"
        df.to_parquet(output_file, index=False)
        print(f"\nSaved {len(df)} measurements to: {output_file}")

        # --- MLflow Logging ---
        # To disable MLflow logging, comment out the following lines.
        try:
            mlflow.set_experiment("/Shared/LSM-PoissonMPI/Experiment-03-Communication")
            with mlflow.start_run(run_name=f"Communication-Data-np{size}") as run:
                print(f"INFO: Started MLflow run '{run.info.run_name}' for artifact logging.")
                
                mlflow.log_param("ranks", size)
                mlflow.log_param("problem_sizes", PROBLEM_SIZES)
                mlflow.log_param("iterations", ITERATIONS)

                if output_file.exists():
                    mlflow.log_artifact(str(output_file))
                    print(f"  ✓ Logged artifact: {output_file.name}")
        
        except Exception as e:
            print(f"  ✗ WARNING: MLflow logging failed: {e}")

"""
if __name__ == "__main__":
    main()

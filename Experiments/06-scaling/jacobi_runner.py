"""
Poisson Scaling Experiment
==========================

MPI Jacobi solver for 3D Poisson equation with MLflow logging.

This script is responsible for orchestrating the experiment, including
setting up MLflow, running the solver, and logging the results.

Usage:
    # Direct CLI args:
    mpiexec -n 4 uv run python jacobi_runner.py --N 64 --numba

    # Job array mode (reads LSB_JOBINDEX):
    mpiexec -n 8 uv run python jacobi_runner.py \
        --runtime-config HPC/Configs/runtime/experiments.yaml \
        --experiment jacobi_strong --numba

    # Local test with specific index:
    uv run python jacobi_runner.py \
        --runtime-config HPC/Configs/runtime/experiments.yaml \
        --experiment jacobi_strong --index 1 --numba
"""

import argparse
import os
from dataclasses import asdict
from mpi4py import MPI

# Project-specific imports
from Poisson import JacobiPoisson, get_project_root
from utils.mlflow.io import (
    setup_mlflow_tracking,
    start_mlflow_run_context,
    log_parameters,
    log_metrics_dict,
    log_timeseries_metrics,
    log_artifact_file,
    log_lsf_logs,
)
from runtime_config import load_runtime_config


def parse_args():
    parser = argparse.ArgumentParser(description="MPI Jacobi solver for 3D Poisson equation")

    # Runtime config mode
    parser.add_argument("--runtime-config", type=str, help="Path to runtime YAML config")
    parser.add_argument("--experiment", type=str, help="Experiment name in config")
    parser.add_argument("--index", type=int, help="Task index (default: LSB_JOBINDEX)")

    # Direct CLI args (used if no runtime-config)
    parser.add_argument("--N", type=int, default=16, help="Grid size N³")
    parser.add_argument("--tol", type=float, default=0.0, help="Convergence tolerance")
    parser.add_argument("--max-iter", type=int, default=5, help="Max iterations")
    parser.add_argument("--omega", type=float, default=0.8, help="Relaxation parameter")
    parser.add_argument("--strategy", choices=["sliced", "cubic"], default="sliced", help="Decomposition strategy")
    parser.add_argument("--communicator", choices=["numpy", "custom"], default="numpy", help="Halo exchange communicator")
    parser.add_argument("--numba", action="store_true", help="Use Numba kernel")

    # Logging options
    parser.add_argument("--job-name", type=str, default=None, help="LSF Job Name for log retrieval")
    parser.add_argument("--log-dir", type=str, default="logs/lsf", help="Directory for LSF logs")
    parser.add_argument("--experiment-name", type=str, default=None, help="MLflow experiment name")

    return parser.parse_args()


def resolve_params(args):
    """Resolve parameters from either runtime config or CLI args."""
    if args.runtime_config and args.experiment:
        # Load from config file
        cfg = load_runtime_config(args.runtime_config, args.experiment, args.index)

        # Handle numba_threads from config
        if "numba_threads" in cfg:
            os.environ["NUMBA_NUM_THREADS"] = str(cfg.pop("numba_threads"))

        return {
            "N": cfg.get("N", args.N),
            "tol": cfg.get("tol", args.tol),
            "max_iter": cfg.get("max_iter", args.max_iter),
            "omega": cfg.get("omega", args.omega),
            "strategy": cfg.get("strategy", args.strategy),
            "communicator": cfg.get("communicator", args.communicator),
        }
    else:
        # Use CLI args directly
        return {
            "N": args.N,
            "tol": args.tol,
            "max_iter": args.max_iter,
            "omega": args.omega,
            "strategy": args.strategy,
            "communicator": args.communicator,
        }


# --- Main ---
args = parse_args()
params = resolve_params(args)

# --- MPI and Setup ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_ranks = comm.Get_size()

# --- Solver Configuration ---
solver = JacobiPoisson(
    N=params["N"],
    strategy=params["strategy"],
    communicator=params["communicator"],
    omega=params["omega"],
    tolerance=params["tol"],
    max_iter=params["max_iter"],
    use_numba=args.numba,
)

if rank == 0:
    print("INFO: Setting up MLflow tracking...")
    setup_mlflow_tracking()
    print(f"Solver configured: N={params['N']}³, ranks={n_ranks}, strategy={params['strategy']}, comm={params['communicator']}")
    print(f"  Kernel: {'Numba' if args.numba else 'NumPy'}, omega={params['omega']}")
    if args.runtime_config:
        idx = args.index or int(os.environ.get("LSB_JOBINDEX", "1"))
        print(f"  Config: {args.experiment} task {idx}")

# --- Solver Execution ---
if args.numba:
    solver.warmup()

# Synchronize before timing
if n_ranks > 1:
    comm.Barrier()

t0 = MPI.Wtime()
solver.solve()
wall_time = MPI.Wtime() - t0

# --- Post-processing ---
if rank == 0:
    solver.results.wall_time = wall_time
    n_interior = (params["N"] - 2) ** 3
    total_updates = n_interior * solver.results.iterations
    solver.results.mlups = total_updates / (wall_time * 1e6)
solver.compute_l2_error()

# --- Logging and Summary on Rank 0 only ---
if rank == 0:
    project_root = get_project_root()
    data_dir = project_root / "data" / "06-scaling"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_file = data_dir / f"poisson_N{params['N']}_p{n_ranks}_{params['strategy']}_{params['communicator']}.h5"
    solver.save_hdf5(output_file)

    # --- MLflow Logging ---
    experiment_name = args.experiment_name or args.experiment or "Experiment-06-Scaling"
    parent_run_name = f"N{params['N']}"
    run_name = f"N{params['N']}_p{n_ranks}_{params['strategy']}_{params['communicator']}"

    with start_mlflow_run_context(experiment_name, parent_run_name, run_name, args=args):
        log_parameters(asdict(solver.config))
        log_metrics_dict(asdict(solver.results))
        log_timeseries_metrics(solver.timeseries)
        log_artifact_file(output_file)
        log_lsf_logs(args.job_name, args.log_dir)

    # --- Final Summary ---
    print("\n--- Run Complete ---")
    print(f"Results saved to: {output_file}")
    print("\nSolution Status:")
    print(f"  Converged: {solver.results.converged}")
    print(f"  Iterations: {solver.results.iterations}")
    print(f"  L2 error: {solver.results.final_error:.6e}")
    print(f"  Wall time: {solver.results.wall_time:.2f} seconds")
    print(f"  Performance: {solver.results.mlups:.2f} Mlup/s")

    total_time = (solver.results.total_compute_time or 0) + (solver.results.total_halo_time or 0) + (solver.results.total_mpi_comm_time or 0)
    if total_time > 0:
        print("\nTiming breakdown:")
        print(f"  Compute:      {solver.results.total_compute_time or 0:.3f}s ({100 * (solver.results.total_compute_time or 0) / total_time:.1f}%)")
        print(f"  Halo exchange:{solver.results.total_halo_time or 0:.3f}s ({100 * (solver.results.total_halo_time or 0) / total_time:.1f}%)")
        print(f"  MPI allreduce:{solver.results.total_mpi_comm_time or 0:.3f}s ({100 * (solver.results.total_mpi_comm_time or 0) / total_time:.1f}%)")

# Final barrier for clean exit
if n_ranks > 1:
    comm.Barrier()

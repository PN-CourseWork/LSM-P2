"""
FMG Scaling Experiment
======================

MPI Full Multigrid (FMG) solver for 3D Poisson equation with MLflow logging.

This script runs FMG scaling experiments with configurable decomposition
and communication strategies.

Usage:
    # Direct CLI args:
    mpiexec -n 8 uv run python fmg_runner.py --N 65 --cycles 1

    # Job array mode (reads LSB_JOBINDEX):
    mpiexec -n 27 uv run python fmg_runner.py \
        --runtime-config HPC/Configs/runtime/experiments.yaml \
        --experiment fmg_strong

    # Local test with specific index:
    uv run python fmg_runner.py \
        --runtime-config HPC/Configs/runtime/experiments.yaml \
        --experiment fmg_hybrid --index 5
"""

import argparse
import os
from dataclasses import asdict
from mpi4py import MPI

# Project-specific imports
from Poisson import (
    MultigridPoisson,
    get_project_root,
)
from utils.mlflow.io import (
    setup_mlflow_tracking,
    start_mlflow_run_context,
    log_parameters,
    log_metrics_dict,
    log_artifact_file,
    log_lsf_logs,
)
from runtime_config import load_runtime_config


def parse_args():
    parser = argparse.ArgumentParser(description="MPI FMG solver for 3D Poisson equation")

    # Runtime config mode
    parser.add_argument("--runtime-config", type=str, help="Path to runtime YAML config")
    parser.add_argument("--experiment", type=str, help="Experiment name in config")
    parser.add_argument("--index", type=int, help="Task index (default: LSB_JOBINDEX)")

    # Direct CLI args (used if no runtime-config)
    parser.add_argument("--N", type=int, default=65, help="Grid size N³ (should be 2^k + 1)")
    parser.add_argument("--cycles", type=int, default=1, help="Number of FMG cycles")
    parser.add_argument("--n-smooth", type=int, default=3, help="Smoothing steps per level")
    parser.add_argument("--omega", type=float, default=2/3, help="Relaxation parameter")
    parser.add_argument("--strategy", choices=["sliced", "cubic"], default="cubic", help="Decomposition strategy")
    parser.add_argument("--communicator", choices=["numpy", "custom"], default="custom", help="Halo exchange communicator")

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

        # Handle numba_threads from config (for hybrid experiments)
        if "numba_threads" in cfg:
            os.environ["NUMBA_NUM_THREADS"] = str(cfg.pop("numba_threads"))

        return {
            "N": cfg.get("N", args.N),
            "cycles": cfg.get("cycles", args.cycles),
            "n_smooth": cfg.get("n_smooth", args.n_smooth),
            "omega": cfg.get("omega", args.omega),
            "strategy": cfg.get("strategy", args.strategy),
            "communicator": cfg.get("communicator", args.communicator),
        }
    else:
        # Use CLI args directly
        return {
            "N": args.N,
            "cycles": args.cycles,
            "n_smooth": args.n_smooth,
            "omega": args.omega,
            "strategy": args.strategy,
            "communicator": args.communicator,
        }


# --- Main ---
args = parse_args()
params = resolve_params(args)

# --- MPI Setup ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_ranks = comm.Get_size()

# --- Solver Configuration ---
solver = MultigridPoisson(
    N=params["N"],
    omega=params["omega"],
    n_smooth=params["n_smooth"],
    decomposition_strategy=params["strategy"],
    communicator=params["communicator"],
    tolerance=1e-16,  # FMG doesn't iterate to convergence
    max_iter=params["cycles"],  # Used for tracking
)

if rank == 0:
    print("INFO: Setting up MLflow tracking...")
    setup_mlflow_tracking()
    print(f"FMG configured: N={params['N']}³, ranks={n_ranks}, strategy={params['strategy']}, comm={params['communicator']}")
    print(f"  Cycles: {params['cycles']}, n_smooth={params['n_smooth']}, omega={params['omega']}")
    print(f"  Multigrid levels: {solver.levels}")
    if args.runtime_config:
        idx = args.index or int(os.environ.get("LSB_JOBINDEX", "1"))
        print(f"  Config: {args.experiment} task {idx}")
        numba_threads = os.environ.get("NUMBA_NUM_THREADS", "1")
        print(f"  NUMBA_NUM_THREADS: {numba_threads}")

# --- Solver Execution ---
# Synchronize before timing
if n_ranks > 1:
    comm.Barrier()

t0 = MPI.Wtime()
solver.fmg_solve(cycles=params["cycles"])
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
    data_dir = project_root / "data" / "06-scaling" / "fmg"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_file = data_dir / f"fmg_N{params['N']}_p{n_ranks}_{params['strategy']}_{params['communicator']}.h5"
    solver.save_hdf5(output_file)

    # --- MLflow Logging ---
    experiment_name = args.experiment_name or args.experiment or "Experiment-06-Scaling-FMG"
    parent_run_name = f"FMG_N{params['N']}"
    run_name = f"FMG_N{params['N']}_p{n_ranks}_{params['strategy']}"

    with start_mlflow_run_context(experiment_name, parent_run_name, run_name, args=args):
        log_parameters({
            "N": params["N"],
            "cycles": params["cycles"],
            "n_smooth": params["n_smooth"],
            "omega": params["omega"],
            "strategy": params["strategy"],
            "communicator": params["communicator"],
            "n_ranks": n_ranks,
            "levels": solver.levels,
            "numba_threads": os.environ.get("NUMBA_NUM_THREADS", "1"),
        })
        log_metrics_dict(asdict(solver.results))
        log_artifact_file(output_file)
        log_lsf_logs(args.job_name, args.log_dir)

    # --- Final Summary ---
    print("\n--- FMG Run Complete ---")
    print(f"Results saved to: {output_file}")
    print("\nSolution Status:")
    print(f"  Converged: {solver.results.converged}")
    print(f"  Iterations: {solver.results.iterations}")
    print(f"  L2 error: {solver.results.final_error:.6e}")
    print(f"  Wall time: {solver.results.wall_time:.4f} seconds")
    print(f"  Performance: {solver.results.mlups:.2f} Mlup/s")

    total_time = (solver.results.total_compute_time or 0) + (solver.results.total_halo_time or 0) + (solver.results.total_mpi_comm_time or 0)
    if total_time > 0:
        print("\nTiming breakdown:")
        print(f"  Compute:      {solver.results.total_compute_time or 0:.4f}s ({100 * (solver.results.total_compute_time or 0) / total_time:.1f}%)")
        print(f"  Halo exchange:{solver.results.total_halo_time or 0:.4f}s ({100 * (solver.results.total_halo_time or 0) / total_time:.1f}%)")
        print(f"  MPI allreduce:{solver.results.total_mpi_comm_time or 0:.4f}s ({100 * (solver.results.total_mpi_comm_time or 0) / total_time:.1f}%)")

# Final barrier for clean exit
if n_ranks > 1:
    comm.Barrier()

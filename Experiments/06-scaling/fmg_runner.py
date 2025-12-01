"""
FMG Scaling Experiment
======================

MPI Full Multigrid (FMG) solver for 3D Poisson equation with MLflow logging.

This script runs FMG scaling experiments with configurable decomposition
and communication strategies.

Usage:
    mpiexec -n 8 uv run python fmg_runner.py --N 65 --cycles 1
    mpiexec -n 27 uv run python fmg_runner.py --N 129 --strategy cubic --communicator custom
"""

import argparse
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
)

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="MPI FMG solver for 3D Poisson equation")
parser.add_argument("--N", type=int, default=65, help="Grid size N³ (should be 2^k + 1)")
parser.add_argument("--cycles", type=int, default=1, help="Number of FMG cycles")
parser.add_argument("--n-smooth", type=int, default=3, help="Smoothing steps per level")
parser.add_argument("--omega", type=float, default=2/3, help="Relaxation parameter")
parser.add_argument("--strategy", choices=["sliced", "cubic"], default="cubic", help="Decomposition strategy")
parser.add_argument("--communicator", choices=["numpy", "custom"], default="custom", help="Halo exchange communicator")
parser.add_argument("--job-name", type=str, default=None, help="LSF Job Name for log retrieval")
parser.add_argument("--experiment-name", type=str, default=None, help="MLflow experiment name")
args = parser.parse_args()

# --- MPI Setup ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_ranks = comm.Get_size()

# --- Solver Configuration ---
solver = MultigridPoisson(
    N=args.N,
    omega=args.omega,
    n_smooth=args.n_smooth,
    decomposition_strategy=args.strategy,
    communicator=args.communicator,
    tolerance=1e-16,  # FMG doesn't iterate to convergence
    max_iter=args.cycles,  # Used for tracking
)

if rank == 0:
    print("INFO: Setting up MLflow tracking...")
    setup_mlflow_tracking()
    print(f"FMG configured: N={args.N}³, ranks={n_ranks}, strategy={args.strategy}, comm={args.communicator}")
    print(f"  Cycles: {args.cycles}, n_smooth={args.n_smooth}, omega={args.omega}")
    print(f"  Multigrid levels: {solver.levels}")

# --- Solver Execution ---
# Synchronize before timing
if n_ranks > 1:
    comm.Barrier()

t0 = MPI.Wtime()
solver.fmg_solve(cycles=args.cycles)
wall_time = MPI.Wtime() - t0

# --- Post-processing ---
# compute_l2_error uses MPI allreduce, so ALL ranks must call it
if rank == 0:
    solver.results.wall_time = wall_time
    # Compute Mlup/s based on finest grid size
    # Note: FMG does work at multiple levels, but we report based on finest grid
    n_interior = (args.N - 2) ** 3
    total_updates = n_interior * solver.results.iterations
    solver.results.mlups = total_updates / (wall_time * 1e6)
solver.compute_l2_error()

# --- Logging and Summary on Rank 0 only ---
if rank == 0:
    # Save solution to HDF5
    project_root = get_project_root()
    data_dir = project_root / "data" / "06-scaling" / "fmg"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_file = data_dir / f"fmg_N{args.N}_p{n_ranks}_{args.strategy}_{args.communicator}.h5"
    solver.save_hdf5(output_file)

    # --- MLflow Logging ---
    experiment_name = args.experiment_name or "Experiment-06-Scaling-FMG"

    # Find or create parent run, then start nested child run
    parent_run_name = f"FMG_N{args.N}"
    run_name = f"FMG_N{args.N}_p{n_ranks}_{args.strategy}"

    with start_mlflow_run_context(experiment_name, parent_run_name, run_name, args=args):
        # Log configuration
        log_parameters({
            "N": args.N,
            "cycles": args.cycles,
            "n_smooth": args.n_smooth,
            "omega": args.omega,
            "strategy": args.strategy,
            "communicator": args.communicator,
            "n_ranks": n_ranks,
            "levels": solver.levels,
        })
        # Log results
        log_metrics_dict(asdict(solver.results))
        log_artifact_file(output_file)

    # --- Final Summary ---
    print("\n--- FMG Run Complete ---")
    print(f"Results saved to: {output_file}")
    print("\nSolution Status:")
    print(f"  Converged: {solver.results.converged}")
    print(f"  Iterations: {solver.results.iterations}")
    print(f"  L2 error: {solver.results.final_error:.6e}")
    print(f"  Wall time: {solver.results.wall_time:.4f} seconds")

    # Timing breakdown
    total_time = (solver.results.total_compute_time or 0) + (solver.results.total_halo_time or 0) + (solver.results.total_mpi_comm_time or 0)
    if total_time > 0:
        print("\nTiming breakdown:")
        print(f"  Compute:      {solver.results.total_compute_time or 0:.4f}s ({100 * (solver.results.total_compute_time or 0) / total_time:.1f}%)")
        print(f"  Halo exchange:{solver.results.total_halo_time or 0:.4f}s ({100 * (solver.results.total_halo_time or 0) / total_time:.1f}%)")
        print(f"  MPI allreduce:{solver.results.total_mpi_comm_time or 0:.4f}s ({100 * (solver.results.total_mpi_comm_time or 0) / total_time:.1f}%)")

# Final barrier for clean exit
if n_ranks > 1:
    comm.Barrier()

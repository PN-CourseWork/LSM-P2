"""
Poisson Scaling Experiment
==========================

MPI Jacobi solver for 3D Poisson equation with MLflow logging.

This script is responsible for orchestrating the experiment, including
setting up MLflow, running the solver, and logging the results.

Usage:
    mpiexec -n 4 uv run python compute_scaling.py --N 64
    mpiexec -n 8 uv run python compute_scaling.py --N 128 --tol 1e-8 --numba
"""

import argparse
import sys
from dataclasses import asdict
from mpi4py import MPI
import mlflow

# Project-specific imports
from Poisson import (
    JacobiPoisson,
    DomainDecomposition,
    NumpyHaloExchange,
    CustomHaloExchange,
    get_project_root,
)
from utils.mlflow.io import (
    setup_mlflow_tracking,
    start_mlflow_run_context,
    log_parameters,
    log_metrics_dict,
    log_timeseries_metrics,
    log_artifact_file,
)

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="MPI Jacobi solver for 3D Poisson equation")
parser.add_argument("--N", type=int, default=16, help="Grid size N³")
parser.add_argument("--tol", type=float, default=1e-6, help="Convergence tolerance")
parser.add_argument("--max-iter", type=int, default=5, help="Max iterations")
parser.add_argument("--omega", type=float, default=0.8, help="Relaxation parameter")
parser.add_argument("--strategy", choices=["sliced", "cubic"], default="sliced", help="Decomposition strategy")
parser.add_argument("--communicator", choices=["numpy", "custom"], default="numpy", help="Halo exchange communicator")
parser.add_argument("--numba", action="store_true", help="Use Numba kernel")
parser.add_argument("--job-name", type=str, default=None, help="LSF Job Name for log retrieval")
parser.add_argument("--log-dir", type=str, default="logs", help="Directory for LSF logs")
parser.add_argument("--experiment-name", type=str, default=None, help="MLflow experiment name")
args = parser.parse_args()

# --- MPI and Setup ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_ranks = comm.Get_size()

# --- Solver Configuration ---
decomp = DomainDecomposition(N=args.N, size=n_ranks, strategy=args.strategy)
halo = CustomHaloExchange() if args.communicator == "custom" else NumpyHaloExchange()
solver = JacobiPoisson(
    N=args.N,
    omega=args.omega,
    tolerance=args.tol,
    max_iter=args.max_iter,
    use_numba=args.numba,
    decomposition=decomp,
    communicator=halo,
)

if rank == 0:
    print("INFO: Setting up MLflow tracking...")
    setup_mlflow_tracking()
    print(f"Solver configured: N={args.N}³, ranks={n_ranks}, strategy={args.strategy}, comm={args.communicator}")
    print(f"  Kernel: {'Numba' if args.numba else 'NumPy'}, omega={args.omega}")

# --- Solver Execution ---
# All ranks participate in the solve process.
if args.numba:
    solver.warmup()

# Synchronize before timing
if n_ranks > 1:
    comm.Barrier()

t0 = MPI.Wtime()
solver.solve()
wall_time = MPI.Wtime() - t0

# --- Post-processing ---
# compute_l2_error uses MPI allreduce, so ALL ranks must call it
if rank == 0:
    solver.results.wall_time = wall_time
    # Compute Mlup/s (Million Lattice Updates per Second)
    # Interior points: (N-2)³ updated per iteration
    n_interior = (args.N - 2) ** 3
    total_updates = n_interior * solver.results.iterations
    solver.results.mlups = total_updates / (wall_time * 1e6)
solver.compute_l2_error()

# --- Logging and Summary on Rank 0 only ---
if rank == 0:
    # Save solution to HDF5
    project_root = get_project_root()
    data_dir = project_root / "data" / "06-scaling"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_file = data_dir / f"poisson_N{args.N}_p{n_ranks}_{args.strategy}_{args.communicator}.h5"
    solver.save_hdf5(output_file)

    # --- MLflow Logging ---
    experiment_name = args.experiment_name or "Experiment-06-Scaling"

    # Find or create parent run, then start nested child run
    parent_run_name = f"N{args.N}"
    run_name = f"N{args.N}_p{n_ranks}_{args.strategy}_{args.communicator}"

    with start_mlflow_run_context(experiment_name, parent_run_name, run_name, args=args):
        # Log all data
        log_parameters(asdict(solver.config))
        log_metrics_dict(asdict(solver.results))
        log_timeseries_metrics(solver.timeseries)
        log_artifact_file(output_file)

    # --- Final Summary ---
    print("\n--- Run Complete ---")
    print(f"Results saved to: {output_file}")
    print("\nSolution Status:")
    print(f"  Converged: {solver.results.converged}")
    print(f"  Iterations: {solver.results.iterations}")
    print(f"  L2 error: {solver.results.final_error:.6e}")
    print(f"  Wall time: {solver.results.wall_time:.2f} seconds")
    print(f"  Performance: {solver.results.mlups:.2f} Mlup/s")

    # Timing breakdown
    total_time = (solver.results.total_compute_time or 0) + (solver.results.total_halo_time or 0) + (solver.results.total_mpi_comm_time or 0)
    if total_time > 0:
        print("\nTiming breakdown:")
        print(f"  Compute:      {solver.results.total_compute_time or 0:.3f}s ({100 * (solver.results.total_compute_time or 0) / total_time:.1f}%)")
        print(f"  Halo exchange:{solver.results.total_halo_time or 0:.3f}s ({100 * (solver.results.total_halo_time or 0) / total_time:.1f}%)")
        print(f"  MPI allreduce:{solver.results.total_mpi_comm_time or 0:.3f}s ({100 * (solver.results.total_mpi_comm_time or 0) / total_time:.1f}%)")

# Final barrier for clean exit
if n_ranks > 1:
    comm.Barrier()

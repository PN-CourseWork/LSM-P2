"""
Poisson Scaling Experiment
==========================

MPI Jacobi solver for 3D Poisson equation with MLflow logging.

Usage:
    mpiexec -n 4 uv run python compute_scaling.py --N 64
    mpiexec -n 8 uv run python compute_scaling.py --N 128 --tol 1e-8 --numba
"""

import argparse
import os

from mpi4py import MPI

from Poisson import (
    JacobiPoisson,
    DomainDecomposition,
    NumpyHaloExchange,
    CustomHaloExchange,
    get_project_root,
)

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="MPI Jacobi solver for 3D Poisson equation"
)
parser.add_argument("--N", type=int, default=64, help="Grid size N³ (default: 64)")
parser.add_argument(
    "--tol", type=float, default=1e-6, help="Convergence tolerance (default: 1e-6)"
)
parser.add_argument(
    "--max-iter", type=int, default=50000, help="Max iterations (default: 50000)"
)
parser.add_argument(
    "--omega", type=float, default=0.8, help="Relaxation parameter (default: 0.8)"
)
parser.add_argument(
    "--strategy",
    choices=["sliced", "cubic"],
    default="sliced",
    help="Decomposition strategy",
)
parser.add_argument(
    "--communicator",
    choices=["numpy", "custom"],
    default="numpy",
    help="Halo exchange communicator",
)
parser.add_argument("--numba", action="store_true", help="Use Numba kernel")
parser.add_argument("--job-name", type=str, default=None, help="LSF Job Name for log retrieval")
parser.add_argument("--log-dir", type=str, default="logs", help="Directory containing LSF logs")
args = parser.parse_args()

N = args.N
comm = MPI.COMM_WORLD
n_ranks = comm.Get_size()
rank = comm.Get_rank()

# Setup directories
project_root = get_project_root()
data_dir = project_root / "data" / "scaling"
data_dir.mkdir(parents=True, exist_ok=True)

# Create solver
decomp = DomainDecomposition(N=N, size=n_ranks, strategy=args.strategy)
halo = CustomHaloExchange() if args.communicator == "custom" else NumpyHaloExchange()
solver = JacobiPoisson(
    N=N,
    omega=args.omega,
    tolerance=args.tol,
    max_iter=args.max_iter,
    use_numba=args.numba,
    decomposition=decomp,
    communicator=halo,
)

if rank == 0:
    print(
        f"Solver configured: N={N}³, ranks={n_ranks}, strategy={args.strategy}, comm={args.communicator}"
    )
    print(f"  Kernel: {'Numba' if args.numba else 'NumPy'}, omega={args.omega}")

# MLflow setup with nested runs (auto-detect HPC via LSF env vars)
is_hpc = "LSB_JOBID" in os.environ
experiment_name = "HPC-Poisson-Scaling" if is_hpc else "Poisson-Scaling"
parent_run = f"N{N}"
run_name = f"N{N}_p{n_ranks}_{args.strategy}"
solver.mlflow_start(experiment_name, run_name, parent_run_name=parent_run)

# Save Run ID for external log uploader
if rank == 0 and args.job_name:
    try:
        run_id = mlflow.active_run().info.run_id
        log_path = project_root / args.log_dir
        log_path.mkdir(parents=True, exist_ok=True)
        run_id_file = log_path / f"{args.job_name}.runid"
        with open(run_id_file, "w") as f:
            f.write(run_id)
    except Exception as e:
        print(f"Warning: Could not save run ID to file: {e}")

# Warmup Numba if needed
if args.numba:
    solver.warmup()

# Solve with timing
t0 = MPI.Wtime()
solver.solve()
wall_time = MPI.Wtime() - t0

# Store timing metrics
if rank == 0:
    solver.results.wall_time = wall_time
    solver.results.total_compute_time = sum(solver.timeseries.compute_times)
    solver.results.total_halo_time = sum(solver.timeseries.halo_exchange_times)
    solver.results.total_mpi_comm_time = sum(solver.timeseries.mpi_comm_times)

# Compute L2 error
solver.compute_l2_error()

# Save solution
output_file = data_dir / f"poisson_N{N}_p{n_ranks}.h5"
solver.save_hdf5(output_file)
solver.mlflow_log_artifact(str(output_file))

if rank == 0:
    print(f"\nResults saved to: {output_file}")
    
    # Flush streams to ensure logs on disk are up to date for the external uploader
    import sys
    sys.stdout.flush()
    sys.stderr.flush()

# End MLflow run
solver.mlflow_end()

# Summary
if rank == 0:
    print("\nSolution Status:")
    print(f"  Converged: {solver.results.converged}")
    print(f"  Iterations: {solver.results.iterations}")
    print(f"  L2 error: {solver.results.final_error:.6e}")
    print(f"  Wall time: {wall_time:.2f} seconds")

    # Timing breakdown
    total = (
        solver.results.total_compute_time
        + solver.results.total_halo_time
        + solver.results.total_mpi_comm_time
    )
    if total > 0:
        print("\nTiming breakdown:")
        print(
            f"  Compute:      {solver.results.total_compute_time:.3f}s ({100 * solver.results.total_compute_time / total:.1f}%)"
        )
        print(
            f"  Halo exchange:{solver.results.total_halo_time:.3f}s ({100 * solver.results.total_halo_time / total:.1f}%)"
        )
        print(
            f"  MPI allreduce:{solver.results.total_mpi_comm_time:.3f}s ({100 * solver.results.total_mpi_comm_time / total:.1f}%)"
        )

"""
Scaling Experiment: Single Solver Run with MLflow Logging
==========================================================

Runs a single solver configuration and logs results to MLflow.
Designed to be called from sweep.py or job scripts with varying parameters.

Usage:
    mpiexec -n <ranks> uv run python run_solver.py --N <size> [options]

Examples:
    mpiexec -n 4 uv run python run_solver.py --N 64 --strategy sliced
    mpiexec -n 8 uv run python run_solver.py --N 128 --strategy cubic --max-iter 10000
"""

import argparse
from mpi4py import MPI
from Poisson import (
    JacobiPoisson,
    DomainDecomposition,
    NumpyHaloExchange,
    DatatypeCommunicator,
)

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Jacobi solver with MLflow logging")
    parser.add_argument("--N", type=int, required=True, help="Grid size")
    parser.add_argument("--strategy", choices=["sliced", "cubic"], default="sliced",
                        help="Decomposition strategy")
    parser.add_argument("--communicator", choices=["numpy", "datatype"], default="numpy",
                        help="Communication method")
    parser.add_argument("--max-iter", type=int, default=10000, help="Maximum iterations")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Convergence tolerance")
    parser.add_argument("--omega", type=float, default=1.0, help="Relaxation parameter")
    parser.add_argument("--experiment", type=str, default="poisson-scaling",
                        help="MLflow experiment name")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    return parser.parse_args()


args = parse_args()

# Print configuration on rank 0
if rank == 0:
    print(f"\n{'='*60}")
    print(f"Scaling Experiment: N={args.N}, ranks={size}")
    print(f"Strategy: {args.strategy}, Communicator: {args.communicator}")
    print(f"{'='*60}\n")

# Create decomposition and communicator
decomposition = DomainDecomposition(N=args.N, size=size, strategy=args.strategy)

if args.communicator == "numpy":
    communicator = NumpyHaloExchange()
else:
    communicator = DatatypeCommunicator()

# Create solver
solver = JacobiPoisson(
    N=args.N,
    omega=args.omega,
    max_iter=args.max_iter,
    tolerance=args.tolerance,
    decomposition=decomposition,
    communicator=communicator,
)

# Start MLflow logging
if not args.no_mlflow:
    solver.mlflow_start(args.experiment)

# Run solver
t_start = MPI.Wtime()
solver.solve()
t_total = MPI.Wtime() - t_start

# Compute error against analytical solution
solver.summary()

# Print results on rank 0
if rank == 0:
    print(f"Iterations: {solver.results.iterations}")
    print(f"Converged: {solver.results.converged}")
    print(f"L2 Error: {solver.results.final_error:.6e}")
    print(f"Total time: {t_total:.3f}s")

    # Log additional timing metric
    if not args.no_mlflow:
        import mlflow
        mlflow.log_metric("wall_time", t_total)

# End MLflow logging
if not args.no_mlflow:
    solver.mlflow_end()

if rank == 0:
    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}\n")



"""
Unified Scaling Experiment Runner
=================================

MPI solver for 3D Poisson equation with MLflow logging.
Supports both Jacobi and FMG solvers, selected by experiment name.

Usage:
    # Direct CLI args:
    mpiexec -n 4 uv run python runner.py --solver jacobi --N 64 --numba

    # Job array mode (reads LSB_JOBINDEX):
    mpiexec -n 8 uv run python runner.py \
        --runtime-config HPC/Configs/runtime/experiments.yaml \
        --experiment jacobi_strong_1node
"""

import argparse
import os
from dataclasses import asdict
from mpi4py import MPI

from Poisson import JacobiPoisson, MultigridPoisson, get_project_root
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
    parser = argparse.ArgumentParser(description="Unified MPI Poisson solver")

    # Runtime config mode
    parser.add_argument("--runtime-config", type=str, help="Path to runtime YAML config")
    parser.add_argument("--experiment", type=str, help="Experiment name in config")
    parser.add_argument("--index", type=int, help="Task index (default: LSB_JOBINDEX)")

    # Solver selection (auto-detected from experiment name if not set)
    parser.add_argument("--solver", choices=["jacobi", "fmg"], help="Solver type")

    # Common params
    parser.add_argument("--N", type=int, default=64, help="Grid size N³")
    parser.add_argument("--omega", type=float, default=0.8, help="Relaxation parameter")
    parser.add_argument("--strategy", choices=["sliced", "cubic"], default="sliced")
    parser.add_argument("--communicator", choices=["numpy", "custom"], default="custom")

    # Jacobi-specific
    parser.add_argument("--tol", type=float, default=0.0, help="Convergence tolerance")
    parser.add_argument("--max-iter", type=int, default=100, help="Max iterations")
    parser.add_argument("--numba", action="store_true", help="Use Numba kernel")

    # FMG-specific
    parser.add_argument("--cycles", type=int, default=2, help="FMG cycles")
    parser.add_argument("--n-smooth", type=int, default=3, help="Smoothing steps")

    # Logging
    parser.add_argument("--job-name", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default="logs/lsf")
    parser.add_argument("--experiment-name", type=str, default=None)

    return parser.parse_args()


def detect_solver(experiment_name: str) -> str:
    """Detect solver type from experiment name."""
    if experiment_name.startswith("jacobi"):
        return "jacobi"
    elif experiment_name.startswith("fmg"):
        return "fmg"
    raise ValueError(f"Cannot detect solver from experiment name: {experiment_name}")


def resolve_params(args):
    """Resolve parameters from config or CLI."""
    if args.runtime_config and args.experiment:
        cfg = load_runtime_config(args.runtime_config, args.experiment, args.index)

        # Handle numba_threads
        if "numba_threads" in cfg:
            os.environ["NUMBA_NUM_THREADS"] = str(cfg.pop("numba_threads"))

        return {
            "N": cfg.get("N", args.N),
            "omega": cfg.get("omega", args.omega),
            "strategy": cfg.get("strategy", args.strategy),
            "communicator": cfg.get("communicator", args.communicator),
            "tol": cfg.get("tol", args.tol),
            "max_iter": cfg.get("max_iter", args.max_iter),
            "cycles": cfg.get("cycles", args.cycles),
            "n_smooth": cfg.get("n_smooth", args.n_smooth),
        }
    else:
        return {
            "N": args.N,
            "omega": args.omega,
            "strategy": args.strategy,
            "communicator": args.communicator,
            "tol": args.tol,
            "max_iter": args.max_iter,
            "cycles": args.cycles,
            "n_smooth": args.n_smooth,
        }


def run_jacobi(params, args, comm, rank, n_ranks):
    """Run Jacobi solver."""
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
        setup_mlflow_tracking()
        print(f"Jacobi: N={params['N']}³, ranks={n_ranks}, {params['strategy']}, {params['communicator']}")
        print(f"  Kernel: {'Numba' if args.numba else 'NumPy'}, omega={params['omega']}")

    if args.numba:
        solver.warmup()

    if n_ranks > 1:
        comm.Barrier()

    t0 = MPI.Wtime()
    solver.solve()
    wall_time = MPI.Wtime() - t0

    if rank == 0:
        solver.results.wall_time = wall_time
        n_interior = (params["N"] - 2) ** 3
        solver.results.mlups = n_interior * solver.results.iterations / (wall_time * 1e6)
    solver.compute_l2_error()

    if rank == 0:
        # Save results
        project_root = get_project_root()
        data_dir = project_root / "data" / "06-scaling" / "jacobi"
        data_dir.mkdir(parents=True, exist_ok=True)
        output_file = data_dir / f"jacobi_N{params['N']}_p{n_ranks}_{params['strategy']}.h5"
        solver.save_hdf5(output_file)

        # MLflow
        exp_name = args.experiment_name or args.experiment or "Experiment-06-Scaling"
        parent_run = f"N{params['N']}"
        run_name = f"jacobi_N{params['N']}_p{n_ranks}_{params['strategy']}"

        with start_mlflow_run_context(exp_name, parent_run, run_name, args=args):
            log_parameters(asdict(solver.config))
            log_metrics_dict(asdict(solver.results))
            log_timeseries_metrics(solver.timeseries)
            log_artifact_file(output_file)
            log_lsf_logs(args.job_name, args.log_dir)

        print(f"\n--- Jacobi Complete ---")
        print(f"  Iterations: {solver.results.iterations}, L2 error: {solver.results.final_error:.6e}")
        print(f"  Wall time: {solver.results.wall_time:.2f}s, {solver.results.mlups:.2f} Mlup/s")


def run_fmg(params, args, comm, rank, n_ranks):
    """Run FMG solver."""
    solver = MultigridPoisson(
        N=params["N"],
        omega=params["omega"],
        n_smooth=params["n_smooth"],
        decomposition_strategy=params["strategy"],
        communicator=params["communicator"],
        tolerance=1e-16,
        max_iter=params["cycles"],
    )

    if rank == 0:
        setup_mlflow_tracking()
        print(f"FMG: N={params['N']}³, ranks={n_ranks}, {params['strategy']}, {params['communicator']}")
        print(f"  Cycles: {params['cycles']}, n_smooth={params['n_smooth']}, levels={solver.levels}")
        numba_threads = os.environ.get("NUMBA_NUM_THREADS", "1")
        print(f"  NUMBA_NUM_THREADS: {numba_threads}")

    if n_ranks > 1:
        comm.Barrier()

    t0 = MPI.Wtime()
    solver.fmg_solve(cycles=params["cycles"])
    wall_time = MPI.Wtime() - t0

    if rank == 0:
        solver.results.wall_time = wall_time
        n_interior = (params["N"] - 2) ** 3
        solver.results.mlups = n_interior * solver.results.iterations / (wall_time * 1e6)
    solver.compute_l2_error()

    if rank == 0:
        # Save results
        project_root = get_project_root()
        data_dir = project_root / "data" / "06-scaling" / "fmg"
        data_dir.mkdir(parents=True, exist_ok=True)
        output_file = data_dir / f"fmg_N{params['N']}_p{n_ranks}_{params['strategy']}.h5"
        solver.save_hdf5(output_file)

        # MLflow
        exp_name = args.experiment_name or args.experiment or "Experiment-06-Scaling-FMG"
        parent_run = f"FMG_N{params['N']}"
        run_name = f"fmg_N{params['N']}_p{n_ranks}_{params['strategy']}"

        with start_mlflow_run_context(exp_name, parent_run, run_name, args=args):
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

        print(f"\n--- FMG Complete ---")
        print(f"  Iterations: {solver.results.iterations}, L2 error: {solver.results.final_error:.6e}")
        print(f"  Wall time: {solver.results.wall_time:.4f}s, {solver.results.mlups:.2f} Mlup/s")


# --- Main ---
args = parse_args()
params = resolve_params(args)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_ranks = comm.Get_size()

# Detect solver type
solver_type = args.solver or (detect_solver(args.experiment) if args.experiment else "jacobi")

if rank == 0 and args.experiment:
    idx = args.index or int(os.environ.get("LSB_JOBINDEX", "1"))
    print(f"=== {args.experiment} task {idx} ===")

if solver_type == "jacobi":
    run_jacobi(params, args, comm, rank, n_ranks)
else:
    run_fmg(params, args, comm, rank, n_ranks)

if n_ranks > 1:
    comm.Barrier()

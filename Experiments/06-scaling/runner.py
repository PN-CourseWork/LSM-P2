"""
Unified Scaling Experiment Runner
=================================

MPI solver for 3D Poisson equation with MLflow logging.
Supports both Jacobi and FMG solvers, selected by experiment name.

Usage:
    # Job array mode (reads LSB_JOBINDEX):
    mpiexec -n 8 uv run python runner.py +experiment=jacobi_strong_1node

    # Local test:
    uv run python runner.py +experiment=jacobi_strong_1node ++sweep.grid.ranks=[1]
"""

import os
import sys
import itertools
from argparse import Namespace
from dataclasses import asdict

import hydra
from omegaconf import DictConfig, OmegaConf
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


def expand_sweep(sweep_cfg):
    """Expand grid and matrix sweeps into a list of parameter dicts."""
    grid = sweep_cfg.get("grid", {})
    matrix = sweep_cfg.get("matrix", [])

    # 1. Expand grid sweep (Cartesian product)
    grid_combos = [{}]
    if grid:
        keys = list(grid.keys())
        values = list(grid.values())
        grid_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # 2. Matrix sweep (explicit combinations)
    # If matrix is empty, use a dummy list with one empty dict
    matrix_combos = matrix if matrix else [{}]

    # 3. Combine: Cartesian product of (grid x matrix)
    final_combos = []
    for g in grid_combos:
        for m in matrix_combos:
            # Merge grid and matrix (matrix overrides grid if collision)
            c = g.copy()
            c.update(m)
            final_combos.append(c)

    return final_combos


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
        exp_name = args.experiment_name or "Experiment-06-Scaling"
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
        exp_name = args.experiment_name or "Experiment-06-Scaling-FMG"
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


@hydra.main(config_path="../hydra-conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)  # Allow adding keys from sweep (e.g. ranks)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    # 1. Resolve Sweep Parameters
    if cfg.get("sweep"):
        sweep_dict = OmegaConf.to_container(cfg.sweep, resolve=True)
        if sweep_dict.get("grid") or sweep_dict.get("matrix"):
            idx = int(os.environ.get("LSB_JOBINDEX", "1"))
            
            combos = expand_sweep(sweep_dict)
            
            if not (1 <= idx <= len(combos)):
                if rank == 0:
                    print(f"Error: LSB_JOBINDEX {idx} out of range (1-{len(combos)})")
                sys.exit(1)
                
            selected = combos[idx - 1]
            
            # Apply selected parameters to cfg
            for k, v in selected.items():
                cfg[k] = v
            
            if rank == 0:
                print(f"Hydra: Selected config {idx}/{len(combos)}: {selected}")

    # 2. Construct Args and Params
    params = {
        "N": cfg.N,
        "omega": cfg.omega,
        "strategy": cfg.strategy,
        "communicator": cfg.communicator,
        "tol": cfg.tol,
        "max_iter": cfg.max_iter,
        "cycles": cfg.cycles,
        "n_smooth": cfg.n_smooth,
    }

    args = Namespace(
        numba=cfg.use_numba,
        experiment_name=cfg.experiment_name,
        job_name=cfg.job_name or os.environ.get("LSB_JOBNAME"),
        log_dir=cfg.log_dir,
    )

    # Handle numba threads if present in selected params (passed via cfg or sweep)
    if "numba_threads" in cfg:
        os.environ["NUMBA_NUM_THREADS"] = str(cfg.numba_threads)

    # 3. Run Solver
    solver_type = cfg.get("solver", "jacobi")
    
    if solver_type == "jacobi":
        run_jacobi(params, args, comm, rank, n_ranks)
    elif solver_type == "fmg":
        run_fmg(params, args, comm, rank, n_ranks)
    else:
        if rank == 0:
            print(f"Unknown solver: {solver_type}")
        sys.exit(1)

if __name__ == "__main__":
    main()

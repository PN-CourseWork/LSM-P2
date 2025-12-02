"""
Kernel Experiments Runner
=========================

Single-run entry point for kernel benchmarks and convergence tests.
Driven by Hydra for parameter sweeps and MLflow for tracking.

Usage:
    # 1. Run a single configuration (using default 'benchmark' config)
    uv run python Experiments/01-kernels/run_kernel.py N=100 kernel=numba threads=4

    # 2. Run a specific configuration defined in a config file
    #    (e.g., convergence.yaml)
    uv run python Experiments/01-kernels/run_kernel.py --config-name=convergence N=25 kernel=numpy

    # 3. Perform a sequential parameter sweep (Multirun) for benchmarks
    #    This executes sequentially to ensure accurate timing measurements.
    uv run python Experiments/01-kernels/run_kernel.py --multirun

    # 4. Perform a PARALLEL parameter sweep (Multirun) for convergence tests
    #    This executes in parallel (via submitit_local) for faster completion.
    uv run python Experiments/01-kernels/run_kernel.py --config-name=convergence --multirun

    # 5. Override parameters during a multirun (e.g., sweep fewer N values)
    uv run python Experiments/01-kernels/run_kernel.py --multirun N=50,100

    # MLflow runs will be created automatically for each execution.
"""

import hydra
from omegaconf import DictConfig
import numpy as np
import mlflow
from scipy.ndimage import laplace
from mpi4py import MPI
import sys
from dataclasses import dataclass, field
from typing import List

from Poisson import get_project_root, NumPyKernel, NumbaKernel, sinusoidal_source_term
from utils.mlflow.io import start_mlflow_run_context, log_parameters, log_metrics_dict, log_timeseries_metrics, setup_mlflow_tracking # Re-enabled setup_mlflow_tracking

def run_kernel(kernel, f, max_iter, track_residuals=False):
    """Run kernel iterations."""
    N = f.shape[0]
    u = np.zeros((N, N, N), dtype=np.float64)
    u_old = np.zeros((N, N, N), dtype=np.float64)
    
    residuals = []
    h2 = kernel.parameters.h**2
    
    t0 = MPI.Wtime()
    for i in range(max_iter): # Changed to iterate explicitly for residual tracking
        kernel.step(u_old, u, f)
        
        if track_residuals and (i % 10 == 0 or i == max_iter -1): # Log every 10th iter or final
            # Compute residual ||Au - f||_inf on interior
            Au = -laplace(u) / h2
            # Slice to interior 1:-1
            diff = Au[1:-1, 1:-1, 1:-1] - f[1:-1, 1:-1, 1:-1]
            res = np.max(np.abs(diff))
            residuals.append(res)
            
        u, u_old = u_old, u
    wall_time = MPI.Wtime() - t0
    
    return u, wall_time, residuals

@hydra.main(config_path="../hydra-conf", config_name="experiment/01-kernels", version_base=None)
def main(cfg: DictConfig):
    # Setup MLflow tracking
    setup_mlflow_tracking(mode=cfg.mlflow.mode)

    is_convergence = (cfg.get("run_mode") == "convergence")
    
    f = sinusoidal_source_term(cfg.N) if is_convergence else np.ones((cfg.N, cfg.N, cfg.N))
    
    if cfg.kernel == "numpy":
        kernel = NumPyKernel(
            N=cfg.N, 
            omega=cfg.omega, 
            tolerance=cfg.tolerance, 
            max_iter=cfg.max_iter
        )
    elif cfg.kernel == "numba":
        kernel = NumbaKernel(
            N=cfg.N, 
            omega=cfg.omega, 
            tolerance=cfg.tolerance, 
            max_iter=cfg.max_iter,
            numba_threads=cfg.threads
        )
    else:
        raise ValueError(f"Unknown kernel type: {cfg.kernel}")

    if cfg.kernel == "numba":
        kernel.warmup()

    # 2. Run
    print(f"Running {cfg.kernel} (N={cfg.N}, threads={cfg.threads if cfg.kernel=='numba' else 'N/A'})...")
    u_final, wall_time, residuals = run_kernel(kernel, f, cfg.max_iter, track_residuals=is_convergence)

    # 3. Calculate Metrics
    n_interior = (cfg.N - 2) ** 3
    total_updates = n_interior * cfg.max_iter
    mlups = total_updates / (wall_time * 1e6)
    
    metrics = {
        "wall_time": wall_time,
        "mlups": mlups,
        "iterations": cfg.max_iter,
    }
    
    if is_convergence and residuals:
         metrics["final_residual"] = residuals[-1]
         print(f"  Final Residual: {residuals[-1]:.6e}")

    print(f"  Wall time: {wall_time:.4f}s")
    print(f"  Performance: {mlups:.2f} Mlup/s")

    # 4. MLflow Logging
    run_name = f"{cfg.kernel}_N{cfg.N}_t{cfg.threads}"
    
    @dataclass
    class KernelTimeSeries:
        residual_history: List[float] = field(default_factory=list)

    ts_data = KernelTimeSeries(residual_history=residuals) if residuals else None

    with start_mlflow_run_context(
        experiment_name=cfg.experiment_name or "kernels",
        parent_run_name=f"Batch_{cfg.get('run_mode', 'manual')}",
        child_run_name=run_name
    ):
        log_parameters({
            "run_mode": cfg.get("run_mode", "manual"),
            "N": cfg.N,
            "kernel": cfg.kernel,
            "threads": cfg.threads,
            "omega": cfg.omega,
            "max_iter": cfg.max_iter
        })
        log_metrics_dict(metrics)
        
        if ts_data:
            log_timeseries_metrics(ts_data)

if __name__ == "__main__":
    main()

"""
Kernel Experiments
==================

Comprehensive kernel validation and benchmarking:
1. Convergence validation with analytical solution
2. Fixed iteration performance benchmark
3. Tolerance-based time-to-convergence benchmark
"""
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import asdict
from scipy.ndimage import laplace

from Poisson import problems
from Poisson.kernels import NumPyKernel, NumbaKernel

# %%
# Configuration
# -------------

# Get the data directory
data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "01-kernels"
data_dir.mkdir(parents=True, exist_ok=True)

# Problem sizes (smaller for convergence to machine epsilon)
problem_sizes = [25]

# Kernel configurations
omega = 1.0
thread_counts = [1, 4, 6, 8]

# Clean up old files
for old_file in data_dir.glob("*.parquet"):
    old_file.unlink()

# %%
# Experiment 1: Convergence Validation
# =====================================
# Compare NumPy vs Numba convergence by tracking iterative residual

print("EXPERIMENT 1: Convergence Validation")

all_dfs = []
tolerance = 0.0  # Near machine epsilon (2.22e-16 for float64)
max_iter = 5000  # More iterations to reach machine precision

for N in problem_sizes:
    # Setup problem with source term
    f = problems.sinusoidal_source_term(N)

    # Create kernels
    numpy_kernel = NumPyKernel(N=N, omega=omega, tolerance=tolerance, max_iter=max_iter)
    numba_kernel = NumbaKernel(N=N, omega=omega, tolerance=tolerance, max_iter=max_iter, numba_threads=4)

    # Warm up Numba on first iteration
    if N == problem_sizes[0]:
        numba_kernel.warmup()

    kernels = [('numpy', numpy_kernel), ('numba', numba_kernel)]

    for kernel_name, kernel in kernels:
        # Initialize arrays
        u = np.zeros((N, N, N), dtype=np.float64)
        u_old = np.zeros((N, N, N), dtype=np.float64)

        # Initialize algebraic residuals list
        if kernel.timeseries.physical_errors is None:
            kernel.timeseries.physical_errors = []

        h = kernel.parameters.h
        h2 = h * h

        for iteration in range(max_iter):
            residual = kernel.step(u_old, u, f)

            # Compute algebraic residual ||Au - f||_inf where Au = -∇²u
            # Only compute on interior points (where the PDE holds)
            Au = -laplace(u) / h2
            interior_residual = Au[1:-1, 1:-1, 1:-1] - f[1:-1, 1:-1, 1:-1]
            algebraic_residual = np.max(np.abs(interior_residual))
            kernel.timeseries.physical_errors.append(algebraic_residual)

            if residual < tolerance:
                kernel.metrics.converged = True
                break

            u, u_old = u_old, u

        # Convert to DataFrame
        df = pd.DataFrame(asdict(kernel.timeseries))
        df['iteration'] = range(len(df))
        df['kernel'] = kernel_name
        df['N'] = N
        df['omega'] = omega
        df['tolerance'] = tolerance
        all_dfs.append(df)

# Save convergence validation results
df_conv = pd.concat(all_dfs, ignore_index=True)
output_path = data_dir / "kernel_convergence.parquet"
df_conv.to_parquet(output_path, index=False)

# %%
# Experiment 2: Fixed Iteration Benchmark
# ========================================
# Measure per-iteration performance with fixed iteration count

print("EXPERIMENT 2: Fixed Iteration Benchmark")

problem_sizes = [25, 50, 75, 100, 125, 150]
all_dfs = []
max_iter = 200
tolerance = 0.0  # Never converge

# NumPy baseline
for N in problem_sizes:
    kernel = NumPyKernel(N=N, omega=omega, tolerance=tolerance, max_iter=max_iter)

    # Initialize arrays
    u = np.zeros((N, N, N), dtype=np.float64)
    u_old = np.zeros((N, N, N), dtype=np.float64)
    f = np.ones((N, N, N), dtype=np.float64)

    for iteration in range(max_iter):
        kernel.step(u_old, u, f)
        u, u_old = u_old, u

    # Convert to DataFrame
    df = pd.DataFrame(asdict(kernel.timeseries))
    df['iteration'] = range(len(df))
    df['kernel'] = 'numpy'
    df['N'] = N
    df['omega'] = omega
    df['max_iter'] = max_iter
    df['use_numba'] = False
    df['num_threads'] = 0
    all_dfs.append(df)

# Numba thread scaling
for num_threads in thread_counts :
    for idx, N in enumerate(problem_sizes):
        kernel = NumbaKernel(N=N, omega=omega, tolerance=tolerance, max_iter=max_iter, numba_threads=num_threads)

        # Warm up on first problem size
        if idx == 0:
            kernel.warmup()

        # Initialize arrays
        u = np.zeros((N, N, N), dtype=np.float64)
        u_old = np.zeros((N, N, N), dtype=np.float64)
        f = np.ones((N, N, N), dtype=np.float64)

        for iteration in range(max_iter):
            kernel.step(u_old, u, f)
            u, u_old = u_old, u

        # Convert to DataFrame
        df = pd.DataFrame(asdict(kernel.timeseries))
        df['iteration'] = range(len(df))
        df['kernel'] = 'numba'
        df['N'] = N
        df['omega'] = omega
        df['max_iter'] = max_iter
        df['use_numba'] = True
        df['num_threads'] = num_threads
        all_dfs.append(df)

# Save benchmark results
df_bench = pd.concat(all_dfs, ignore_index=True)
output_path = data_dir / "kernel_benchmark.parquet"
df_bench.to_parquet(output_path, index=False)


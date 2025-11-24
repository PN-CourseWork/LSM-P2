"""
Kernel Performance Benchmark
=============================

Benchmark NumPy vs Numba kernels with fixed iteration count across different
problem sizes and thread configurations.

"""
import numpy as np
import pandas as pd

from Poisson import JacobiPoisson
from utils import datatools

# %%
# Test Configuration
# ------------------
#
# We benchmark across four problem sizes with multiple thread configurations
# for Numba. The tolerance is set to 0.0 to ensure exactly 100 iterations run.

problem_sizes = [25, 50, 75, 100]  # Grid sizes: N×N×N
omega = 0.75                        # Relaxation parameter
max_iter = 100                      # Fixed iterations for benchmark
tolerance = 0.0                     # Never converge - run all iterations

# Thread counts to test for Numba
thread_counts = [1, 4, 6, 8, 10]

# %%
# Initialize Storage
# ------------------
#
# Clean up old benchmark files and prepare storage for results.

#TODO: avoid using datatools here...
data_dir = datatools.get_data_dir()
print("\nCleaning up old benchmark files...")
for old_file in data_dir.glob("kernel_benchmark*.parquet"):
    old_file.unlink()
    print(f"  Deleted: {old_file.name}")

all_results = []

# %%
# NumPy Baseline
# --------------
#
# First, establish the NumPy baseline performance across all problem sizes.
# This provides the reference for computing Numba speedups.
#TODO: use the kernel class directly instead of the JacobiPoisson...
#TODO: use dataframe directly

for N in problem_sizes:
    print(f"\nTesting N={N}, kernel=numpy")
    print("-" * 60)

    solver = JacobiPoisson(
        N=N,
        omega=omega,
        max_iter=max_iter,
        tolerance=tolerance,
        use_numba=False,
    )

    print("  Solving...")
    solver.solve()

    # Extract results
    res = solver.results
    compute_time = sum(solver.timeseries.compute_times)
    avg_iter_time = compute_time / res.iterations if res.iterations > 0 else 0

    result_dict = {
        'N': N,
        'omega': omega,
        'tolerance': tolerance,
        'max_iter': max_iter,
        'kernel': 'numpy',
        'use_numba': False,
        'num_threads': 0,
        'iterations': res.iterations,
        'converged': res.converged,
        'final_error': res.final_error,
        'compute_time': compute_time,
        'avg_iter_time': avg_iter_time,
    }

    all_results.append(result_dict)

# %%
# Numba Thread Scaling
# ---------------------
#
# Test Numba kernel performance with different thread counts. For each thread
# configuration, we benchmark across all problem sizes.

for num_threads in thread_counts:
    for N in problem_sizes:
        print(f"\nTesting N={N}, kernel=numba, threads={num_threads}")
        print("-" * 60)

        solver = JacobiPoisson(
            N=N,
            omega=omega,
            max_iter=max_iter,
            tolerance=tolerance,
            use_numba=True,
            num_threads=num_threads,
        )

        # Warmup for first problem size only
        if N == problem_sizes[0]:
            print("  Warming up Numba JIT...")
            solver.warmup(N=10)

        print("  Solving...")
        solver.solve()

        # Extract results
        res = solver.results
        compute_time = sum(solver.timeseries.compute_times)
        avg_iter_time = compute_time / res.iterations if res.iterations > 0 else 0

        result_dict = {
            'N': N,
            'omega': omega,
            'tolerance': tolerance,
            'max_iter': max_iter,
            'kernel': 'numba',
            'use_numba': True,
            'num_threads': num_threads,
            'iterations': res.iterations,
            'converged': res.converged,
            'final_error': res.final_error,
            'compute_time': compute_time,
            'avg_iter_time': avg_iter_time,
        }

        all_results.append(result_dict)

        print(f"  Iterations: {res.iterations}")
        print(f"  Total compute time: {compute_time:.4f}s")
        print(f"  Avg iteration time: {avg_iter_time*1000:.3f}ms")

# %%
# Save Results
# ------------
#
# Store benchmark results for analysis by the plotting script.

df = pd.DataFrame(all_results)
output_path = data_dir / "kernel_benchmark.parquet"
datatools.save_simulation_data(df, output_path, format="parquet")


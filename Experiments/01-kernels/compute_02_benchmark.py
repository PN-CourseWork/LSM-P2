"""
Kernel Performance Benchmark
=============================

Benchmark NumPy vs Numba kernels with fixed iteration count across different
problem sizes and thread configurations.

This measures pure computational performance by running exactly 100 iterations
without convergence checking, isolating kernel execution time from iterative
solver behavior.
"""

# %%
# Introduction
# ------------
#
# This benchmark measures the raw computational performance of NumPy and Numba
# kernels by running a **fixed number of iterations** (100) without checking
# for convergence. This approach isolates kernel performance from the iterative
# convergence behavior measured in other experiments.
#
# We test:
#
# * **NumPy baseline** - Pure NumPy implementation (single-threaded)
# * **Numba with varying thread counts** - JIT-compiled parallel execution
#
# The goal is to identify the optimal thread configuration and quantify the
# speedup that Numba JIT compilation provides over NumPy.

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

print("Kernel Performance Benchmark")
print("=" * 60)
print(f"Problem sizes: {problem_sizes}")
print(f"Fixed iterations: {max_iter}")
print(f"Numba thread counts: {thread_counts}")

# %%
# Initialize Storage
# ------------------
#
# Clean up old benchmark files and prepare storage for results.

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

print("\n" + "=" * 60)
print("NumPy Baseline")
print("=" * 60)

for N in problem_sizes:
    print(f"\nTesting N={N} ({N**3:,} grid points), kernel=numpy")
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

    print(f"  Iterations: {res.iterations}")
    print(f"  Total compute time: {compute_time:.4f}s")
    print(f"  Avg iteration time: {avg_iter_time*1000:.3f}ms")

# %%
# Numba Thread Scaling
# ---------------------
#
# Test Numba kernel performance with different thread counts. For each thread
# configuration, we benchmark across all problem sizes.

for num_threads in thread_counts:
    print("\n" + "=" * 60)
    print(f"Numba ({num_threads} threads)")
    print("=" * 60)

    for N in problem_sizes:
        print(f"\nTesting N={N} ({N**3:,} grid points), kernel=numba, threads={num_threads}")
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

print("\n" + "=" * 60)
print("Saving Results")
print("=" * 60)

df = pd.DataFrame(all_results)
output_path = data_dir / "kernel_benchmark.parquet"
datatools.save_simulation_data(df, output_path, format="parquet")

print(f"\nBenchmark results saved to: {output_path}")
print(f"Total records: {len(df)}")
print(f"Configurations tested: NumPy + {len(thread_counts)} Numba configs")
print(f"Problem sizes: {sorted(df['N'].unique())}")
print("\n" + "=" * 60)
print("Kernel benchmarks completed!")
print("=" * 60)

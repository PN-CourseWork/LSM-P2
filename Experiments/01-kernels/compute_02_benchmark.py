"""
Kernel Performance Benchmark
=============================

Benchmark NumPy vs Numba kernels across different problem sizes and thread counts.

This measures pure computational performance by running a fixed number of iterations
(no convergence check) to compare kernel implementations and thread scaling.
"""

import os
import numpy as np
import pandas as pd

from Poisson import JacobiPoisson
from utils import datatools

print("Kernel Performance Benchmarks")
print("=" * 60)

# Test parameters
problem_sizes = [25, 50, 75, 100]
omega = 0.75
max_iter = 100  # Fixed iterations for pure kernel benchmark
tolerance = 0.0  # Never converge - run all iterations

# Thread counts to test
thread_counts = [1, 4, 6, 8, 10]

# Get data directory and clean old files
data_dir = datatools.get_data_dir()
print("\nCleaning up old benchmark files...")
for old_file in data_dir.glob("kernel_benchmark*.parquet"):
    old_file.unlink()
    print(f"  Deleted: {old_file.name}")

# Storage for all results
all_results = []

# ============================================================================
# NumPy Baseline
# ============================================================================

print("\n" + "=" * 60)
print("NumPy Baseline")
print("=" * 60)

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
    solver.summary()

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
    print(f"  Converged: {res.converged}")
    print(f"  Compute time: {compute_time:.4f}s")
    print(f"  Avg iteration time: {avg_iter_time:.6f}s")

# ============================================================================
# Numba with Different Thread Counts
# ============================================================================

for num_threads in thread_counts:
    print("\n" + "=" * 60)
    print(f"Numba ({num_threads} threads)")
    print("=" * 60)

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
        solver.summary()

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
        print(f"  Converged: {res.converged}")
        print(f"  Compute time: {compute_time:.4f}s")
        print(f"  Avg iteration time: {avg_iter_time:.6f}s")

# ============================================================================
# Save Results
# ============================================================================

print("\n" + "=" * 60)
print("Saving Results")
print("=" * 60)

df = pd.DataFrame(all_results)
output_path = data_dir / "kernel_benchmark.parquet"
datatools.save_simulation_data(df, output_path, format="parquet")

print(f"\nBenchmark results saved to: {output_path}")
print(f"Total records: {len(df)}")
print("\n" + "=" * 60)
print("Kernel benchmarks completed!")
print("=" * 60)

"""
Convergence-Based Performance Benchmark
========================================

Benchmark NumPy vs Numba kernels with fixed tolerance to measure time-to-convergence
across different problem sizes and thread configurations.

This measures practical solver performance by allowing natural convergence to a
strict tolerance, combining both kernel speed and convergence behavior.
"""

# %%
# Introduction
# ------------
#
# This benchmark measures **time-to-convergence** rather than raw kernel speed.
# Unlike the fixed-iteration benchmark, here we set a strict tolerance
# (:math:`10^{-10}`) and measure how long each kernel takes to converge.
#
# This approach captures:
#
# * **Kernel computational speed** - How fast each iteration executes
# * **Convergence behavior** - How many iterations are needed
# * **Practical performance** - Total time to reach solution
#
# The combination of these factors determines the practical performance of the
# solver for real problems where we need accurate solutions.

import numpy as np
import pandas as pd

from Poisson import JacobiPoisson
from utils import datatools

# %%
# Test Configuration
# ------------------
#
# We use a strict tolerance to ensure high-quality solutions. The maximum
# iteration limit is set high enough that convergence should always be reached.

problem_sizes = [25, 50, 75, 100]  # Grid sizes: N×N×N
omega = 0.75                        # Relaxation parameter
max_iter = 10000                    # Maximum iterations
tolerance = 1e-10                   # Strict convergence criterion

# Thread counts to test for Numba
thread_counts = [1, 4, 6, 8, 10]

print("Convergence-Based Performance Benchmark")
print("=" * 60)
print(f"Problem sizes: {problem_sizes}")
print(f"Convergence tolerance: {tolerance:.2e}")
print(f"Maximum iterations: {max_iter:,}")
print(f"Numba thread counts: {thread_counts}")

# %%
# Initialize Storage
# ------------------
#
# Clean up old tolerance benchmark files and prepare storage for results.

data_dir = datatools.get_data_dir()
print("\nCleaning up old tolerance benchmark files...")
for old_file in data_dir.glob("tolerance_benchmark*.parquet"):
    old_file.unlink()
    print(f"  Deleted: {old_file.name}")

all_results = []

# %%
# NumPy Baseline
# --------------
#
# Establish NumPy baseline by measuring time-to-convergence for each problem
# size. This provides the reference for computing Numba speedups.

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

    print("  Solving until convergence...")
    solver.solve()

    # Extract results
    res = solver.results
    total_time = sum(solver.timeseries.compute_times)
    avg_iter_time = total_time / res.iterations if res.iterations > 0 else 0

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
        'total_time': total_time,
        'avg_iter_time': avg_iter_time,
    }

    all_results.append(result_dict)

    print(f"  Iterations to convergence: {res.iterations:,}")
    print(f"  Converged: {res.converged}")
    print(f"  Total time: {total_time:.4f}s")
    print(f"  Avg iteration time: {avg_iter_time*1000:.3f}ms")

# %%
# Numba Thread Scaling
# ---------------------
#
# Test Numba kernel performance with different thread counts. For each thread
# configuration, measure time-to-convergence across all problem sizes.

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

        print("  Solving until convergence...")
        solver.solve()

        # Extract results
        res = solver.results
        total_time = sum(solver.timeseries.compute_times)
        avg_iter_time = total_time / res.iterations if res.iterations > 0 else 0

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
            'total_time': total_time,
            'avg_iter_time': avg_iter_time,
        }

        all_results.append(result_dict)

        print(f"  Iterations to convergence: {res.iterations:,}")
        print(f"  Converged: {res.converged}")
        print(f"  Total time: {total_time:.4f}s")
        print(f"  Avg iteration time: {avg_iter_time*1000:.3f}ms")

# %%
# Save Results
# ------------
#
# Store convergence-based benchmark results for analysis by the plotting script.

print("\n" + "=" * 60)
print("Saving Results")
print("=" * 60)

df = pd.DataFrame(all_results)
output_path = data_dir / "tolerance_benchmark.parquet"
datatools.save_simulation_data(df, output_path, format="parquet")

print(f"\nBenchmark results saved to: {output_path}")
print(f"Total records: {len(df)}")
print(f"Configurations tested: NumPy + {len(thread_counts)} Numba configs")
print(f"Problem sizes: {sorted(df['N'].unique())}")
print("\n" + "=" * 60)
print("Tolerance-based benchmarks completed!")
print("=" * 60)

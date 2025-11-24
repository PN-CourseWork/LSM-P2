"""
Convergence-Based Performance Benchmark
========================================

Benchmark NumPy vs Numba kernels with fixed tolerance to measure time-to-convergence
across different problem sizes and thread configurations.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from Poisson import JacobiPoisson

# %%
# Test Configuration
# ------------------
#
# We use a strict tolerance to avoid premature convergence.  

problem_sizes = [25, 50, 75]  # Grid sizes: N×N×N
omega = 0.75                        # Relaxation parameter
max_iter = 10000                    # Maximum iterations
tolerance = 1e-10                   # Strict convergence criterion

# Thread counts to test for Numba
thread_counts = [1, 4, 6, 8]

# %%
# Initialize Storage
# ------------------

# Get the data directory
data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "01-kernels"
data_dir.mkdir(parents=True, exist_ok=True)

print("\nCleaning up old tolerance benchmark files...")
for old_file in data_dir.glob("tolerance_benchmark*.parquet"):
    old_file.unlink()
    print(f"  Deleted: {old_file.name}")

all_results = []

# %%
# NumPy Baseline
# --------------

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

# %%
# Numba Thread Scaling
# ---------------------

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

        # Warm up the Numba kernel to trigger JIT compilation.
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

# %%
# Save Results
# ------------

print("\n" + "=" * 60)
print("Saving Results")
print("=" * 60)

df = pd.DataFrame(all_results)
output_path = data_dir / "tolerance_benchmark.parquet"
df.to_parquet(output_path, index=False)
print(f"Saved to: {output_path}")


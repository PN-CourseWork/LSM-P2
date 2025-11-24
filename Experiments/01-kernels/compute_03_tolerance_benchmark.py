"""
Convergence-Based Performance Benchmark
========================================

Benchmark NumPy vs Numba kernels with fixed tolerance to measure time-to-convergence
across different problem sizes and thread configurations.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from Poisson.kernels import NumPyKernel, NumbaKernel

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

records = []

# %%
# NumPy Baseline
# --------------

print("\n" + "=" * 60)
print("NumPy Baseline")
print("=" * 60)

for N in problem_sizes:
    print(f"\nTesting N={N}, kernel=numpy")
    print("-" * 60)

    kernel = NumPyKernel(N=N, omega=omega, tolerance=tolerance, max_iter=max_iter)

    # Initialize arrays
    u = np.zeros((N, N, N), dtype=np.float64)
    u_old = np.zeros((N, N, N), dtype=np.float64)
    f = np.ones((N, N, N), dtype=np.float64)

    print("  Solving until convergence...")
    for iteration in range(max_iter):
        residual = kernel.step(u_old, u, f)

        if residual < tolerance:
            kernel.metrics.converged = True
            break

        u, u_old = u_old, u

    avg_iter_time = kernel.metrics.total_compute_time / kernel.metrics.iterations if kernel.metrics.iterations > 0 else 0

    records.append({
        'N': N,
        'omega': omega,
        'tolerance': tolerance,
        'max_iter': max_iter,
        'kernel': 'numpy',
        'use_numba': False,
        'num_threads': 0,
        'iterations': kernel.metrics.iterations,
        'converged': kernel.metrics.converged,
        'final_error': float(kernel.metrics.final_residual),
        'total_time': kernel.metrics.total_compute_time,
        'avg_iter_time': avg_iter_time,
    })

    print(f"  Iterations: {kernel.metrics.iterations}, Converged: {kernel.metrics.converged}")
    print(f"  Total time: {kernel.metrics.total_compute_time:.4f}s")
    print(f"  Avg iteration time: {avg_iter_time*1000:.3f}ms")

# %%
# Numba Thread Scaling
# ---------------------

for num_threads in thread_counts:
    print("\n" + "=" * 60)
    print(f"Numba ({num_threads} threads)")
    print("=" * 60)

    for idx, N in enumerate(problem_sizes):
        print(f"\nTesting N={N}, kernel=numba, threads={num_threads}")
        print("-" * 60)

        kernel = NumbaKernel(N=N, omega=omega, tolerance=tolerance, max_iter=max_iter, num_threads=num_threads)

        # Warm up on first problem size for this thread configuration
        if idx == 0:
            print("  Warming up Numba JIT...")
            kernel.warmup()

        # Initialize arrays
        u = np.zeros((N, N, N), dtype=np.float64)
        u_old = np.zeros((N, N, N), dtype=np.float64)
        f = np.ones((N, N, N), dtype=np.float64)

        print("  Solving until convergence...")
        for iteration in range(max_iter):
            residual = kernel.step(u_old, u, f)

            if residual < tolerance:
                kernel.metrics.converged = True
                break

            u, u_old = u_old, u

        avg_iter_time = kernel.metrics.total_compute_time / kernel.metrics.iterations if kernel.metrics.iterations > 0 else 0

        records.append({
            'N': N,
            'omega': omega,
            'tolerance': tolerance,
            'max_iter': max_iter,
            'kernel': 'numba',
            'use_numba': True,
            'num_threads': num_threads,
            'iterations': kernel.metrics.iterations,
            'converged': kernel.metrics.converged,
            'final_error': float(kernel.metrics.final_residual),
            'total_time': kernel.metrics.total_compute_time,
            'avg_iter_time': avg_iter_time,
        })

        print(f"  Iterations: {kernel.metrics.iterations}, Converged: {kernel.metrics.converged}")
        print(f"  Total time: {kernel.metrics.total_compute_time:.4f}s")
        print(f"  Avg iteration time: {avg_iter_time*1000:.3f}ms")

# %%
# Save Results
# ------------

df = pd.DataFrame(records)
output_path = data_dir / "tolerance_benchmark.parquet"
df.to_parquet(output_path, index=False)

print("=" * 60)
print("Tolerance-based benchmarks completed!")
print("=" * 60)
print(f"Saved to: {output_path}")
print(f"Total records: {len(df)}")
print(f"Configurations tested: NumPy + {len(thread_counts)} Numba configs")
print(f"Problem sizes: {sorted(df['N'].unique())}")


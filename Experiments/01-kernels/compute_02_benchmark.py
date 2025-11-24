"""
Kernel Performance Benchmark
=============================

Benchmark NumPy vs Numba kernels with fixed iteration count across different
problem sizes and thread configurations.

"""
import numpy as np
import pandas as pd
from pathlib import Path

from Poisson.kernels import NumPyKernel, NumbaKernel

# %%
# Test Configuration
# ------------------
#
# We benchmark across four problem sizes with multiple thread configurations
# for Numba. The tolerance is set to 0.0 to ensure exactly 100 iterations run.

problem_sizes = [25, 50, 75]  # Grid sizes: N×N×N
omega = 0.75                        # Relaxation parameter
max_iter = 100                      # Fixed iterations for benchmark
tolerance = 0.0                     # Never converge - run all iterations

# Thread counts to test for Numba
thread_counts = [1, 4, 6, 8, 10]

# %%
# Initialize Storage
# ------------------

# Get the data directory
data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "01-kernels"
data_dir.mkdir(parents=True, exist_ok=True)

print("\nCleaning up old benchmark files...")
for old_file in data_dir.glob("kernel_benchmark*.parquet"):
    old_file.unlink()
    print(f"  Deleted: {old_file.name}")

records = []

# %%
# NumPy Baseline
# --------------

for N in problem_sizes:
    print(f"\nTesting N={N}, kernel=numpy")
    print("-" * 60)

    # Setup problem
    kernel = NumPyKernel(N=N, omega=omega, tolerance=tolerance, max_iter=max_iter)

    # Initialize arrays
    u = np.zeros((N, N, N), dtype=np.float64)
    u_old = np.zeros((N, N, N), dtype=np.float64)
    f = np.ones((N, N, N), dtype=np.float64)

    # Benchmark iterations
    print("  Running iterations...")
    for iteration in range(max_iter):
        kernel.step(u_old, u, f)
        u, u_old = u_old, u  # Swap buffers

    # Kernel automatically tracked everything
    records.append({
        'N': N,
        'omega': omega,
        'max_iter': max_iter,
        'kernel': 'numpy',
        'use_numba': False,
        'num_threads': 0,
        'iterations': kernel.metrics.iterations,
        'compute_time': kernel.metrics.total_compute_time,
        'avg_iter_time': kernel.metrics.total_compute_time / kernel.metrics.iterations,
    })

    print(f"  Total time: {kernel.metrics.total_compute_time:.4f}s")
    print(f"  Avg iteration time: {(kernel.metrics.total_compute_time / kernel.metrics.iterations)*1000:.3f}ms")

# %%
# Numba Thread Scaling
# ---------------------

for num_threads in thread_counts:
    print(f"\n{'='*60}")
    print(f"Numba ({num_threads} threads)")
    print('='*60)

    for idx, N in enumerate(problem_sizes):
        print(f"\nTesting N={N}, kernel=numba, threads={num_threads}")
        print("-" * 60)

        # Setup problem
        kernel = NumbaKernel(N=N, omega=omega, tolerance=tolerance, max_iter=max_iter, num_threads=num_threads)

        # Warm up on first problem size for this thread configuration
        if idx == 0:
            print("  Warming up Numba JIT...")
            kernel.warmup()

        # Initialize arrays
        u = np.zeros((N, N, N), dtype=np.float64)
        u_old = np.zeros((N, N, N), dtype=np.float64)
        f = np.ones((N, N, N), dtype=np.float64)

        # Benchmark iterations
        print("  Running iterations...")
        for iteration in range(max_iter):
            kernel.step(u_old, u, f)
            u, u_old = u_old, u  # Swap buffers

        # Kernel automatically tracked everything
        records.append({
            'N': N,
            'omega': omega,
            'max_iter': max_iter,
            'kernel': 'numba',
            'use_numba': True,
            'num_threads': num_threads,
            'iterations': kernel.metrics.iterations,
            'compute_time': kernel.metrics.total_compute_time,
            'avg_iter_time': kernel.metrics.total_compute_time / kernel.metrics.iterations,
        })

        print(f"  Total time: {kernel.metrics.total_compute_time:.4f}s")
        print(f"  Avg iteration time: {(kernel.metrics.total_compute_time / kernel.metrics.iterations)*1000:.3f}ms")

# %%
# Save Results
# ------------

df = pd.DataFrame(records)
output_path = data_dir / "kernel_benchmark.parquet"
df.to_parquet(output_path, index=False)
print(f"Saved to: {output_path}")


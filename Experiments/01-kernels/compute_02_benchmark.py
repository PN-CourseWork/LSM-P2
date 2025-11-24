"""
Kernel Performance Benchmark
=============================

Benchmark NumPy vs Numba kernels with fixed iteration count across different
problem sizes and thread configurations.

"""
import time
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

data = {
    'N': [],
    'omega': [],
    'max_iter': [],
    'kernel': [],
    'use_numba': [],
    'num_threads': [],
    'iterations': [],
    'compute_time': [],
    'avg_iter_time': [],
}

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
    start_time = time.perf_counter()

    for iteration in range(max_iter):
        kernel.step(u_old, u, f)
        u, u_old = u_old, u  # Swap buffers

    compute_time = time.perf_counter() - start_time
    avg_iter_time = compute_time / max_iter

    data['N'].append(N)
    data['omega'].append(omega)
    data['max_iter'].append(max_iter)
    data['kernel'].append('numpy')
    data['use_numba'].append(False)
    data['num_threads'].append(0)
    data['iterations'].append(max_iter)
    data['compute_time'].append(compute_time)
    data['avg_iter_time'].append(avg_iter_time)

    print(f"  Total time: {compute_time:.4f}s")
    print(f"  Avg iteration time: {avg_iter_time*1000:.3f}ms")

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
        start_time = time.perf_counter()

        for iteration in range(max_iter):
            kernel.step(u_old, u, f)
            u, u_old = u_old, u  # Swap buffers

        compute_time = time.perf_counter() - start_time
        avg_iter_time = compute_time / max_iter

        data['N'].append(N)
        data['omega'].append(omega)
        data['max_iter'].append(max_iter)
        data['kernel'].append('numba')
        data['use_numba'].append(True)
        data['num_threads'].append(num_threads)
        data['iterations'].append(max_iter)
        data['compute_time'].append(compute_time)
        data['avg_iter_time'].append(avg_iter_time)

        print(f"  Total time: {compute_time:.4f}s")
        print(f"  Avg iteration time: {avg_iter_time*1000:.3f}ms")

# %%
# Save Results
# ------------

df = pd.DataFrame(data)
output_path = data_dir / "kernel_benchmark.parquet"
df.to_parquet(output_path, index=False)
print(f"Saved to: {output_path}")


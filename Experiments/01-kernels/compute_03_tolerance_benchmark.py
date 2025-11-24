"""
Convergence-Based Performance Benchmark
========================================

Benchmark NumPy vs Numba kernels with fixed tolerance to measure time-to-convergence
across different problem sizes and thread configurations.
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

data = {
    'N': [],
    'omega': [],
    'tolerance': [],
    'max_iter': [],
    'kernel': [],
    'use_numba': [],
    'num_threads': [],
    'iterations': [],
    'converged': [],
    'final_error': [],
    'total_time': [],
    'avg_iter_time': [],
}

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
    start_time = time.perf_counter()

    converged = False
    iterations = 0
    for iteration in range(max_iter):
        residual = kernel.step(u_old, u, f)
        iterations = iteration + 1

        if residual < tolerance:
            converged = True
            break

        u, u_old = u_old, u

    total_time = time.perf_counter() - start_time
    avg_iter_time = total_time / iterations if iterations > 0 else 0

    data['N'].append(N)
    data['omega'].append(omega)
    data['tolerance'].append(tolerance)
    data['max_iter'].append(max_iter)
    data['kernel'].append('numpy')
    data['use_numba'].append(False)
    data['num_threads'].append(0)
    data['iterations'].append(iterations)
    data['converged'].append(converged)
    data['final_error'].append(float(residual))
    data['total_time'].append(total_time)
    data['avg_iter_time'].append(avg_iter_time)

    print(f"  Iterations: {iterations}, Converged: {converged}")
    print(f"  Total time: {total_time:.4f}s")
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
        start_time = time.perf_counter()

        converged = False
        iterations = 0
        for iteration in range(max_iter):
            residual = kernel.step(u_old, u, f)
            iterations = iteration + 1

            if residual < tolerance:
                converged = True
                break

            u, u_old = u_old, u

        total_time = time.perf_counter() - start_time
        avg_iter_time = total_time / iterations if iterations > 0 else 0

        data['N'].append(N)
        data['omega'].append(omega)
        data['tolerance'].append(tolerance)
        data['max_iter'].append(max_iter)
        data['kernel'].append('numba')
        data['use_numba'].append(True)
        data['num_threads'].append(num_threads)
        data['iterations'].append(iterations)
        data['converged'].append(converged)
        data['final_error'].append(float(residual))
        data['total_time'].append(total_time)
        data['avg_iter_time'].append(avg_iter_time)

        print(f"  Iterations: {iterations}, Converged: {converged}")
        print(f"  Total time: {total_time:.4f}s")
        print(f"  Avg iteration time: {avg_iter_time*1000:.3f}ms")

# %%
# Save Results
# ------------

df = pd.DataFrame(data)
output_path = data_dir / "tolerance_benchmark.parquet"
df.to_parquet(output_path, index=False)

print("=" * 60)
print("Tolerance-based benchmarks completed!")
print("=" * 60)
print(f"Saved to: {output_path}")
print(f"Total records: {len(df)}")
print(f"Configurations tested: NumPy + {len(thread_counts)} Numba configs")
print(f"Problem sizes: {sorted(df['N'].unique())}")


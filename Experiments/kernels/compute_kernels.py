"""
Generate Kernel Benchmark Data
===============================

Benchmark Numba JIT-compiled kernels vs pure NumPy implementations.

This script compares performance across different problem sizes,
including warmup iterations to account for JIT compilation overhead.
"""

# %%
# Benchmark Configuration
# -----------------------
#
# Test with: N = [25, 50, 75, 100]
# Compare NumPy vs Numba kernels
# Include warmup iterations for fair JIT comparison

import numpy as np
import pandas as pd

from Poisson import JacobiPoisson
from utils import datatools

print("Kernel Performance Benchmarks")
print("=" * 60)

# Test parameters (small sizes for now)
problem_sizes = [25, 50, 75, 100]
omega = 0.75
max_iter = 1000
tolerance = 1e-5
kernel_types = [("numpy", False), ("numba", True)]

# Storage for results
results = []

for N in problem_sizes:
    for kernel_name, use_numba in kernel_types:
        print(f"\nTesting N={N}, kernel={kernel_name}")
        print("-" * 60)

        # Create solver (no decomposition = sequential mode)
        solver = JacobiPoisson(
            N=N,
            omega=omega,
            max_iter=max_iter,
            tolerance=tolerance,
            use_numba=use_numba,
        )

        # Warmup for numba
        if use_numba:
            print("  Warming up Numba JIT...")
            solver.warmup(N=10)

        # Solve
        print("  Solving...")
        solver.solve()
        solver.summary()

        # Get base results as DataFrame
        df_result = pd.concat([
            solver._dataclass_to_df(solver.config),
            solver._dataclass_to_df(solver.global_results)
        ], axis=1)

        # Add experiment-specific fields
        df_result['kernel'] = kernel_name
        df_result['use_numba'] = use_numba
        df_result['compute_time'] = solver.global_results.compute_time
        df_result['avg_iter_time'] = (
            solver.global_results.compute_time / solver.global_results.iterations
            if solver.global_results.iterations > 0 else 0
        )

        results.append(df_result)

        # Print summary
        res = solver.global_results
        avg_iter_time = (
            res.compute_time / res.iterations if res.iterations > 0 else 0
        )
        print(f"  Iterations: {res.iterations}")
        print(f"  Converged: {res.converged}")
        print(f"  Compute time: {res.compute_time:.4f}s")
        print(f"  Avg iteration time: {avg_iter_time:.6f}s")

# %%
# Save Results
# ------------

# Concatenate all result DataFrames
df = pd.concat(results, ignore_index=True)

# Get data directory
data_dir = datatools.get_data_dir()
output_path = data_dir / "kernel_benchmark.parquet"

# Save
datatools.save_simulation_data(df, output_path, format="parquet")

print("\n" + "=" * 60)
print("Kernel benchmark data generated successfully!")
print(f"Saved to: {output_path}")

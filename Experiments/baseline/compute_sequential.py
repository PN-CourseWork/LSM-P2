"""
Generate Sequential Baseline Data
==================================

This script generates reference data for the sequential Jacobi solver,
which serves as the baseline for all parallel implementations.

Run this script to generate baseline performance data that will be used
by the visualization scripts in this directory.
"""

# %%
# Configuration
# -------------
#
# Test with small problem sizes to verify implementation.
# Problem sizes: N = [50, 100]
# Convergence tolerance: 1e-5
# Maximum iterations: 10000
# Kernels: Both NumPy and Numba variants

import numpy as np
import pandas as pd

from Poisson import SequentialJacobi
from utils import datatools

# %%
# Run Experiments
# ---------------
#
# We'll test both kernel variants (NumPy and Numba) across multiple problem sizes.

# Test parameters
problem_sizes = [50, 100]
omega = 0.75
max_iter = 10000
tolerance = 1e-5
kernel_types = [("numpy", False), ("numba", True)]

# Storage for results
results = []

print("Running Sequential Baseline Experiments")
print("=" * 60)

for N in problem_sizes:
    for kernel_name, use_numba in kernel_types:
        print(f"\nTesting N={N}, kernel={kernel_name}")
        print("-" * 60)

        # Create solver
        solver = SequentialJacobi(
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

        # Extract results
        res = solver.global_results
        config = solver.config

        # Store results
        result_dict = {
            "N": N,
            "kernel": kernel_name,
            "use_numba": use_numba,
            "omega": omega,
            "tolerance": tolerance,
            "max_iter": max_iter,
            "iterations": res.iterations,
            "converged": res.converged,
            "final_error": res.final_error,
            "wall_time": res.wall_time,
            "compute_time": res.compute_time,
            "avg_iter_time": res.compute_time / res.iterations if res.iterations > 0 else 0,
        }
        results.append(result_dict)

        # Print summary
        print(f"  Converged: {res.converged} (iterations: {res.iterations})")
        print(f"  Final error: {res.final_error:.2e}")
        print(f"  Wall time: {res.wall_time:.4f}s")
        print(f"  Compute time: {res.compute_time:.4f}s")
        print(f"  Avg iteration time: {result_dict['avg_iter_time']:.6f}s")

# %%
# Save Results
# ------------
#
# Save to data/baseline/ directory

# Create DataFrame
df = pd.DataFrame(results)

# Get data directory
data_dir = datatools.get_data_dir()
output_path = data_dir / "sequential_baseline.parquet"

# Save
datatools.save_simulation_data(df, output_path, format="parquet")

print("\n" + "=" * 60)
print("Sequential baseline data generated successfully!")
print(f"Saved to: {output_path}")

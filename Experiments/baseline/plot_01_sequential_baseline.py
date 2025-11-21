#!/usr/bin/env python3
"""
Sequential Baseline Performance
================================

Convergence and solution quality for the sequential Jacobi solver.

This experiment establishes baseline metrics:
- Compute time comparison (NumPy vs Numba)
- Iteration performance (time per iteration)
- Problem size scaling

All parallel implementations are compared against this baseline.
"""

# %%
# Setup
# -----

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from utils import datatools

# Load data
# For Sphinx-Gallery, use direct path relative to repo root
repo_root = datatools.get_repo_root()
data_dir = repo_root / "data" / "baseline"
df = datatools.load_simulation_data(data_dir, "sequential_baseline")

# %%
# Kernel Performance Comparison
# ------------------------------
#
# Compare NumPy vs Numba kernels across problem sizes.

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Extract data by kernel
numpy_data = df[df["kernel"] == "numpy"]
numba_data = df[df["kernel"] == "numba"]

problem_sizes = numpy_data["N"].values

# Plot 1: Compute time comparison
ax1.plot(problem_sizes, numpy_data["compute_time"].values,
         marker="o", label="NumPy", linewidth=2)
ax1.plot(problem_sizes, numba_data["compute_time"].values,
         marker="s", label="Numba", linewidth=2)
ax1.set_xlabel("Problem Size (N)")
ax1.set_ylabel("Compute Time (s)")
ax1.set_title("Compute Time: NumPy vs Numba")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Speedup
speedup = numpy_data["compute_time"].values / numba_data["compute_time"].values
ax2.plot(problem_sizes, speedup, marker="o", linewidth=2, color="green")
ax2.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
ax2.set_xlabel("Problem Size (N)")
ax2.set_ylabel("Speedup (NumPy / Numba)")
ax2.set_title("Numba Speedup vs NumPy")
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# %%
# Iteration Performance
# ---------------------
#
# Average time per iteration for each kernel.

fig, ax = plt.subplots(figsize=(8, 6))

x = np.arange(len(problem_sizes))
width = 0.35

ax.bar(x - width/2, numpy_data["avg_iter_time"].values * 1000,
       width, label="NumPy", alpha=0.8)
ax.bar(x + width/2, numba_data["avg_iter_time"].values * 1000,
       width, label="Numba", alpha=0.8)

ax.set_xlabel("Problem Size (N)")
ax.set_ylabel("Average Iteration Time (ms)")
ax.set_title("Iteration Performance by Kernel")
ax.set_xticks(x)
ax.set_xticklabels(problem_sizes)
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()

# %%
# Performance Summary Table
# -------------------------
#
# Detailed breakdown of solver performance.

print("\nSequential Baseline Performance Summary")
print("=" * 80)
print(f"\n{'N':<8} {'Kernel':<10} {'Iterations':<12} {'Final Error':<15} "
      f"{'Compute Time':<15} {'Speedup':<10}")
print("-" * 80)

for i, size in enumerate(problem_sizes):
    numpy_row = numpy_data.iloc[i]
    numba_row = numba_data.iloc[i]
    speedup = numpy_row["compute_time"] / numba_row["compute_time"]

    print(f"{size:<8} {'NumPy':<10} {numpy_row['iterations']:<12} "
          f"{numpy_row['final_error']:<15.2e} {numpy_row['compute_time']:<15.6f} {'-':<10}")
    print(f"{size:<8} {'Numba':<10} {numba_row['iterations']:<12} "
          f"{numba_row['final_error']:<15.2e} {numba_row['compute_time']:<15.6f} "
          f"{speedup:<10.2f}x")
    print()

# %%
# Key Findings
# ------------
#
# **Numba Acceleration:** JIT compilation provides significant speedup over NumPy
#
# **Convergence:** Both kernels achieve identical convergence (same iterations and error)
#
# **Scaling:** Compute time increases with problem size as expected (O(N³) per iteration)

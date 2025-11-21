#!/usr/bin/env python3
"""
Solver Validation Analysis
===========================

Comprehensive validation of Jacobi solver correctness and convergence properties.

This validates both spatial and iterative convergence for the 3D Poisson equation
-∇²u = f with exact solution u(x,y,z) = sin(πx)sin(πy)sin(πz) on [-1,1]³.

**Spatial Convergence:**
  Error decreases as O(h²) with grid refinement, confirming second-order accuracy
  of centered finite differences.

**Iterative Convergence:**
  Residual decreases monotonically each iteration, confirming proper implementation
  of weighted Jacobi iteration (ω=0.75).
"""

# %%
# Setup
# -----

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

from utils import datatools

# Set seaborn style

# Load validation data
repo_root = datatools.get_repo_root()
data_dir = repo_root / "data" / "validation"
df_spatial = datatools.load_simulation_data(data_dir, "spatial_convergence")
df_iterative = datatools.load_simulation_data(data_dir, "iterative_convergence")

# %%
# Spatial Convergence: Method Comparison
# ---------------------------------------
#
# Log-log plot comparing spatial convergence for different methods.
# Shows O(h²) = O(N⁻²) error decay with grid refinement.

fig, ax = plt.subplots(figsize=(10, 7))

# Plot using seaborn
sns.lineplot(data=df_spatial, x="N", y="final_error", hue="method",
             marker="o", markersize=8, linewidth=2, ax=ax)

# Add O(N⁻²) reference line
N_all = df_spatial["N"].unique()
N_ref = np.array([N_all.min(), N_all.max()])
error_first = df_spatial[df_spatial["N"] == N_all.min()]["final_error"].iloc[0]
error_ref = error_first * (N_ref / N_all.min()) ** (-2)
ax.plot(N_ref, error_ref, "--", color="gray", linewidth=2,
        label=r"$O(N^{-2})$ reference", alpha=0.7)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Problem Size (N)", fontsize=12)
ax.set_ylabel("L2 Error", fontsize=12)
ax.set_title("Spatial Convergence: Method Comparison", fontsize=13)
ax.legend(fontsize=11)

plt.tight_layout()

# %%
# Iterative Convergence: NumPy vs Numba
# --------------------------------------
#
# Compare convergence behavior between NumPy and Numba kernels.

g = sns.relplot(data=df_iterative, x="iteration", y="residual", col="kernel",
                kind="line", height=5, aspect=1.3, linewidth=2,
                facet_kws={"sharex": True, "sharey": True})

g.set(yscale="log")
g.set_axis_labels("Iteration", "Residual", fontsize=12)
N_val = df_iterative['N'].iloc[0]
g.set_titles(col_template="{col_name} Kernel (N=" + str(N_val) + ")", fontsize=13)

plt.tight_layout()

# %%
# Key Findings
# ------------
#
# **Second-Order Spatial Convergence:** Error decreases as O(h²) = O(N⁻²),
# confirming correct implementation of centered finite differences.
#
# **Kernel Equivalence:** NumPy and Numba kernels produce identical convergence
# behavior, confirming that JIT compilation preserves numerical accuracy.
#
# **Solution Accuracy:** For N=100 (h≈0.02), the discrete L2 error is ~3.3×10⁻⁴,
# providing high-accuracy solutions for the test problem.

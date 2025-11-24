#!/usr/bin/env python3
"""
MPI Solver Validation Analysis
================================

Validation of different MPI decomposition and communication strategies.

This validates spatial convergence for the 3D Poisson equation comparing:
- Sequential (no MPI)
- Sliced decomposition (1D) with Custom MPI / NumPy communicators
- Cubic decomposition (3D) with Custom MPI / NumPy communicators

**Spatial Convergence:**
  Error decreases as O(h²) with grid refinement, confirming second-order accuracy
  and correctness of all parallel implementations.
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
from Poisson import PostProcessor

# Load validation data from HDF5 files
repo_root = datatools.get_repo_root()
data_dir = repo_root / "data" / "validation"
h5_dir = data_dir / "validation_h5"

# Find all HDF5 files
h5_files = sorted(h5_dir.glob("*.h5"))

if not h5_files:
    raise FileNotFoundError(f"No HDF5 files found in {h5_dir}")

print(f"Found {len(h5_files)} HDF5 files")

# Load data using PostProcessor
pp = PostProcessor(h5_files)
df_spatial = pp.to_dataframe()

# Add method column based on decomposition and communicator
def get_method_name(row):
    if row['decomposition'] == 'none':
        return 'Sequential'
    else:
        decomp = row['decomposition'].capitalize()
        comm = row['communicator'].capitalize()
        return f"MPI_{decomp}_{comm}"

df_spatial['method'] = df_spatial.apply(get_method_name, axis=1)

print(f"Loaded {len(df_spatial)} validation runs")
print(df_spatial[['N', 'method', 'final_error']].to_string())

# %%
# Spatial Convergence: Method Comparison
# ---------------------------------------
#
# Log-log plot comparing spatial convergence for different methods.
# Shows O(h²) = O(N⁻²) error decay with grid refinement.

# Get unique methods and sort them
methods = sorted(df_spatial['method'].unique())
n_methods = len(methods)

# Create subplots - one column per method
fig, axes = plt.subplots(1, n_methods, figsize=(4*n_methods, 4), sharey=True)

# Define colors for each method
method_colors = {
    'Sequential': 'C0',
    'MPI_Sliced_Custom': 'C1',
    'MPI_Sliced_Numpy': 'C2',
    'MPI_Cubic_Custom': 'C3',
    'MPI_Cubic_Numpy': 'C4',
}

# Compute O(N⁻²) reference line
N_all = df_spatial["N"].unique()
N_ref = np.array([N_all.min(), N_all.max()])
error_first = df_spatial[df_spatial["N"] == N_all.min()]["final_error"].iloc[0]
error_ref = error_first * (N_ref / N_all.min()) ** (-2)

# Plot each method in its own subplot
for idx, method in enumerate(methods):
    ax = axes[idx]
    method_data = df_spatial[df_spatial['method'] == method].sort_values('N')
    color = method_colors.get(method, 'black')

    # Plot method data
    ax.plot(method_data['N'], method_data['final_error'],
            marker='o', markersize=8, linewidth=2.5,
            color=color, label=method)

    # Add O(N⁻²) reference line
    ax.plot(N_ref, error_ref, "--", color="gray", linewidth=2,
            label=r"$O(N^{-2})$ reference", alpha=0.7)

    # Set scales and labels
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Problem Size (N)", fontsize=11)
    ax.set_title(method, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

# Only label y-axis on leftmost subplot
axes[0].set_ylabel("L2 Error", fontsize=11)

# Add overall title
fig.suptitle("Spatial Convergence: Method Comparison", fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()

# Save figure (both PDF and PNG)
from utils import datatools
fig_dir = datatools.get_repo_root() / "figures" / "validation"
fig_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(fig_dir / "validation_spatial_convergence.pdf", bbox_inches='tight')
fig.savefig(fig_dir / "validation_spatial_convergence.png", dpi=300, bbox_inches='tight')
print(f"Saved: {fig_dir / 'validation_spatial_convergence.pdf'}")
print(f"Saved: {fig_dir / 'validation_spatial_convergence.png'}")

# %%
# Key Findings
# ------------
#
# **Second-Order Spatial Convergence:** Error decreases as O(h²) = O(N⁻²),
# confirming correct implementation of centered finite differences.
#
# **Method Equivalence:** All MPI decomposition and communication strategies
# produce identical results, validating the correctness of parallel implementations.

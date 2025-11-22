#!/usr/bin/env python3
"""
Numba vs NumPy Kernel Comparison
=================================

Direct performance comparison between JIT-compiled Numba kernels
and pure NumPy implementations of the Jacobi iteration.

This experiment measures:
- Raw iteration time (excluding I/O and setup)
- Speedup factor (Numba vs NumPy)
- Scaling with problem size N
"""

# %%
# Setup
# -----

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import datatools

sns.set_theme(style="whitegrid")

# %%
# Load Data
# ---------

data_dir = repo_root / "data" / "kernels"
df = datatools.load_simulation_data(data_dir, "kernel_benchmark")

# %%
# Kernel Performance Comparison
# ------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot absolute timings
for kernel in df['kernel'].unique():
    kernel_data = df[df['kernel'] == kernel]
    ax1.plot(kernel_data['N'], kernel_data['avg_iter_time'],
             marker='o', label=kernel, linewidth=2)

ax1.set_xlabel("Problem Size N")
ax1.set_ylabel("Time per Iteration (s)")
ax1.set_title("Kernel Performance")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Calculate and plot speedup factor
pivot_df = df.pivot(index='N', columns='kernel', values='avg_iter_time')
speedup = pivot_df['numpy'] / pivot_df['numba']

ax2.plot(speedup.index, speedup.values, marker='o', linewidth=2, color='C2')
ax2.axhline(y=1, color='k', linestyle='--', alpha=0.3, label='No speedup')
ax2.set_xlabel("Problem Size N")
ax2.set_ylabel("Speedup (NumPy / Numba)")
ax2.set_title("Numba Speedup Factor")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
fig_dir = repo_root / "figures" / "kernels"
fig_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(fig_dir / "02_numba_vs_numpy.png", dpi=300, bbox_inches='tight')
print(f"Saved: {fig_dir / '02_numba_vs_numpy.png'}")

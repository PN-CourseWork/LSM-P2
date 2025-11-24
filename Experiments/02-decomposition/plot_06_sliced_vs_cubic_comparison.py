#!/usr/bin/env python3
"""
Sliced vs Cubic Decomposition Comparison
=========================================

Direct comparison of 1D sliced and 3D cubic decomposition strategies.

This experiment compares:
- Communication overhead (2 planes vs 6 faces)
- Scalability characteristics
- Optimal decomposition for different problem sizes
- Crossover points where one outperforms the other
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

data_dir = repo_root / "data" / "decomposition"
# Load np=2 data
df = datatools.load_simulation_data(data_dir, "decomposition_comparison_np2")

# %%
# Communication Overhead Comparison
# ----------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot communication time
for decomp in df['decomposition'].unique():
    decomp_data = df[df['decomposition'] == decomp]
    ax1.plot(decomp_data['N'], decomp_data['halo_exchange_time'],
             marker='o', label=decomp, linewidth=2)

ax1.set_xlabel("Problem Size N")
ax1.set_ylabel("Halo Exchange Time (s)")
ax1.set_title("Communication Overhead: Sliced vs Cubic")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot total time
for decomp in df['decomposition'].unique():
    decomp_data = df[df['decomposition'] == decomp]
    ax2.plot(decomp_data['N'], decomp_data['wall_time'],
             marker='o', label=decomp, linewidth=2)

ax2.set_xlabel("Problem Size N")
ax2.set_ylabel("Total Wall Time (s)")
ax2.set_title("Total Performance: Sliced vs Cubic")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
fig_dir = repo_root / "figures" / "decomposition"
fig_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(fig_dir / "06_sliced_vs_cubic_comparison.png", dpi=300, bbox_inches='tight')
print(f"Saved: {fig_dir / '06_sliced_vs_cubic_comparison.png'}")

# %%
# Communication Overhead Percentage
# ----------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

for decomp in df['decomposition'].unique():
    decomp_data = df[df['decomposition'] == decomp]
    comm_pct = 100 * decomp_data['halo_exchange_time'] / decomp_data['wall_time']
    ax.plot(decomp_data['N'], comm_pct,
            marker='o', label=decomp, linewidth=2)

ax.set_xlabel("Problem Size N")
ax.set_ylabel("Communication Overhead (%)")
ax.set_title("Communication Overhead Percentage: Sliced vs Cubic")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
fig.savefig(fig_dir / "06_communication_overhead_pct.png", dpi=300, bbox_inches='tight')
print(f"Saved: {fig_dir / '06_communication_overhead_pct.png'}")

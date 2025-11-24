#!/usr/bin/env python3
"""
Strong Scaling Analysis
=======================

Parallel speedup with fixed problem size and increasing rank count.

This experiment measures:
- Speedup: T(1) / T(N) where N is rank count
- Parallel efficiency: Speedup / N
- Scalability limits (where does speedup plateau?)
- Communication overhead as function of rank count
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

data_dir = repo_root / "data" / "scaling"

# Load data from different MPI sizes and combine
dfs = []
for np_file in data_dir.glob("scaling_analysis_np*.parquet"):
    df = pd.read_parquet(np_file)
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

# Filter for strong scaling only
df_strong = df[df['scaling_type'] == 'strong'].copy()

# %%
# Strong Scaling: Speedup
# ------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Get baseline (sequential or smallest rank) time for each decomposition
for decomp in df_strong['decomposition'].unique():
    decomp_data = df_strong[df_strong['decomposition'] == decomp].sort_values('mpi_size')

    # Use smallest rank as baseline
    baseline_time = decomp_data.iloc[0]['wall_time']
    ranks = decomp_data['mpi_size'].values
    times = decomp_data['wall_time'].values
    speedup = baseline_time / times

    ax1.plot(ranks, speedup, marker='o', label=decomp, linewidth=2)

# Plot ideal speedup
ranks_ideal = np.array([1, 2, 4])
ax1.plot(ranks_ideal, ranks_ideal, 'k--', alpha=0.3, label='Ideal', linewidth=1.5)

ax1.set_xlabel("Number of Ranks")
ax1.set_ylabel("Speedup (T_1 / T_n)")
ax1.set_title("Strong Scaling: Speedup")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Parallel Efficiency
for decomp in df_strong['decomposition'].unique():
    decomp_data = df_strong[df_strong['decomposition'] == decomp].sort_values('mpi_size')

    baseline_time = decomp_data.iloc[0]['wall_time']
    ranks = decomp_data['mpi_size'].values
    times = decomp_data['wall_time'].values
    speedup = baseline_time / times
    efficiency = speedup / ranks

    ax2.plot(ranks, efficiency, marker='o', label=decomp, linewidth=2)

ax2.axhline(y=1, color='k', linestyle='--', alpha=0.3, label='Ideal', linewidth=1.5)
ax2.set_xlabel("Number of Ranks")
ax2.set_ylabel("Parallel Efficiency (Speedup / N)")
ax2.set_title("Strong Scaling: Parallel Efficiency")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1.2)

plt.tight_layout()

# Save figure
fig_dir = repo_root / "figures" / "scaling"
fig_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(fig_dir / "10_strong_scaling.png", dpi=300, bbox_inches='tight')
print(f"Saved: {fig_dir / '10_strong_scaling.png'}")

# %%
# Time Breakdown
# --------------

fig, ax = plt.subplots(figsize=(10, 6))

for decomp in df_strong['decomposition'].unique():
    decomp_data = df_strong[df_strong['decomposition'] == decomp].sort_values('mpi_size')

    ranks = decomp_data['mpi_size'].values
    compute_times = decomp_data['compute_time'].values
    comm_times = decomp_data['halo_exchange_time'].values

    ax.plot(ranks, compute_times, marker='o', linestyle='-',
            label=f'{decomp} - Compute', linewidth=2)
    ax.plot(ranks, comm_times, marker='s', linestyle='--',
            label=f'{decomp} - Communication', linewidth=2)

ax.set_xlabel("Number of Ranks")
ax.set_ylabel("Time (s)")
ax.set_title("Strong Scaling: Compute vs Communication Time")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
fig.savefig(fig_dir / "10_time_breakdown.png", dpi=300, bbox_inches='tight')
print(f"Saved: {fig_dir / '10_time_breakdown.png'}")

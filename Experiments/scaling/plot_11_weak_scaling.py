#!/usr/bin/env python3
"""
Weak Scaling Analysis
=====================

Parallel efficiency with constant work per rank.

This experiment measures:
- Time consistency (ideally constant as we scale)
- Weak scaling efficiency: T(1) / T(N)
- How communication overhead grows with system size
- Maximum scalable problem size
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

# Filter for weak scaling only
df_weak = df[df['scaling_type'] == 'weak'].copy()

# %%
# Weak Scaling: Time Consistency
# -------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot absolute times
for decomp in df_weak['decomposition'].unique():
    decomp_data = df_weak[df_weak['decomposition'] == decomp].sort_values('mpi_size')

    ranks = decomp_data['mpi_size'].values
    times = decomp_data['wall_time'].values

    ax1.plot(ranks, times, marker='o', label=decomp, linewidth=2)

# Ideal line (constant time)
if len(df_weak) > 0:
    baseline = df_weak.groupby('decomposition')['wall_time'].first().mean()
    ranks_range = df_weak['mpi_size'].unique()
    ax1.axhline(y=baseline, color='k', linestyle='--',
                alpha=0.3, label='Ideal (T_1)', linewidth=1.5)

ax1.set_xlabel("Number of Ranks")
ax1.set_ylabel("Wall Time (s)")
ax1.set_title("Weak Scaling: Time Consistency")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Weak scaling efficiency
for decomp in df_weak['decomposition'].unique():
    decomp_data = df_weak[df_weak['decomposition'] == decomp].sort_values('mpi_size')

    baseline_time = decomp_data.iloc[0]['wall_time']
    ranks = decomp_data['mpi_size'].values
    times = decomp_data['wall_time'].values
    efficiency = baseline_time / times

    ax2.plot(ranks, efficiency, marker='o', label=decomp, linewidth=2)

ax2.axhline(y=1, color='k', linestyle='--', alpha=0.3, label='Ideal', linewidth=1.5)
ax2.set_xlabel("Number of Ranks")
ax2.set_ylabel("Weak Scaling Efficiency (T_1 / T_n)")
ax2.set_title("Weak Scaling: Parallel Efficiency")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1.2)

plt.tight_layout()

# Save figure
fig_dir = repo_root / "figures" / "scaling"
fig_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(fig_dir / "11_weak_scaling.png", dpi=300, bbox_inches='tight')
print(f"Saved: {fig_dir / '11_weak_scaling.png'}")

# %%
# Communication Growth
# --------------------

fig, ax = plt.subplots(figsize=(10, 6))

for decomp in df_weak['decomposition'].unique():
    decomp_data = df_weak[df_weak['decomposition'] == decomp].sort_values('mpi_size')

    ranks = decomp_data['mpi_size'].values
    comm_fraction = decomp_data['halo_exchange_time'] / decomp_data['wall_time']

    ax.plot(ranks, comm_fraction, marker='o', label=decomp, linewidth=2)

ax.set_xlabel("Number of Ranks")
ax.set_ylabel("Communication Time Fraction")
ax.set_title("Weak Scaling: Communication Overhead Growth")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)

plt.tight_layout()

# Save figure
fig.savefig(fig_dir / "11_communication_growth.png", dpi=300, bbox_inches='tight')
print(f"Saved: {fig_dir / '11_communication_growth.png'}")

#!/usr/bin/env python3
"""
Communication Method Comparison
================================

Direct comparison of custom MPI datatypes vs NumPy array communication.

This experiment compares:
- Total communication overhead
- Code complexity vs performance trade-off
- When to use each method
- Crossover points and recommendations
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

data_dir = repo_root / "data" / "communication"
df = datatools.load_simulation_data(data_dir, "communication_comparison_np2")

# %%
# Total Overhead Comparison
# --------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot communication time by method
for method in df['method'].unique():
    method_data = df[df['method'] == method]
    ax1.plot(method_data['N'], method_data['halo_exchange_time'],
             marker='o', label=method, linewidth=2)

ax1.set_xlabel("Problem Size N")
ax1.set_ylabel("Halo Exchange Time (s)")
ax1.set_title("Communication Time by Method")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot communication overhead percentage
for method in df['method'].unique():
    method_data = df[df['method'] == method]
    ax2.plot(method_data['N'], method_data['comm_overhead_pct'],
             marker='o', label=method, linewidth=2)

ax2.set_xlabel("Problem Size N")
ax2.set_ylabel("Communication Overhead (%)")
ax2.set_title("Communication Overhead Percentage")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
fig_dir = repo_root / "figures" / "communication"
fig_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(fig_dir / "09_communication_overhead.png", dpi=300, bbox_inches='tight')
print(f"Saved: {fig_dir / '09_communication_overhead.png'}")

# %%
# Performance by Decomposition and Communicator
# ----------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Sliced decomposition comparison
sliced_data = df[df['decomposition'] == 'sliced']
for comm in sliced_data['communicator'].unique():
    comm_data = sliced_data[sliced_data['communicator'] == comm]
    ax1.plot(comm_data['N'], comm_data['halo_exchange_time'],
             marker='o', label=f'{comm}', linewidth=2)

ax1.set_xlabel("Problem Size N")
ax1.set_ylabel("Halo Exchange Time (s)")
ax1.set_title("Sliced Decomposition: Custom vs NumPy")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Cubic decomposition comparison
cubic_data = df[df['decomposition'] == 'cubic']
for comm in cubic_data['communicator'].unique():
    comm_data = cubic_data[cubic_data['communicator'] == comm]
    ax2.plot(comm_data['N'], comm_data['halo_exchange_time'],
             marker='o', label=f'{comm}', linewidth=2)

ax2.set_xlabel("Problem Size N")
ax2.set_ylabel("Halo Exchange Time (s)")
ax2.set_title("Cubic Decomposition: Custom vs NumPy")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
fig.savefig(fig_dir / "09_decomposition_comm_comparison.png", dpi=300, bbox_inches='tight')
print(f"Saved: {fig_dir / '09_decomposition_comm_comparison.png'}")

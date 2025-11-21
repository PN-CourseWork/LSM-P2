#!/usr/bin/env python3
"""
Cubic Decomposition Analysis
=============================

Performance analysis of 3D cubic domain decomposition.

This experiment analyzes:
- Compute vs communication time breakdown
- Scaling with problem size N (fixed ranks)
- Scaling with rank count (fixed problem size)
- Ghost face exchange overhead (6 faces vs 2 planes)
"""

# %%
# Setup
# -----

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# TODO: Load data from decomposition/compute_cubic.py output

# %%
# Time Breakdown
# --------------

fig, ax = plt.subplots(figsize=(10, 6))

# TODO: Stacked bar chart showing compute vs communication time
ax.set_xlabel("Problem Size N")
ax.set_ylabel("Time (s)")
ax.set_title("Cubic Decomposition: Compute vs Communication")
ax.legend()

plt.tight_layout()
plt.show()

# %%
# Scaling Analysis
# ----------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# TODO: Problem size scaling (fixed ranks)
ax1.set_xlabel("Problem Size N")
ax1.set_ylabel("Total Time (s)")
ax1.set_title("Problem Size Scaling (Fixed Ranks)")

# TODO: Rank count scaling (fixed problem size)
ax2.set_xlabel("Number of Ranks")
ax2.set_ylabel("Total Time (s)")
ax2.set_title("Strong Scaling (Fixed Problem Size)")

plt.tight_layout()
plt.show()

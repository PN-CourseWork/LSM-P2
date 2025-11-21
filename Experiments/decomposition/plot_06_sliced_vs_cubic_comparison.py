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

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# TODO: Load data from both compute_sliced.py and compute_cubic.py

# %%
# Communication Overhead Comparison
# ----------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

# TODO: Plot communication time for both decompositions
ax.set_xlabel("Problem Size N")
ax.set_ylabel("Communication Time (s)")
ax.set_title("Communication Overhead: Sliced vs Cubic")
ax.legend()

plt.tight_layout()
plt.show()

# %%
# Overall Performance Comparison
# -------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

# TODO: Plot total time for both decompositions
ax.set_xlabel("Problem Size N")
ax.set_ylabel("Total Time (s)")
ax.set_title("Total Performance: Sliced vs Cubic")
ax.legend()

plt.tight_layout()
plt.show()

# %%
# Efficiency Analysis
# -------------------

fig, ax = plt.subplots(figsize=(10, 6))

# TODO: Plot parallel efficiency (speedup/ranks)
ax.set_xlabel("Number of Ranks")
ax.set_ylabel("Parallel Efficiency")
ax.axhline(y=1, color='k', linestyle='--', alpha=0.3)
ax.set_title("Parallel Efficiency: Sliced vs Cubic")
ax.legend()

plt.tight_layout()
plt.show()

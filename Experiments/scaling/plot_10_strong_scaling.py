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

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# TODO: Load data from scaling/compute_scaling.py

# %%
# Strong Scaling: Speedup
# ------------------------

fig, ax = plt.subplots(figsize=(10, 6))

# TODO: Plot speedup vs rank count
# Plot ideal speedup line (y = x)
ranks = np.array([1, 2, 4, 8, 16, 32, 64])
ax.plot(ranks, ranks, 'k--', alpha=0.3, label='Ideal')

ax.set_xlabel("Number of Ranks")
ax.set_ylabel("Speedup (T₁ / Tₙ)")
ax.set_xscale("log", base=2)
ax.set_yscale("log", base=2)
ax.set_title("Strong Scaling: Speedup")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Strong Scaling: Efficiency
# ---------------------------

fig, ax = plt.subplots(figsize=(10, 6))

# TODO: Plot parallel efficiency (speedup/ranks)
ax.axhline(y=1, color='k', linestyle='--', alpha=0.3, label='Ideal')

ax.set_xlabel("Number of Ranks")
ax.set_ylabel("Parallel Efficiency (Speedup / N)")
ax.set_xscale("log", base=2)
ax.set_title("Strong Scaling: Parallel Efficiency")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Time Breakdown
# --------------

fig, ax = plt.subplots(figsize=(10, 6))

# TODO: Stacked area or bar showing compute vs communication time
ax.set_xlabel("Number of Ranks")
ax.set_ylabel("Time (s)")
ax.set_xscale("log", base=2)
ax.set_title("Strong Scaling: Compute vs Communication Time")
ax.legend()

plt.tight_layout()
plt.show()

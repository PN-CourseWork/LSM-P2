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

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# TODO: Load data from scaling/compute_scaling.py

# %%
# Weak Scaling: Time Consistency
# -------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

# TODO: Plot total time vs rank count (should be flat for ideal weak scaling)
ax.axhline(y=1, color='k', linestyle='--', alpha=0.3, label='Ideal (T₁)')

ax.set_xlabel("Number of Ranks")
ax.set_ylabel("Normalized Time (T / T₁)")
ax.set_xscale("log", base=2)
ax.set_title("Weak Scaling: Time Consistency")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Weak Scaling: Efficiency
# -------------------------

fig, ax = plt.subplots(figsize=(10, 6))

# TODO: Plot weak scaling efficiency: T(1) / T(N)
ax.axhline(y=1, color='k', linestyle='--', alpha=0.3, label='Ideal')

ax.set_xlabel("Number of Ranks")
ax.set_ylabel("Weak Scaling Efficiency")
ax.set_xscale("log", base=2)
ax.set_title("Weak Scaling: Parallel Efficiency")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Communication Growth
# --------------------

fig, ax = plt.subplots(figsize=(10, 6))

# TODO: Plot communication time fraction vs rank count
ax.set_xlabel("Number of Ranks")
ax.set_ylabel("Communication Time Fraction")
ax.set_xscale("log", base=2)
ax.set_title("Weak Scaling: Communication Overhead Growth")

plt.tight_layout()
plt.show()

#!/usr/bin/env python3
"""
Numba Thread Scaling
====================

Analyze how Numba kernel performance scales with OpenMP thread count.

This experiment investigates:
- Optimal thread count for different problem sizes
- Parallel efficiency (speedup / number of threads)
- Thread overhead vs computational benefit
"""

# %%
# Setup
# -----

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# TODO: Load data from kernels/compute_kernels.py output

# %%
# Thread Scaling Performance
# ---------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# TODO: Plot speedup vs thread count
ax1.set_xlabel("Number of Threads")
ax1.set_ylabel("Speedup (vs 1 thread)")
ax1.plot([1, 8], [1, 8], 'k--', alpha=0.3, label='Ideal')
ax1.set_title("Thread Scaling")
ax1.legend()

# TODO: Plot parallel efficiency
ax2.set_xlabel("Number of Threads")
ax2.set_ylabel("Parallel Efficiency")
ax2.axhline(y=1, color='k', linestyle='--', alpha=0.3)
ax2.set_title("Parallel Efficiency")

plt.tight_layout()
plt.show()

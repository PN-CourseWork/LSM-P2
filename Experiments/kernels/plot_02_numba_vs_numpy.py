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

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# TODO: Load data from kernels/compute_kernels.py output

# %%
# Kernel Performance Comparison
# ------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# TODO: Plot absolute timings
ax1.set_xlabel("Problem Size N")
ax1.set_ylabel("Time per Iteration (s)")
ax1.set_title("Kernel Performance")
ax1.legend()

# TODO: Plot speedup factor
ax2.set_xlabel("Problem Size N")
ax2.set_ylabel("Speedup (Numba / NumPy)")
ax2.axhline(y=1, color='k', linestyle='--', alpha=0.3)
ax2.set_title("Numba Speedup Factor")

plt.tight_layout()
plt.show()

#!/usr/bin/env python3
"""
NumPy Array Communication Performance
======================================

Analysis of NumPy ascontiguousarray() communication performance.

This experiment analyzes:
- Buffer copy overhead (ascontiguousarray)
- Transfer time for ghost exchanges
- Memory overhead (temporary buffers)
- Scaling with problem size
"""

# %%
# Setup
# -----

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# TODO: Load data from communication/compute_communication.py

# %%
# Buffer Copy Overhead
# --------------------

fig, ax = plt.subplots(figsize=(10, 6))

# TODO: Plot ascontiguousarray() copy time
ax.set_xlabel("Problem Size N")
ax.set_ylabel("Buffer Copy Time (s)")
ax.set_title("NumPy Array Copy Overhead")

plt.tight_layout()
plt.show()

# %%
# Ghost Exchange Performance
# --------------------------

fig, ax = plt.subplots(figsize=(10, 6))

# TODO: Plot ghost exchange time per iteration
ax.set_xlabel("Problem Size N")
ax.set_ylabel("Ghost Exchange Time (s)")
ax.set_title("NumPy Arrays: Communication Time")

plt.tight_layout()
plt.show()

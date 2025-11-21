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

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# TODO: Load data from communication/compute_communication.py

# %%
# Total Overhead Comparison
# --------------------------

fig, ax = plt.subplots(figsize=(10, 6))

# TODO: Plot total communication time for both methods
ax.set_xlabel("Problem Size N")
ax.set_ylabel("Communication Time (s)")
ax.set_title("Communication Overhead: Custom MPI vs NumPy Arrays")
ax.legend()

plt.tight_layout()
plt.show()

# %%
# Performance vs Code Complexity
# -------------------------------

# TODO: Qualitative analysis or table showing trade-offs
# - Custom MPI: Faster but more complex
# - NumPy: Simpler but has copy overhead

# %%
# Recommendations
# ---------------

# TODO: Based on data, provide recommendations for:
# - Small vs large problem sizes
# - Development vs production code
# - Simple vs complex decompositions

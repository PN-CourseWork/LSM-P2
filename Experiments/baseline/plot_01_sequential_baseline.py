#!/usr/bin/env python3
"""
Sequential Baseline Performance
================================

Convergence and solution quality for the sequential Jacobi solver.

This experiment establishes baseline metrics:
- Convergence rate vs iteration count
- Solution accuracy (L2 error vs exact solution)
- Computational cost per iteration

All parallel implementations are compared against this baseline.
"""

# %%
# Setup
# -----

import numpy as np
import matplotlib.pyplot as plt

# TODO: Migrate from Legacy/sequential/plot_sequential.py
# TODO: Load data from baseline/compute_sequential.py output

# %%
# Convergence History
# -------------------

fig, ax = plt.subplots(figsize=(8, 6))
# TODO: Plot residual vs iteration
ax.set_xlabel("Iteration")
ax.set_ylabel("Residual")
ax.set_yscale("log")
ax.set_title("Sequential Solver Convergence")
plt.tight_layout()
plt.show()

# %%
# Solution Accuracy
# -----------------

fig, ax = plt.subplots(figsize=(8, 6))
# TODO: Plot solution slice and error
ax.set_title("Sequential Solution Quality")
plt.tight_layout()
plt.show()

"""
Generate Sequential Baseline Data
==================================

This script generates reference data for the sequential Jacobi solver,
which serves as the baseline for all parallel implementations.

Run this script to generate baseline performance data that will be used
by the visualization scripts in this directory.
"""

# %%
# Configuration
# -------------
#
# TODO: Migrate from Legacy/sequential/compute_sequential.py
# TODO: Update to use new data directory structure

# Example configuration (to be implemented):
# - Problem sizes: N = [50, 100, 200, 400]
# - Convergence tolerance: 1e-5
# - Maximum iterations: 10000
# - Kernel: Both NumPy and Numba variants

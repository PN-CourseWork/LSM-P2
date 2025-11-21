"""
Generate Cubic Decomposition Data
==================================

Run MPI cubic decomposition experiments with 3D Cartesian domain
decomposition.

Distributes the grid across all three spatial dimensions, exchanging
6 ghost faces (±X, ±Y, ±Z). This provides better load balance for
large rank counts compared to sliced decomposition.
"""

# %%
# MPI Configuration
# -----------------
#
# TODO: Migrate from Legacy/cubic/compute_cubic.py
# TODO: Test with different problem sizes: N = [100, 200, 400]
# TODO: Test with different rank counts: ranks = [8, 27, 64] (perfect cubes)
# TODO: Record compute time, communication time, halo exchange time

# Run with: mpiexec -n <ranks> python compute_cubic.py

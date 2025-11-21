"""
Generate Kernel Benchmark Data
===============================

Benchmark Numba JIT-compiled kernels vs pure NumPy implementations.

This script compares ``jacobi_step_numba()`` and ``jacobi_step_numpy()``
performance across different problem sizes and thread counts, including
warmup iterations to account for JIT compilation overhead.
"""

# %%
# Benchmark Configuration
# -----------------------
#
# TODO: Migrate from Legacy/Numba-Benchmark/compute_numba_benchmark.py
# TODO: Test with: N = [50, 100, 200, 400]
# TODO: Test with: threads = [1, 2, 4, 8]
# TODO: Include warmup iterations for fair JIT comparison

# Example benchmark parameters:
# - Problem sizes to test
# - Thread counts for Numba
# - Number of iterations per test
# - Warmup iterations before timing

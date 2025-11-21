#!/usr/bin/env python3
"""
Benchmark Numba vs NumPy kernels.

Compares jacobi_step_numba() and jacobi_step_numpy() performance across:
- Different problem sizes (N)
- Different thread counts (Numba only)
- Multiple iterations to account for JIT compilation warmup
"""

# TODO: Migrate from Legacy/Numba-Benchmark/compute_numba_benchmark.py
# TODO: Test with: N = [50, 100, 200, 400]
# TODO: Test with: threads = [1, 2, 4, 8]
# TODO: Include warmup iterations for fair JIT comparison

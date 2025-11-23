"""
Generate Kernel Validation Data
=================================

Validation experiments comparing NumPy and Numba kernel implementations.

**Iterative Convergence:**
  Verifies that NumPy and Numba kernels produce identical convergence behavior
  by tracking residual history over iterations for a fixed problem size.
"""

# %%
# Imports
# -------

import numpy as np
from pathlib import Path
from mpi4py import MPI
from Poisson import JacobiPoisson
from utils import datatools

# %%
# MPI Setup
# ---------

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Only run on single rank
if size > 1:
    if rank == 0:
        print("ERROR: This script must be run with a single MPI rank")
        print("Usage: uv run python compute_kernel_validation.py")
    exit(1)

# %%
# Test Parameters
# ---------------

N = 50  # Problem size
omega = 0.75
max_iter = 500  # Enough to see convergence behavior

# %%
# Initialize Results Storage
# ---------------------------

data_dir = datatools.get_data_dir()
h5_dir = data_dir / "validation_h5"
h5_dir.mkdir(parents=True, exist_ok=True)

if rank == 0:
    print("Kernel Comparison: NumPy vs Numba")
    print("=" * 60)

# %%
# Test NumPy Kernel
# -----------------

if rank == 0:
    print(f"\nNumPy Kernel (N={N})")
    print("-" * 60)

solver_numpy = JacobiPoisson(N=N, omega=omega, max_iter=max_iter, use_numba=False)
solver_numpy.solve()
solver_numpy.summary()

h5_file_numpy = h5_dir / f"kernel_numpy_N{N}.h5"
solver_numpy.save_hdf5(h5_file_numpy)

if rank == 0:
    res = solver_numpy.results
    print(f"  Iterations: {res.iterations}")
    print(f"  Converged: {res.converged}")
    print(f"  Final error (L2): {res.final_error:.4e}")
    print(f"  Saved to: {h5_file_numpy}")

# %%
# Test Numba Kernel
# -----------------

if rank == 0:
    print(f"\nNumba Kernel (N={N})")
    print("-" * 60)

solver_numba = JacobiPoisson(N=N, omega=omega, max_iter=max_iter, use_numba=True)

# Warmup JIT compilation
if rank == 0:
    print("  Warming up Numba JIT...")
solver_numba.warmup(N=10)

solver_numba.solve()
solver_numba.summary()

h5_file_numba = h5_dir / f"kernel_numba_N{N}.h5"
solver_numba.save_hdf5(h5_file_numba)

if rank == 0:
    res = solver_numba.results
    print(f"  Iterations: {res.iterations}")
    print(f"  Converged: {res.converged}")
    print(f"  Final error (L2): {res.final_error:.4e}")
    print(f"  Saved to: {h5_file_numba}")

# %%
# Summary
# -------

if rank == 0:
    print("\n" + "=" * 60)
    print("Kernel validation data generated successfully!")
    print(f"HDF5 files: {h5_dir}")
    print("=" * 60)

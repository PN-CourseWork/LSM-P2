"""
Generate Solver Validation Data
================================

Validation experiments to verify solver correctness and convergence properties.

**Spatial Convergence:**
  Tests solution accuracy vs problem size (N) by comparing numerical solutions
  against the exact analytical solution: u(x,y,z) = sin(πx)sin(πy)sin(πz).

**Iterative Convergence:**
  Verifies Jacobi iteration convergence by tracking residual decrease over
  iterations for a fixed problem size.
"""

# %%
# Imports
# -------

import numpy as np
import pandas as pd
from pathlib import Path
from mpi4py import MPI
from Poisson import (
    JacobiPoisson,
    PostProcessor,
    SlicedDecomposition,
    CubicDecomposition,
    CustomMPICommunicator,
    NumpyCommunicator,
)
from utils import datatools

# %%
# MPI Setup
# ---------

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# %%
# Test Parameters
# ---------------
#
# Test problem sizes for spatial convergence: N = [25, 50]
# Use tighter tolerance to see full convergence behavior

problem_sizes = [25, 50]
omega = 0.75
max_iter = 5000

# %%
# Initialize Results Storage
# ---------------------------

# Create output directory for HDF5 files
data_dir = datatools.get_data_dir()
h5_dir = data_dir / "validation_h5"
h5_dir.mkdir(parents=True, exist_ok=True)

hdf5_files = []  # Will collect HDF5 file paths

if rank == 0:
    print("Spatial Convergence Analysis")
    print("=" * 60)

# %%
# Sequential Solver Tests
# ------------------------
#
# Test JacobiPoisson without decomposition (sequential mode).
# Only runs when script is executed with a single MPI rank.

if size == 1:
    if rank == 0:
        print("\nSequential (NumPy)")
        print("=" * 60)

    for N in problem_sizes:
        if rank == 0:
            print(f"\nTesting N={N} (h={2.0/(N-1):.6f})")
            print("-" * 60)

        # JacobiPoisson with no decomposition = sequential
        solver = JacobiPoisson(N=N, omega=omega, max_iter=max_iter, use_numba=False)
        solver.solve()
        solver.summary()

        # Save to HDF5
        h5_file = h5_dir / f"sequential_N{N}.h5"
        solver.save_hdf5(h5_file)

        if rank == 0:
            hdf5_files.append(h5_file)

            res = solver.results
            print(f"  Iterations: {res.iterations}")
            print(f"  Converged: {res.converged}")
            print(f"  Final error (L2): {res.final_error:.4e}")
            print(f"  Saved to: {h5_file}")
elif rank == 0:
    print("\nSkipping Sequential tests (use single rank for sequential)")
    print("=" * 60)

# %%
# MPI Solver Tests
# ----------------
#
# Test all combinations of:
# - Decomposition strategies: Sliced (1D) vs Cubic (3D)
# - Communication methods: Custom MPI datatypes vs NumPy arrays

mpi_methods = [
    ("MPI_Sliced_Custom", SlicedDecomposition(), CustomMPICommunicator()),
    ("MPI_Sliced_Numpy", SlicedDecomposition(), NumpyCommunicator()),
    ("MPI_Cubic_Custom", CubicDecomposition(), CustomMPICommunicator()),
    ("MPI_Cubic_Numpy", CubicDecomposition(), NumpyCommunicator()),
]

for method_name, decomposition, communicator in mpi_methods:
    if rank == 0:
        print(f"\n{method_name}")
        print("=" * 60)

    for N in problem_sizes:
        if rank == 0:
            print(f"\nTesting N={N} (h={2.0/(N-1):.6f})")
            print("-" * 60)

        # JacobiPoisson with decomposition = distributed (new cleaner API!)
        solver = JacobiPoisson(
            decomposition=decomposition,
            communicator=communicator,
            N=N,
            omega=omega,
            max_iter=max_iter,
        )
        solver.solve()
        solver.summary()

        # Save to HDF5 (all ranks participate in parallel write)
        h5_file = h5_dir / f"{method_name}_N{N}_np{size}.h5"
        solver.save_hdf5(h5_file)

        if rank == 0:
            hdf5_files.append(h5_file)

            res = solver.results
            print(f"  Iterations: {res.iterations}")
            print(f"  Converged: {res.converged}")
            print(f"  Final error (L2): {res.final_error:.4e}")
            print(f"  Saved to: {h5_file}")

# %%
# Summary
# -------

if rank == 0:
    print("\n" + "=" * 60)
    print("Validation data generated successfully!")
    print(f"HDF5 files: {h5_dir}")
    print(f"Generated {len(hdf5_files)} HDF5 result files")
    print("=" * 60)

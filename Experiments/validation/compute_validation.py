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
from mpi4py import MPI
from Poisson import JacobiPoisson
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

spatial_results = []  # Will collect DataFrames

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

        if rank == 0:
            # Collect results as DataFrame
            import pandas as pd
            df_result = pd.concat([
                solver._dataclass_to_df(solver.config),
                solver._dataclass_to_df(solver.global_results)
            ], axis=1)
            spatial_results.append(df_result)

            res = solver.global_results
            print(f"  Iterations: {res.iterations}")
            print(f"  Converged: {res.converged}")
            print(f"  Final error (L2): {res.final_error:.4e}")
            print(f"  Wall time: {res.wall_time:.4f}s")
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
    ("MPI_Sliced_CustomMPI", "sliced", "custom"),
    ("MPI_Sliced_Numpy", "sliced", "numpy"),
    ("MPI_Cubic_CustomMPI", "cubic", "custom"),
    ("MPI_Cubic_Numpy", "cubic", "numpy"),
]

for method_name, decomposition, communicator in mpi_methods:
    if rank == 0:
        print(f"\n{method_name}")
        print("=" * 60)

    for N in problem_sizes:
        if rank == 0:
            print(f"\nTesting N={N} (h={2.0/(N-1):.6f})")
            print("-" * 60)

        # JacobiPoisson with decomposition = distributed
        solver = JacobiPoisson(
            decomposition=decomposition,
            communicator=communicator,
            N=N,
            omega=omega,
            max_iter=max_iter,
        )
        solver.solve()
        solver.summary()

        if rank == 0:
            # Collect results as DataFrame
            df_result = pd.concat([
                solver._dataclass_to_df(solver.config),
                solver._dataclass_to_df(solver.global_results)
            ], axis=1)
            spatial_results.append(df_result)

            res = solver.global_results
            print(f"  Iterations: {res.iterations}")
            print(f"  Converged: {res.converged}")
            print(f"  Final error (L2): {res.final_error:.4e}")
            print(f"  Wall time: {res.wall_time:.4f}s")

# %%
# Save Results
# ------------
#
# Concatenate all DataFrames and save to parquet format.
# Only rank 0 performs the save operation.

if rank == 0:
    df_spatial = pd.concat(spatial_results, ignore_index=True)
    data_dir = datatools.get_data_dir()
    datatools.save_simulation_data(
        df_spatial,
        data_dir / "spatial_convergence.parquet",
        format="parquet"
    )

# %%
# Summary
# -------

if rank == 0:
    print("\n" + "=" * 60)
    print("Validation data generated successfully!")
    print(f"Spatial convergence: {data_dir / 'spatial_convergence.parquet'}")

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
# Configuration
# -------------
#
# Test problem sizes for spatial convergence: N = [25, 50, 75, 100]
# Fixed problem size for iterative convergence: N = 100
# Use tighter tolerance to see full convergence behavior

import numpy as np
import pandas as pd
from mpi4py import MPI

from Poisson import JacobiPoisson
from utils import datatools

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# %%
# Spatial Convergence Test
# -------------------------
#
# Measure solution error vs grid size to verify spatial convergence rate
# for different solver methods (Sequential and MPI with different strategies).
# Expected: O(h²) convergence for second-order finite differences.

if rank == 0:
    print("Spatial Convergence Analysis")
    print("=" * 60)

# Test parameters
problem_sizes = [25, 50]
omega = 0.75
max_iter = 5000

spatial_results = []

# Test Sequential solver (NumPy kernel only) - only when running with single rank
size = comm.Get_size()
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
            spatial_results.append(solver.to_dict())
            res = solver.global_results
            print(f"  Iterations: {res.iterations}")
            print(f"  Converged: {res.converged}")
            print(f"  Final error (L2): {res.final_error:.4e}")
            print(f"  Wall time: {res.wall_time:.4f}s")
elif rank == 0:
    print("\nSkipping Sequential tests (use single rank for sequential)")
    print("=" * 60)

# Test all MPI solver combinations
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
            spatial_results.append(solver.to_dict())
            res = solver.global_results
            print(f"  Iterations: {res.iterations}")
            print(f"  Converged: {res.converged}")
            print(f"  Final error (L2): {res.final_error:.4e}")
            print(f"  Wall time: {res.wall_time:.4f}s")

# Save spatial convergence data (only on rank 0)
if rank == 0:
    df_spatial = pd.DataFrame(spatial_results)
    data_dir = datatools.get_data_dir()
    datatools.save_simulation_data(
        df_spatial,
        data_dir / "spatial_convergence.parquet",
        format="parquet"
    )

if rank == 0:
    print("\n" + "=" * 60)
    print("Validation data generated successfully!")
    print(f"Spatial convergence: {data_dir / 'spatial_convergence.parquet'}")

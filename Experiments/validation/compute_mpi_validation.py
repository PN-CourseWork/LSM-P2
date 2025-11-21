"""
Generate MPI Solver Validation Data
====================================

Validation experiments for modular MPI solvers to verify correctness
and compare different domain decomposition and communication strategies.

Must be run with mpiexec:
    mpiexec -n 2 python compute_mpi_validation.py
    mpiexec -n 4 python compute_mpi_validation.py

**Methods Tested:**
  - MPI_Sliced_CustomMPI (1D decomposition + MPI datatypes)
  - MPI_Sliced_Numpy (1D decomposition + NumPy arrays)
  - MPI_Cubic_CustomMPI (3D decomposition + MPI datatypes)
  - MPI_Cubic_Numpy (3D decomposition + NumPy arrays)
"""

import numpy as np
import pandas as pd
from mpi4py import MPI

from Poisson import MPIJacobi
from utils import datatools

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Test parameters
problem_sizes = [25, 50, 75]
omega = 0.75
max_iter = 20000
tolerance = 1e-12

# Define all methods to test
methods = [
    ("MPI_Sliced_CustomMPI", "sliced", "custom"),
    ("MPI_Sliced_Numpy", "sliced", "numpy"),
    ("MPI_Cubic_CustomMPI", "cubic", "custom"),
    ("MPI_Cubic_Numpy", "cubic", "numpy"),
]

if rank == 0:
    print(f"MPI Validation with {size} processes")
    print("=" * 60)

spatial_results = []

for method_name, decomposition, communicator in methods:
    if rank == 0:
        print(f"\n{method_name}")
        print("=" * 60)

    for N in problem_sizes:
        if rank == 0:
            print(f"\nTesting N={N} (h={2.0/(N-1):.6f})")
            print("-" * 60)

        # Create and solve
        solver = MPIJacobi(
            decomposition=decomposition,
            communicator=communicator,
            N=N,
            omega=omega,
            max_iter=max_iter,
            tolerance=tolerance,
            use_numba=False,  # Use NumPy kernel for consistency
        )
        solver.solve()

        # Gather results on rank 0
        if rank == 0:
            res = solver.global_results
            h = 2.0 / (N - 1)

            result = {
                "N": N,
                "h": h,
                "method": method_name,
                "iterations": res.iterations,
                "converged": res.converged,
                "final_error": res.final_error,
                "wall_time": res.wall_time,
            }
            spatial_results.append(result)

            print(f"  Iterations: {res.iterations}")
            print(f"  Converged: {res.converged}")
            print(f"  Final error (L2): {res.final_error:.4e}")
            print(f"  Wall time: {res.wall_time:.4f}s")

# Save results on rank 0
if rank == 0:
    df_spatial = pd.DataFrame(spatial_results)
    data_dir = datatools.get_data_dir()
    datatools.save_simulation_data(
        df_spatial,
        data_dir / f"mpi_validation_np{size}.parquet",
        format="parquet"
    )

    print("\n" + "=" * 60)
    print("MPI validation data generated successfully!")
    print(f"Results: {data_dir / f'mpi_validation_np{size}.parquet'}")

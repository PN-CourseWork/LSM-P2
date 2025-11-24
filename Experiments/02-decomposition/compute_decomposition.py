"""
Generate Decomposition Comparison Data
========================================

Compare Sliced (1D) vs Cubic (3D) domain decompositions.

Tests how communication and computation scale with problem size for different
decomposition strategies. Run with mpiexec to test MPI implementations.
"""

# %%
# Decomposition Experiments
# --------------------------
#
# Compare Sliced vs Cubic decompositions across problem sizes
# Measure: compute time, communication time, halo exchange time

import numpy as np
import pandas as pd
from mpi4py import MPI

from Poisson import JacobiPoisson
from utils import datatools

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    print(f"Decomposition Comparison with {size} processes")
    print("=" * 60)

# Test parameters (small sizes for now)
problem_sizes = [25, 50, 75]
omega = 0.75
max_iter = 1000
tolerance = 1e-5

# Decomposition strategies to test
decompositions = [
    ("Sliced", "sliced"),
    ("Cubic", "cubic"),
]

results = []

for decomp_name, decomp_type in decompositions:
    if rank == 0:
        print(f"\n{decomp_name} Decomposition")
        print("=" * 60)

    for N in problem_sizes:
        if rank == 0:
            print(f"\nTesting N={N} (h={2.0/(N-1):.6f})")
            print("-" * 60)

        # Create solver with decomposition
        solver = JacobiPoisson(
            decomposition=decomp_type,
            communicator="numpy",  # Use numpy communicator
            N=N,
            omega=omega,
            max_iter=max_iter,
            tolerance=tolerance,
            use_numba=False,
        )

        solver.solve()
        solver.summary()

        if rank == 0:
            # Get base results as DataFrame
            df_result = pd.concat([
                solver._dataclass_to_df(solver.config),
                solver._dataclass_to_df(solver.global_results)
            ], axis=1)

            # Add decomposition-specific fields
            df_result['decomposition'] = decomp_name
            df_result['mpi_size'] = size
            df_result['compute_time'] = solver.global_results.compute_time
            df_result['mpi_comm_time'] = solver.global_results.mpi_comm_time
            df_result['halo_exchange_time'] = solver.global_results.halo_exchange_time

            results.append(df_result)

            res = solver.global_results
            print(f"  Iterations: {res.iterations}")
            print(f"  Converged: {res.converged}")
            print(f"  Wall time: {res.wall_time:.4f}s")
            print(f"  Compute time: {res.compute_time:.4f}s")
            print(f"  Halo exchange time: {res.halo_exchange_time:.4f}s")

# Save results on rank 0
if rank == 0:
    df = pd.concat(results, ignore_index=True)
    data_dir = datatools.get_data_dir()
    output_path = data_dir / f"decomposition_comparison_np{size}.parquet"

    datatools.save_simulation_data(df, output_path, format="parquet")

    print("\n" + "=" * 60)
    print("Decomposition comparison data generated successfully!")
    print(f"Saved to: {output_path}")

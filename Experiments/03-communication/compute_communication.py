"""
Generate Communication Method Data
===================================

Compare Custom MPI datatypes vs NumPy array communication strategies.

Tests communication overhead for both Sliced and Cubic decompositions.
Run with mpiexec to test MPI implementations.
"""

# %%
# Communication Benchmarks
# ------------------------
#
# Compare: Custom MPI datatypes vs NumPy array communication
# Measure: total time, halo exchange time, communication overhead

import numpy as np
import pandas as pd
from mpi4py import MPI

from Poisson import JacobiPoisson
from utils import datatools

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    print(f"Communication Method Comparison with {size} processes")
    print("=" * 60)

# Test parameters (small sizes for now)
problem_sizes = [25, 50, 75]
omega = 0.75
max_iter = 1000
tolerance = 1e-5

# Test all combinations of decomposition and communicator
methods = [
    ("Sliced_CustomMPI", "sliced", "custom"),
    ("Sliced_Numpy", "sliced", "numpy"),
    ("Cubic_CustomMPI", "cubic", "custom"),
    ("Cubic_Numpy", "cubic", "numpy"),
]

results = []

for method_name, decomp, comm_type in methods:
    if rank == 0:
        print(f"\n{method_name}")
        print("=" * 60)

    for N in problem_sizes:
        if rank == 0:
            print(f"\nTesting N={N} (h={2.0/(N-1):.6f})")
            print("-" * 60)

        # Create solver
        solver = JacobiPoisson(
            decomposition=decomp,
            communicator=comm_type,
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

            # Add communication-specific fields
            df_result['method'] = method_name
            df_result['decomposition'] = decomp
            df_result['communicator'] = comm_type
            df_result['mpi_size'] = size
            df_result['compute_time'] = solver.global_results.compute_time
            df_result['mpi_comm_time'] = solver.global_results.mpi_comm_time
            df_result['halo_exchange_time'] = solver.global_results.halo_exchange_time
            comm_overhead_pct = (
                100 * solver.global_results.halo_exchange_time / solver.global_results.wall_time
                if solver.global_results.wall_time > 0 else 0
            )
            df_result['comm_overhead_pct'] = comm_overhead_pct

            results.append(df_result)

            res = solver.global_results
            print(f"  Iterations: {res.iterations}")
            print(f"  Wall time: {res.wall_time:.4f}s")
            print(f"  Compute time: {res.compute_time:.4f}s")
            print(f"  Halo exchange time: {res.halo_exchange_time:.4f}s")
            print(f"  Comm overhead: {comm_overhead_pct:.1f}%")

# Save results on rank 0
if rank == 0:
    df = pd.concat(results, ignore_index=True)
    data_dir = datatools.get_data_dir()
    output_path = data_dir / f"communication_comparison_np{size}.parquet"

    datatools.save_simulation_data(df, output_path, format="parquet")

    print("\n" + "=" * 60)
    print("Communication comparison data generated successfully!")
    print(f"Saved to: {output_path}")

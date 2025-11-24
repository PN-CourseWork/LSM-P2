"""
Generate Scaling Analysis Data
===============================

Run strong and weak scaling experiments for MPI implementations.

**Strong Scaling:** Fixed problem size with increasing ranks to measure parallel speedup.
**Weak Scaling:** Problem size grows with ranks (constant work per rank) to measure efficiency.

Run with different MPI rank counts: mpiexec -n 2/4/6 python compute_scaling.py
"""

# %%
# Scaling Experiments
# -------------------
#
# Strong scaling: Fixed N, varying ranks
# Weak scaling: N³/ranks = constant

import numpy as np
import pandas as pd
from mpi4py import MPI

from Poisson import JacobiPoisson
from utils import datatools

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    print(f"Scaling Analysis with {size} processes")
    print("=" * 60)

# Test parameters
omega = 0.75
max_iter = 1000
tolerance = 1e-5

results = []

# %%
# Strong Scaling
# --------------
# Fixed problem size, varying number of ranks
# Run this script with mpiexec -n 1/2/4/6/8

if rank == 0:
    print("\nStrong Scaling Test")
    print("=" * 60)

# Fixed problem size for strong scaling
N_strong = 50

for decomp_name, decomp_type in [("Sliced", "sliced"), ("Cubic", "cubic")]:
    if rank == 0:
        print(f"\n{decomp_name} Decomposition, N={N_strong}")
        print("-" * 60)

    solver = JacobiPoisson(
        decomposition=decomp_type,
        communicator="numpy",
        N=N_strong,
        omega=omega,
        max_iter=max_iter,
        tolerance=tolerance,
        use_numba=False,
    )

    solver.solve()
    solver.summary()

    if rank == 0:
        df_result = pd.concat([
            solver._dataclass_to_df(solver.config),
            solver._dataclass_to_df(solver.global_results)
        ], axis=1)
        df_result['scaling_type'] = 'strong'
        df_result['decomposition'] = decomp_name
        df_result['mpi_size'] = size
        df_result['compute_time'] = solver.global_results.compute_time
        df_result['mpi_comm_time'] = solver.global_results.mpi_comm_time
        df_result['halo_exchange_time'] = solver.global_results.halo_exchange_time
        results.append(df_result)

        res = solver.global_results
        print(f"  Wall time: {res.wall_time:.4f}s")
        print(f"  Compute time: {res.compute_time:.4f}s")
        print(f"  Halo exchange time: {res.halo_exchange_time:.4f}s")

# %%
# Weak Scaling
# ------------
# Constant work per rank: N³/ranks = constant
# For weak scaling, adjust N based on number of ranks

if rank == 0:
    print("\n\nWeak Scaling Test")
    print("=" * 60)

# Base work per rank (N=25 for 1 rank gives N³=15625)
base_work = 25**3

# Calculate N for this number of ranks to maintain constant work per rank
N_weak = int((base_work * size) ** (1/3))

if rank == 0:
    print(f"\nTarget work per rank: {base_work} cells")
    print(f"Calculated N for {size} ranks: {N_weak}")

for decomp_name, decomp_type in [("Sliced", "sliced"), ("Cubic", "cubic")]:
    if rank == 0:
        print(f"\n{decomp_name} Decomposition, N={N_weak}")
        print("-" * 60)

    solver = JacobiPoisson(
        decomposition=decomp_type,
        communicator="numpy",
        N=N_weak,
        omega=omega,
        max_iter=max_iter,
        tolerance=tolerance,
        use_numba=False,
    )

    solver.solve()
    solver.summary()

    if rank == 0:
        df_result = pd.concat([
            solver._dataclass_to_df(solver.config),
            solver._dataclass_to_df(solver.global_results)
        ], axis=1)
        df_result['scaling_type'] = 'weak'
        df_result['decomposition'] = decomp_name
        df_result['mpi_size'] = size
        df_result['work_per_rank'] = N_weak**3 / size
        df_result['compute_time'] = solver.global_results.compute_time
        df_result['mpi_comm_time'] = solver.global_results.mpi_comm_time
        df_result['halo_exchange_time'] = solver.global_results.halo_exchange_time
        results.append(df_result)

        res = solver.global_results
        print(f"  Wall time: {res.wall_time:.4f}s")
        print(f"  Compute time: {res.compute_time:.4f}s")
        print(f"  Halo exchange time: {res.halo_exchange_time:.4f}s")

# Save results on rank 0
if rank == 0:
    df = pd.concat(results, ignore_index=True)
    data_dir = datatools.get_data_dir()
    output_path = data_dir / f"scaling_analysis_np{size}.parquet"

    datatools.save_simulation_data(df, output_path, format="parquet")

    print("\n" + "=" * 60)
    print("Scaling analysis data generated successfully!")
    print(f"Saved to: {output_path}")

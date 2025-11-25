"""
Generate Solver Validation Data
================================

Validation experiments to verify solver correctness and convergence properties.

**Spatial Convergence:**
  Tests solution accuracy vs problem size (N) by comparing numerical solutions
  against the exact analytical solution: u(x,y,z) = sin(πx)sin(πy)sin(πz).

Tests all combinations of:
- Decomposition strategies: Sliced (1D) vs Cubic (3D)
- Communication methods: NumPy arrays vs MPI datatypes
"""

# %%
# Imports
# -------

import pandas as pd
from pathlib import Path
from mpi4py import MPI
from Poisson import (
    JacobiPoisson,
    DomainDecomposition,
    NumpyHaloExchange,
    DatatypeCommunicator,
)

# %%
# MPI Setup
# ---------

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# %%
# Get repository paths
# --------------------

repo_root = Path(__file__).resolve().parent.parent.parent
data_dir = repo_root / "data" / "validation"
data_dir.mkdir(parents=True, exist_ok=True)

# %%
# Test Parameters
# ---------------

problem_sizes = [20, 40, 60]
omega = 1.00
max_iter = 10000
tol = 1e-5

if rank == 0:
    print("Spatial Convergence Validation")
    print("=" * 60)
    print(f"Problem sizes: {problem_sizes}")
    print(f"Ranks: {size}")
    print("=" * 60)

# %%
# Test All Configurations
# -----------------------

results = []

configurations = [
    ('sliced', 'numpy'),
    ('sliced', 'datatype'),
    ('cubic', 'numpy'),
    ('cubic', 'datatype'),
]

for strategy, comm_type in configurations:
    comm_name = comm_type.capitalize()

    if rank == 0:
        print(f"\n{strategy.capitalize()} + {comm_name}")
        print("=" * 60)

    # Create communicator
    if comm_type == 'numpy':
        communicator = NumpyHaloExchange()
    else:
        communicator = DatatypeCommunicator()

    for N in problem_sizes:
        h = 2.0 / (N - 1)

        if rank == 0:
            print(f"\n  N={N} (h={h:.6f})")

        # Create decomposition
        decomposition = DomainDecomposition(N=N, size=size, strategy=strategy)

        # Create and run solver
        solver = JacobiPoisson(
            N=N,
            omega=omega,
            max_iter=max_iter,
            tolerance=tol,
            decomposition=decomposition,
            communicator=communicator,
        )
        solver.solve()
        solver.summary()  # Computes L2 error against exact solution

        if rank == 0:
            error = solver.results.final_error
            iterations = solver.results.iterations

            results.append({
                'strategy': strategy,
                'communicator': comm_type,
                'N': N,
                'h': h,
                'error': error,
                'iterations': iterations,
                'size': size
            })

            print(f"    L2 error: {error:.4e}")

# %%
# Save Results
# ------------

if rank == 0:
    df = pd.DataFrame(results)
    output_file = data_dir / f"validation_np{size}.parquet"
    df.to_parquet(output_file)

    print("\n" + "=" * 60)
    print("Validation complete!")
    print(f"Saved: {output_file}")
    print("=" * 60)
    print(f"\nGenerated {len(results)} results")

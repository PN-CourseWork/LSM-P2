"""
Solver Validation: Fixed Iterations with Analytical Comparison
===============================================================

Validates solver correctness by running a fixed number of iterations and
comparing the global solution against the exact analytical solution.

Key validation criteria:
1. All decomposition/communication methods produce the SAME L2 error
2. L2 error shows O(h²) spatial convergence as N increases
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
    DomainDecomposition,
    NumpyHaloExchange,
    DatatypeCommunicator,
)
from Poisson.problems import sinusoidal_exact_solution


# %%
# Error Computation
# -----------------

def compute_l2_error(u_numerical, u_exact, h):
    """Compute L2 error between numerical and exact solutions.

    Parameters
    ----------
    u_numerical : np.ndarray
        Numerical solution array (N, N, N)
    u_exact : np.ndarray
        Exact solution array (N, N, N)
    h : float
        Grid spacing

    Returns
    -------
    float
        Discrete L2 norm of the error (interior points only)
    """
    error_diff = u_numerical[1:-1, 1:-1, 1:-1] - u_exact[1:-1, 1:-1, 1:-1]
    return float(np.sqrt(h**3 * np.sum(error_diff**2)))

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

problem_sizes = [16, 32, 48, 64]
omega = 1.0
fixed_iterations = 5000  # Same iterations for all methods
tol = 1e-12  # Very tight - ensures we always hit max_iter

if rank == 0:
    print("Solver Validation: Fixed Iterations")
    print("=" * 60)
    print(f"Problem sizes: {problem_sizes}")
    print(f"Ranks: {size}")
    print(f"Fixed iterations: {fixed_iterations}")
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

for N in problem_sizes:
    h = 2.0 / (N - 1)

    if rank == 0:
        print(f"\nN={N} (h={h:.4f})")
        print("-" * 40)

    errors_for_N = {}

    for strategy, comm_type in configurations:
        # Create communicator
        if comm_type == 'numpy':
            communicator = NumpyHaloExchange()
        else:
            communicator = DatatypeCommunicator()

        # Create decomposition
        decomposition = DomainDecomposition(N=N, size=size, strategy=strategy)

        # Create and run solver with fixed iterations
        solver = JacobiPoisson(
            N=N,
            omega=omega,
            max_iter=fixed_iterations,
            tolerance=tol,
            decomposition=decomposition,
            communicator=communicator,
        )
        solver.solve()

        # Compute L2 error against analytical solution (rank 0 only)
        if rank == 0:
            u_exact = sinusoidal_exact_solution(N)
            error = compute_l2_error(solver.u_global, u_exact, h)

            config_name = f"{strategy}+{comm_type}"
            errors_for_N[config_name] = error

            results.append({
                'strategy': strategy,
                'communicator': comm_type,
                'N': N,
                'h': h,
                'error': error,
                'iterations': fixed_iterations,
                'size': size
            })

            print(f"  {config_name:20s}: L2 error = {error:.6e}")

    # Verify all methods give same error (within tolerance)
    if rank == 0:
        error_values = list(errors_for_N.values())
        max_diff = max(error_values) - min(error_values)
        rel_diff = max_diff / np.mean(error_values) if np.mean(error_values) > 0 else 0

        if rel_diff < 0.01:  # 1% tolerance
            print(f"  ✓ All methods agree (rel diff: {rel_diff:.2e})")
        else:
            print(f"  ✗ Methods DISAGREE (rel diff: {rel_diff:.2e})")

# %%
# Save Results and Verify O(h²) Convergence
# -----------------------------------------

if rank == 0:
    df = pd.DataFrame(results)
    output_file = data_dir / f"validation_np{size}.parquet"
    df.to_parquet(output_file)

    print("\n" + "=" * 60)
    print("Convergence Analysis (using sliced+numpy as reference)")
    print("=" * 60)

    ref = df[(df['strategy'] == 'sliced') & (df['communicator'] == 'numpy')]
    if len(ref) >= 2:
        # Compute convergence rate from consecutive pairs
        for i in range(len(ref) - 1):
            h1, e1 = ref.iloc[i]['h'], ref.iloc[i]['error']
            h2, e2 = ref.iloc[i + 1]['h'], ref.iloc[i + 1]['error']
            if e1 > 0 and e2 > 0:
                rate = np.log(e1 / e2) / np.log(h1 / h2)
                print(f"  N={ref.iloc[i]['N']:3d} → N={ref.iloc[i+1]['N']:3d}: rate = {rate:.2f}")

    print(f"\nSaved: {output_file}")
    print(f"Generated {len(results)} results")

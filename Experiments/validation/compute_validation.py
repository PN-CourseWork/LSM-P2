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

from Poisson import SequentialJacobi
from utils import datatools

# %%
# Spatial Convergence Test
# -------------------------
#
# Measure solution error vs grid size to verify spatial convergence rate
# for different solver methods (NumPy vs Numba kernels).
# Expected: O(h²) convergence for second-order finite differences.

print("Spatial Convergence Analysis")
print("=" * 60)

# Test parameters
problem_sizes = [25, 50, 75, 100]
omega = 0.75
max_iter = 20000
tolerance = 1e-12  # Very tight tolerance to ensure full convergence
kernel_types = [("NumPy", False), ("Numba", True)]

spatial_results = []

for kernel_name, use_numba_kernel in kernel_types:
    print(f"\n{kernel_name} Kernel")
    print("=" * 60)

    for N in problem_sizes:
        print(f"\nTesting N={N} (h={2.0/(N-1):.6f})")
        print("-" * 60)

        # Create and solve
        solver = SequentialJacobi(
            N=N,
            omega=omega,
            max_iter=max_iter,
            tolerance=tolerance,
            use_numba=use_numba_kernel,
        )

        # Warmup for numba
        if use_numba_kernel:
            solver.warmup(N=10)

        solver.solve()

        # Extract results
        res = solver.global_results
        h = 2.0 / (N - 1)

        result = {
            "N": N,
            "h": h,
            "method": f"Sequential ({kernel_name})",
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

# Save spatial convergence data
df_spatial = pd.DataFrame(spatial_results)
data_dir = datatools.get_data_dir()
datatools.save_simulation_data(
    df_spatial,
    data_dir / "spatial_convergence.parquet",
    format="parquet"
)

# %%
# Iterative Convergence Test
# ---------------------------
#
# Track residual history for fixed problem size comparing NumPy vs Numba kernels.

print("\n" + "=" * 60)
print("Iterative Convergence Analysis")
print("=" * 60)

# Fixed problem size
N_fixed = 100
kernel_types = [("numpy", False), ("numba", True)]

iterative_results = []

for kernel_name, use_numba_kernel in kernel_types:
    print(f"\nTesting {kernel_name} kernel (N={N_fixed})")
    print("-" * 60)

    # Create solver
    solver = SequentialJacobi(
        N=N_fixed,
        omega=omega,
        max_iter=max_iter,
        tolerance=1e-8,
        use_numba=use_numba_kernel,
    )

    # Warmup for numba
    if use_numba_kernel:
        print("  Warming up Numba JIT...")
        solver.warmup(N=10)

    solver.solve()

    # Extract iteration history
    res = solver.global_results
    residual_history = solver.global_timeseries.residual_history

    print(f"  Total iterations: {res.iterations}")
    print(f"  Converged: {res.converged}")
    print(f"  Final residual: {residual_history[-1]:.4e}")
    print(f"  Final error: {res.final_error:.4e}")

    # Add to results
    for iteration, residual in enumerate(residual_history):
        iterative_results.append({
            "iteration": iteration + 1,
            "residual": residual,
            "N": N_fixed,
            "kernel": kernel_name,
        })

df_iterative = pd.DataFrame(iterative_results)

# Save iterative convergence data
datatools.save_simulation_data(
    df_iterative,
    data_dir / "iterative_convergence.parquet",
    format="parquet"
)

print("\n" + "=" * 60)
print("Validation data generated successfully!")
print(f"Spatial convergence: {data_dir / 'spatial_convergence.parquet'}")
print(f"Iterative convergence: {data_dir / 'iterative_convergence.parquet'}")

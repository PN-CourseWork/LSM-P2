"""
Kernel Convergence Validation
==============================

Compare NumPy vs Numba kernel convergence behavior for a fixed problem size.

This validates that both kernels produce identical convergence by tracking
the physical error ||u - u_exact|| against the analytical solution.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from Poisson import problems
from Poisson.kernels import jacobi_step_numpy, jacobi_step_numba
from utils import datatools

print("Kernel Convergence Comparison")
print("=" * 60)

# Fixed small problem for convergence validation
N = 50
omega = 0.75
max_iter = 500
tolerance = 1e-6

# Setup problem
h = 2.0 / (N - 1)
u_exact = problems.sinusoidal_exact_solution(N)
f = problems.sinusoidal_source_term(N)

# Test both kernels
kernels = [
    ('numpy', jacobi_step_numpy),
    ('numba', jacobi_step_numba),
]

results = []

for kernel_name, kernel_func in kernels:
    print(f"\nTesting {kernel_name} kernel...")
    print("-" * 60)

    # Initialize solution
    u = np.zeros((N, N, N), dtype=np.float64)  # Zero initial guess
    u_old = u.copy()

    # Warmup for Numba
    if kernel_name == 'numba':
        print("  Warming up Numba JIT...")
        u_warmup = np.zeros((10, 10, 10), dtype=np.float64)
        u_old_warmup = u_warmup.copy()
        f_warmup = np.zeros((10, 10, 10), dtype=np.float64)
        h_warmup = 2.0 / 9
        for _ in range(5):
            kernel_func(u_old_warmup, u_warmup, f_warmup, h_warmup, omega)

    print("  Iterating...")

    # Manual iteration loop to track physical error
    for iteration in range(max_iter):
        # Perform one Jacobi iteration
        u_old[:] = u
        iterative_residual = kernel_func(u_old, u, f, h, omega)

        # Compute physical error against exact solution
        physical_error = np.sqrt(np.sum((u - u_exact) ** 2)) / N**3

        # Store result
        results.append({
            'kernel': kernel_name,
            'iteration': iteration,
            'physical_error': physical_error,
            'iterative_residual': iterative_residual,
            'N': N,
            'omega': omega,
            'tolerance': tolerance,
        })

        # Check convergence (iterative residual)
        if iterative_residual < tolerance:
            print(f"  Converged at iteration {iteration}")
            print(f"  Iterative residual: {iterative_residual:.4e}")
            print(f"  Physical error: {physical_error:.4e}")
            break
    else:
        print(f"  Did not converge in {max_iter} iterations")
        print(f"  Final iterative residual: {iterative_residual:.4e}")
        print(f"  Final physical error: {physical_error:.4e}")

# Save results
df = pd.DataFrame(results)
data_dir = datatools.get_data_dir()
output_path = data_dir / "kernel_convergence.parquet"
datatools.save_simulation_data(df, output_path, format="parquet")

print("\n" + "=" * 60)
print("Convergence validation completed!")
print(f"Saved to: {output_path}")
print(f"Total data points: {len(df)}")

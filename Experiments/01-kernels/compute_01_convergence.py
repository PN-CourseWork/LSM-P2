"""
Kernel Convergence Validation
==============================

Compare NumPy vs Numba kernel convergence behavior for a fixed problem size.

This validates that both kernels produce identical convergence by tracking
the physical error ||u - u_exact|| against the analytical solution.
"""

import numpy as np
import pandas as pd

from Poisson import problems
from Poisson.kernels import NumPyKernel, NumbaKernel
from utils import datatools

print("Kernel Convergence Comparison")
print("=" * 60)

# Test parameters
problem_sizes = [25, 50, 75]  # Three problem sizes
omega = 0.75
max_iter = 20000
tolerance = 1e-12

# Initialize kernels
numpy_kernel = NumPyKernel()
numba_kernel = NumbaKernel()

kernels = [
    ('numpy', numpy_kernel),
    ('numba', numba_kernel),
]

# Warmup Numba kernel once at the start
print("\nWarming up Numba kernel...")
numba_kernel.warmup(N=10, omega=omega)

results = []

for N in problem_sizes:
    print(f"\n{'='*60}")
    print(f"Problem size N={N}")
    print('='*60)

    # Setup problem
    h = 2.0 / (N - 1)
    u_exact = problems.sinusoidal_exact_solution(N)
    f = problems.sinusoidal_source_term(N)

    for kernel_name, kernel in kernels:
        print(f"\nTesting {kernel_name} kernel...")
        print("-" * 60)

        # Configure kernel
        kernel.configure(h=h, omega=omega)

        # Initialize solution
        u = np.zeros((N, N, N), dtype=np.float64)  # Zero initial guess
        u_old = u.copy()

        print("  Iterating...")

        # Manual iteration loop to track physical error
        for iteration in range(max_iter):
            # Perform one Jacobi iteration
            u_old[:] = u
            iterative_residual = kernel.step(u_old, u, f)

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



"""
Kernel Convergence Validation
==============================

Compare NumPy vs Numba kernel convergence behavior to validate that both
implementations produce identical results when solving the Poisson equation by 
measuring the physical residual :math:`||u - u_{exact}||_2` using a known analytical solution.  

.. math::

    u(x,y,z) = \sin(\pi x) \sin(\pi y) \sin(\pi z)

"""
import numpy as np
import pandas as pd

from Poisson import problems
from Poisson.kernels import NumPyKernel, NumbaKernel
from utils import datatools

# %%
# Test Configuration
# ------------------
#
# We test three problem sizes to verify convergence behavior scales correctly.
# The omega parameter controls the relaxation in Jacobi iteration.

problem_sizes = [25, 50, 75]  # Grid sizes: N×N×N
omega = 0.75                  # Relaxation parameter
max_iter = 20000              # Maximum iterations
tolerance = 1e-12             # Convergence criterion

print("Kernel Convergence Validation")
print("=" * 60)
print(f"Problem sizes: {problem_sizes}")
print(f"Relaxation parameter (ω): {omega}")
print(f"Convergence tolerance: {tolerance:.2e}")
print(f"Maximum iterations: {max_iter}")

# %%
# Initialize Kernels
# ------------------
#
# Create instances of both kernel implementations. The Numba kernel requires
# a warmup phase to compile the JIT functions before accurate timing.

numpy_kernel = NumPyKernel()
numba_kernel = NumbaKernel()

kernels = [
    ('numpy', numpy_kernel),
    ('numba', numba_kernel),
]

print("\nWarming up Numba kernel...")
numba_kernel.warmup(N=10, omega=omega)
print("Warmup complete!")

# %%
# Run Convergence Tests
# ---------------------
#
# For each problem size and kernel, we iterate until convergence and track
# both the iterative residual and physical error at each iteration.

results = []

for N in problem_sizes:
    print(f"\n{'='*60}")
    print(f"Problem size N={N} ({N**3:,} grid points)")
    print('='*60)

    # Setup problem with analytical solution
    h = 2.0 / (N - 1)
    u_exact = problems.sinusoidal_exact_solution(N)
    f = problems.sinusoidal_source_term(N)

    for kernel_name, kernel in kernels:
        print(f"\nTesting {kernel_name} kernel...")
        print("-" * 60)

        # Configure kernel
        kernel.configure(h=h, omega=omega)

        # Initialize solution with zero initial guess
        u = np.zeros((N, N, N), dtype=np.float64)
        u_old = u.copy()

        print("  Iterating...")

        # Iterate until convergence
        for iteration in range(max_iter):
            # Perform one Jacobi iteration
            u_old[:] = u
            iterative_residual = kernel.step(u_old, u, f)

            # Compute physical error against exact solution
            physical_error = np.sqrt(np.sum((u - u_exact) ** 2)) / N**3

            # Store result for this iteration
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
                print(f"    Iterative residual: {iterative_residual:.4e}")
                print(f"    Physical error: {physical_error:.4e}")
                break
        else:
            print(f"  Did not converge in {max_iter} iterations")
            print(f"    Final iterative residual: {iterative_residual:.4e}")
            print(f"    Final physical error: {physical_error:.4e}")

# %%
# Save Results
# ------------
#
# Store the convergence history for both kernels across all problem sizes.
# This data will be used by the plotting script to generate convergence curves.

df = pd.DataFrame(results)
data_dir = datatools.get_data_dir()
output_path = data_dir / "kernel_convergence.parquet"
datatools.save_simulation_data(df, output_path, format="parquet")

print("\n" + "=" * 60)
print("Convergence validation completed!")
print("=" * 60)
print(f"Saved to: {output_path}")
print(f"Total data points: {len(df):,}")
print(f"Kernels tested: {df['kernel'].nunique()}")
print(f"Problem sizes: {sorted(df['N'].unique())}")



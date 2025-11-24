r"""
Kernel Convergence Validation
==============================

Compare NumPy vs Numba kernel convergence behavior to validate that both
implementations produce identical results when solving the Poisson equation by
measuring the physical residual :math:`||u - u_{exact}||_2` using a known analytical solution:

.. math::

    u(x,y,z) = \sin(\pi x) \sin(\pi y) \sin(\pi z)

"""
import numpy as np
import pandas as pd
from pathlib import Path

from Poisson import problems
from Poisson.kernels import NumPyKernel, NumbaKernel

# %%
# Test Configuration
# ------------------
#
# We test three problem sizes. 
# The omega parameter controls the relaxation in Jacobi iteration.

problem_sizes = [25, 50, 75]  # Grid sizes: N×N×N
omega = 0.75                  # Relaxation parameter
max_iter = 20000              # Maximum iterations
tolerance = 1e-12             # Convergence criterion

# %%
# Run Convergence Tests
# ---------------------
#
# For each problem size and kernel, we iterate until convergence and track
# both the iterative residual and physical error at each iteration.

data = {
    'kernel': [],
    'iteration': [],
    'physical_error': [],
    'iterative_residual': [],
    'N': [],
    'omega': [],
    'tolerance': [],
}

for N in problem_sizes:
    print('='*60)
    print(f"Problem size N={N}")
    print('='*60)

    # Setup problem with analytical solution
    u_exact = problems.sinusoidal_exact_solution(N)
    f = problems.sinusoidal_source_term(N)

    # Create kernels for this problem size
    numpy_kernel = NumPyKernel(N=N, omega=omega, tolerance=tolerance, max_iter=max_iter)
    numba_kernel = NumbaKernel(N=N, omega=omega, tolerance=tolerance, max_iter=max_iter, num_threads=4)

    # Warm up Numba on first iteration
    if N == problem_sizes[0]:
        print("Warming up Numba kernel...")
        numba_kernel.warmup()

    kernels = [
        ('numpy', numpy_kernel),
        ('numba', numba_kernel),
    ]

    for kernel_name, kernel in kernels:
        print(f"\nTesting {kernel_name} kernel...")
        print("-" * 60)

        # Initialize arrays
        u = np.zeros((N, N, N), dtype=np.float64)
        u_old = np.zeros((N, N, N), dtype=np.float64)

        print("  Iterating...")

        # Iterate until convergence
        for iteration in range(max_iter):
            # Perform one Jacobi iteration
            iterative_residual = kernel.step(u_old, u, f)

            # Compute physical error against exact solution
            physical_error = np.sqrt(np.sum((u - u_exact) ** 2)) / N**3

            # Store result for this iteration
            data['kernel'].append(kernel_name)
            data['iteration'].append(iteration)
            data['physical_error'].append(physical_error)
            data['iterative_residual'].append(iterative_residual)
            data['N'].append(N)
            data['omega'].append(omega)
            data['tolerance'].append(tolerance)

            # Check convergence (iterative residual)
            if iterative_residual < tolerance:
                print(f"  Converged at iteration {iteration}")
                print(f"    Iterative residual: {iterative_residual:.4e}")
                print(f"    Physical error: {physical_error:.4e}")
                break

            # Swap buffers for next iteration
            u, u_old = u_old, u
        else:
            print(f"  Did not converge in {max_iter} iterations")
            print(f"    Final iterative residual: {iterative_residual:.4e}")
            print(f"    Final physical error: {physical_error:.4e}")

# %%
# Save Results
# ------------

df = pd.DataFrame(data)

# Get the data directory
data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "01-kernels"
data_dir.mkdir(parents=True, exist_ok=True)

output_path = data_dir / "kernel_convergence.parquet"
df.to_parquet(output_path, index=False)
print(f"Saved to: {output_path}")


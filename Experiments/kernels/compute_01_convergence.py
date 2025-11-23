"""
Kernel Convergence Validation
==============================

Compare NumPy vs Numba kernel convergence behavior for a fixed problem size.

This validates that both kernels produce identical iterative convergence,
tracking the residual history to verify numerical equivalence.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from Poisson import JacobiPoisson
from utils import datatools

print("Kernel Convergence Comparison")
print("=" * 60)

# Fixed small problem for convergence validation
N = 50
omega = 0.75
max_iter = 500
tolerance = 1e-6

# Test both kernels
kernels = [
    ('numpy', False),
    ('numba', True),
]

results = []

for kernel_name, use_numba in kernels:
    print(f"\nTesting {kernel_name} kernel...")
    print("-" * 60)

    # Create solver
    solver = JacobiPoisson(
        N=N,
        omega=omega,
        max_iter=max_iter,
        tolerance=tolerance,
        use_numba=use_numba,
    )

    # Warmup for Numba
    if use_numba:
        print("  Warming up Numba JIT...")
        solver.warmup(N=10)

    # Solve
    print("  Solving...")
    solver.solve()
    solver.summary()

    # Extract residual history
    residual_history = solver.timeseries.residual_history

    # Store results
    for iteration, residual in enumerate(residual_history):
        results.append({
            'kernel': kernel_name,
            'iteration': iteration,
            'residual': residual,
            'N': N,
            'omega': omega,
            'tolerance': tolerance,
        })

    print(f"  Iterations: {solver.results.iterations}")
    print(f"  Converged: {solver.results.converged}")
    print(f"  Final error: {solver.results.final_error:.4e}")

# Save results
df = pd.DataFrame(results)
data_dir = datatools.get_data_dir()
output_path = data_dir / "kernel_convergence.parquet"
datatools.save_simulation_data(df, output_path, format="parquet")

print("\n" + "=" * 60)
print("Convergence validation completed!")
print(f"Saved to: {output_path}")
print(f"Total data points: {len(df)}")

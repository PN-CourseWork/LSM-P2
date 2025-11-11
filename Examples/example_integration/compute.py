"""
Numerical Integration Example
==============================

This example demonstrates the use of numerical integration methods
from the numutils package.

We'll integrate several functions and compare the trapezoidal rule
with Simpson's rule.
"""

# %% Imports
from pathlib import Path

import numpy as np
import pandas as pd

from numutils import integrate_simpson, integrate_trapz

# %% Configuration
data_dir = Path("data/example_integration")
data_dir.mkdir(parents=True, exist_ok=True)

# %% Define test functions
print("Testing numerical integration methods...")


def f1(x: float) -> float:
    """Simple polynomial: x^2"""
    return x**2


def f2(x: float) -> float:
    """Trigonometric: sin(x)"""
    return np.sin(x)


def f3(x: float) -> float:
    """Exponential: e^(-x)"""
    return np.exp(-x)


# %% Compute integrals
functions = [
    (r"$x^2$ from 0 to 1", f1, 0, 1, 1 / 3),
    (r"$\sin(x)$ from 0 to $\pi$", f2, 0, np.pi, 2.0),
    (r"$e^{-x}$ from 0 to 2", f3, 0, 2, 1 - np.exp(-2)),
]

# Test different numbers of intervals
N_values = [10, 20, 50, 100, 200, 500]

results = []

for name, func, a, b, exact in functions:
    print(f"\n{name}:")
    print(f"  Exact value: {exact:.10f}")

    for N in N_values:
        trapz_result = integrate_trapz(func, a, b, N)
        simpson_result = integrate_simpson(func, a, b, N)

        trapz_error = abs(trapz_result - exact)
        simpson_error = abs(simpson_result - exact)

        results.append(
            {
                "function": name,
                "N": N,
                "method": "Trapezoidal",
                "result": trapz_result,
                "error": trapz_error,
                "exact": exact,
            }
        )

        results.append(
            {
                "function": name,
                "N": N,
                "method": "Simpson",
                "result": simpson_result,
                "error": simpson_error,
                "exact": exact,
            }
        )

        if N == 100:
            print(f"  N={N}:")
            print(f"    Trapezoidal: {trapz_result:.10f} (error: {trapz_error:.2e})")
            print(
                f"    Simpson:     {simpson_result:.10f} (error: {simpson_error:.2e})"
            )

# %% Save results
df = pd.DataFrame(results)
df.to_parquet(data_dir / "integration_results.parquet", index=False)
print(f"\nSaved results to {data_dir}/integration_results.parquet")

# %%

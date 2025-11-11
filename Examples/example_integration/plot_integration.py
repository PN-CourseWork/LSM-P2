"""
Visualize Integration Results
==============================

Plot convergence of numerical integration methods.
"""

# %% Imports
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %% Setup
setup_plotting()

# Try to find data file (handles both normal execution and sphinx-gallery builds)
data_file = None
for base_path in [Path("."), Path("../.."), Path("../../..")]:
    candidate = base_path / "data/example_integration/integration_results.parquet"
    if candidate.exists():
        data_file = candidate
        break

if data_file is None:
    import sys

    print("Warning: Data file not found. Run compute.py first.")
    sys.exit(0)

fig_dir = Path("docs/source/generated/figures")
for base_path in [Path("."), Path("../.."), Path("../../..")]:
    candidate = base_path / "docs/source/generated/figures"
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        fig_dir = candidate
        break
    except (PermissionError, OSError):
        continue

# %% Load data
df = pd.read_parquet(data_file)

# %% Plot convergence for each function
functions = df["function"].unique()

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for i, func_name in enumerate(functions):
    ax = axes[i]
    func_data = df[df["function"] == func_name]

    for method in ["Trapezoidal", "Simpson"]:
        method_data = func_data[func_data["method"] == method]
        ax.loglog(
            method_data["N"],
            method_data["error"],
            "o-",
            label=method,
            markersize=4,
        )

    ax.set_xlabel("Number of intervals (N)")
    ax.set_ylabel("Absolute error")
    ax.set_title(func_name)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(fig_dir / "integration_convergence.png", dpi=300, bbox_inches="tight")
print(f"Saved figure to {fig_dir}/integration_convergence.png")

# %% Show convergence rates
print("\nConvergence rates (log-log slope):")
for func_name in functions:
    print(f"\n{func_name}:")
    func_data = df[df["function"] == func_name]

    for method in ["Trapezoidal", "Simpson"]:
        method_data = func_data[func_data["method"] == method].sort_values("N")
        N_vals = method_data["N"].values
        errors = method_data["error"].values

        # Compute slope in log-log space (last 3 points)
        if len(N_vals) >= 3:
            log_N = np.log(N_vals[-3:])
            log_err = np.log(errors[-3:])
            slope = (log_err[-1] - log_err[0]) / (log_N[-1] - log_N[0])
            print(f"  {method}: {slope:.2f}")

# %%

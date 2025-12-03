"""Weak scaling plot from existing data.

Uses time per iteration to compare weak scaling efficiency.
"""

import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from Poisson import get_project_root
from utils.mlflow.io import setup_mlflow_tracking, get_mlflow_client
from utils import plotting  # Apply scientific style

# Connect to Databricks
setup_mlflow_tracking("databricks")
client = get_mlflow_client()

# Load all experiments
experiments = client.search_experiments(filter_string="name LIKE '/Shared/LSM-PoissonMPI%'")
all_runs = []
for exp in experiments:
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], max_results=500)
    if not runs.empty:
        all_runs.append(runs)

df = pd.concat(all_runs, ignore_index=True)

# Extract parameters
df["N"] = pd.to_numeric(df.get("params.N"), errors="coerce")
df["n_ranks"] = pd.to_numeric(df.get("params.n_ranks"), errors="coerce")
df["solver"] = df.get("params.solver", "jacobi").fillna("jacobi")
df["strategy"] = df.get("params.strategy", "sliced").fillna("sliced")
df["wall_time"] = pd.to_numeric(df.get("metrics.wall_time"), errors="coerce")
df["iterations"] = pd.to_numeric(df.get("metrics.iterations"), errors="coerce")

# Filter valid
df = df.dropna(subset=["N", "n_ranks", "wall_time", "iterations"])
df = df[df["iterations"] > 0]

# Compute metrics
df["time_per_iter"] = df["wall_time"] / df["iterations"]
df["total_points"] = df["N"] ** 3
df["points_per_rank"] = df["total_points"] / df["n_ranks"]
df["local_N"] = (df["points_per_rank"]) ** (1/3)

# Define weak scaling series (approximate local_N groups)
# Series 1: ~128-129 local_N
series1_pairs = [(129, 1), (257, 8), (513, 64)]
# Series 2: ~256-257 local_N
series2_pairs = [(257, 1), (513, 8)]

def get_series_data(pairs, series_name):
    """Extract data for a weak scaling series."""
    data = []
    for N, ranks in pairs:
        subset = df[(df["N"] == N) & (df["n_ranks"] == ranks)]
        if len(subset) > 0:
            for solver in ["jacobi", "fmg"]:
                for strategy in ["sliced", "cubic"]:
                    s = subset[(subset["solver"] == solver) & (subset["strategy"] == strategy)]
                    if len(s) > 0:
                        data.append({
                            "series": series_name,
                            "N": N,
                            "n_ranks": ranks,
                            "solver": solver.capitalize(),
                            "strategy": strategy.capitalize(),
                            "time_per_iter": s["time_per_iter"].mean(),
                            "time_per_iter_std": s["time_per_iter"].std(),
                            "count": len(s),
                        })
    return pd.DataFrame(data)

df1 = get_series_data(series1_pairs, "~129³/rank")
df2 = get_series_data(series2_pairs, "~257³/rank")
df_weak = pd.concat([df1, df2], ignore_index=True)

print(f"Weak scaling data points: {len(df_weak)}")
print(df_weak.to_string())

# Output directory
fig_dir = get_project_root() / "figures" / "scaling"
fig_dir.mkdir(parents=True, exist_ok=True)

# Plot: Time per iteration vs ranks (ideal = flat line)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, solver in enumerate(["Jacobi", "Fmg"]):
    ax = axes[idx]
    data = df_weak[df_weak["solver"] == solver]

    if data.empty:
        ax.set_title(f"{solver}: No data")
        continue

    for series in data["series"].unique():
        for strategy in ["Cubic", "Sliced"]:
            subset = data[(data["series"] == series) & (data["strategy"] == strategy)]
            if subset.empty:
                continue

            subset = subset.sort_values("n_ranks")
            label = f"{series}, {strategy}"
            marker = "o" if strategy == "Cubic" else "s"
            linestyle = "-" if "129" in series else "--"

            ax.errorbar(
                subset["n_ranks"],
                subset["time_per_iter"] * 1000,  # Convert to ms
                yerr=subset["time_per_iter_std"] * 1000 if subset["time_per_iter_std"].notna().any() else None,
                marker=marker, linestyle=linestyle, label=label, capsize=3
            )

    ax.set_xlabel("Number of MPI Ranks")
    ax.set_ylabel("Time per Iteration (ms)")
    ax.set_title(f"Weak Scaling: {solver}")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
output_file = fig_dir / "07_weak_scaling_time_per_iter.pdf"
fig.savefig(output_file, bbox_inches="tight")
print(f"\nSaved: {output_file}")
plt.close()

# Plot 2: Weak scaling efficiency
fig, ax = plt.subplots(figsize=(8, 5))

for solver in ["Jacobi", "Fmg"]:
    data = df_weak[df_weak["solver"] == solver]
    if data.empty:
        continue

    for series in data["series"].unique():
        for strategy in ["Cubic", "Sliced"]:
            subset = data[(data["series"] == series) & (data["strategy"] == strategy)]
            if len(subset) < 2:
                continue

            subset = subset.sort_values("n_ranks")

            # Efficiency = T(1) / T(P) (ideal = 1.0 for weak scaling)
            t1 = subset[subset["n_ranks"] == subset["n_ranks"].min()]["time_per_iter"].values[0]
            efficiency = t1 / subset["time_per_iter"]

            label = f"{solver} {series} ({strategy})"
            marker = "o" if "Jacobi" in solver else "^"
            linestyle = "-" if "Cubic" in strategy else "--"

            ax.plot(subset["n_ranks"], efficiency, marker=marker, linestyle=linestyle, label=label)

ax.axhline(y=1.0, color="k", linestyle=":", alpha=0.5, label="Ideal")
ax.set_xlabel("Number of MPI Ranks")
ax.set_ylabel("Weak Scaling Efficiency T(1)/T(P)")
ax.set_title("Weak Scaling Efficiency")
ax.set_xscale("log")
ax.legend(fontsize=7, loc="lower left")
ax.grid(True, alpha=0.3)

output_file = fig_dir / "08_weak_scaling_efficiency.pdf"
fig.savefig(output_file, bbox_inches="tight")
print(f"Saved: {output_file}")
plt.close()

print("\nDone!")

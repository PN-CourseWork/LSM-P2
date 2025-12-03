"""Plot weak scaling from all available MLflow data."""

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
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], max_results=1000)
    if not runs.empty:
        all_runs.append(runs)

df = pd.concat(all_runs, ignore_index=True)

# Extract parameters
df["N"] = pd.to_numeric(df.get("params.N"), errors="coerce").astype("Int64")
df["n_ranks"] = pd.to_numeric(df.get("params.n_ranks"), errors="coerce").astype("Int64")
df["solver"] = df.get("params.solver", "jacobi").fillna("jacobi")
df["strategy"] = df.get("params.strategy", "sliced").fillna("sliced")
df["mlups"] = pd.to_numeric(df.get("metrics.mlups"), errors="coerce")
df["wall_time"] = pd.to_numeric(df.get("metrics.wall_time"), errors="coerce")

# Filter valid runs
df = df.dropna(subset=["N", "n_ranks", "mlups", "wall_time"])
df["points_per_rank"] = (df["N"] ** 3) / df["n_ranks"]

# Define weak scaling series (~2.1M points/rank = 129³)
# Valid pairs: 129@1, 257@8, 513@64
WEAK_PAIRS = {
    (129, 1): "129³",
    (257, 8): "257³",
    (513, 64): "513³",
}

# Filter to weak scaling data
def is_weak_scaling(row):
    return (int(row["N"]), int(row["n_ranks"])) in WEAK_PAIRS

df_weak = df[df.apply(is_weak_scaling, axis=1)].copy()
df_weak["grid_label"] = df_weak.apply(lambda r: WEAK_PAIRS[(int(r["N"]), int(r["n_ranks"]))], axis=1)
df_weak["Solver"] = df_weak["solver"].str.upper()
df_weak["Strategy"] = df_weak["strategy"].str.capitalize()

print(f"Weak scaling runs found: {len(df_weak)}")
print(df_weak.groupby(["N", "n_ranks", "solver", "strategy"])["mlups"].agg(["mean", "count"]))

# Compute weak scaling efficiency
# E(P) = T(1) / T(P) * 100 for ideal weak scaling (constant time)
results = []
for solver in df_weak["solver"].unique():
    for strategy in df_weak["strategy"].unique():
        subset = df_weak[(df_weak["solver"] == solver) & (df_weak["strategy"] == strategy)]
        if subset.empty:
            continue

        # Get baseline (129@1)
        baseline = subset[(subset["N"] == 129) & (subset["n_ranks"] == 1)]
        if baseline.empty:
            continue
        T_1 = baseline["wall_time"].mean()

        for (N, P), label in WEAK_PAIRS.items():
            point = subset[(subset["N"] == N) & (subset["n_ranks"] == P)]
            if point.empty:
                continue
            T_P = point["wall_time"].mean()
            mlups_mean = point["mlups"].mean()
            mlups_std = point["mlups"].std()
            efficiency = (T_1 / T_P) * 100 if T_P > 0 else np.nan

            results.append({
                "N": N,
                "n_ranks": P,
                "grid_label": label,
                "solver": solver,
                "Solver": solver.upper(),
                "strategy": strategy,
                "Strategy": strategy.capitalize(),
                "wall_time": T_P,
                "T_1": T_1,
                "efficiency": efficiency,
                "mlups": mlups_mean,
                "mlups_std": mlups_std,
            })

df_ws = pd.DataFrame(results)
print("\nWeak scaling efficiency:")
print(df_ws[["Solver", "Strategy", "n_ranks", "efficiency", "mlups"]].to_string())

# Output directory
fig_dir = get_project_root() / "figures" / "scaling"
fig_dir.mkdir(parents=True, exist_ok=True)

# Plot 1: Weak Scaling Wall Time (should be constant for ideal weak scaling)
fig, ax = plt.subplots(figsize=(8, 5))
sns.lineplot(
    data=df_weak,
    x="n_ranks",
    y="wall_time",
    hue="Solver",
    style="Strategy",
    markers=True,
    markersize=10,
    errorbar=("ci", 95),
    ax=ax,
)
ax.set_xscale("log")
ax.set_xticks([1, 8, 64])
ax.set_xticklabels(["1\n(129³)", "8\n(257³)", "64\n(513³)"])
ax.set_xlabel("Number of Ranks (Grid Size)")
ax.set_ylabel("Wall Time (s)")
ax.set_title("Weak Scaling: Wall Time (~129³ points/rank)")
ax.grid(True, alpha=0.3)
ax.legend(title="Config")

output_file = fig_dir / "03_weak_walltime.pdf"
fig.savefig(output_file, bbox_inches="tight")
print(f"\nSaved: {output_file}")
plt.close()

# Plot 2: Weak Scaling Efficiency (only for configs with baseline)
if not df_ws.empty:
    fig, ax = plt.subplots(figsize=(8, 5))

    # Only plot Jacobi (has most complete data)
    df_jacobi = df_ws[df_ws["solver"] == "jacobi"]
    if not df_jacobi.empty:
        sns.lineplot(
            data=df_jacobi,
            x="n_ranks",
            y="efficiency",
            hue="Strategy",
            markers=True,
            markersize=10,
            ax=ax,
        )

        # Ideal line at 100%
        ax.axhline(y=100, color="k", linestyle="--", alpha=0.5, label="Ideal")

        ax.set_xscale("log")
        ax.set_xticks([1, 8, 64])
        ax.set_xticklabels(["1\n(129³)", "8\n(257³)", "64\n(513³)"])
        ax.set_xlabel("Number of Ranks (Grid Size)")
        ax.set_ylabel("Efficiency (%)")
        ax.set_title("Weak Scaling Efficiency: Jacobi (~129³ points/rank)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        output_file = fig_dir / "04_weak_efficiency.pdf"
        fig.savefig(output_file, bbox_inches="tight")
        print(f"Saved: {output_file}")
    plt.close()

print("\nDone!")

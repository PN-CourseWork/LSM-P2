"""Strong scaling comparison: Jacobi vs FMG vs FMG Hybrid.

X-axis: Total parallel processes (ranks × threads for hybrid)
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
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], max_results=1000)
    if not runs.empty:
        all_runs.append(runs)

df = pd.concat(all_runs, ignore_index=True)

# Extract parameters
df["N"] = pd.to_numeric(df.get("params.N"), errors="coerce").astype("Int64")
df["n_ranks"] = pd.to_numeric(df.get("params.n_ranks"), errors="coerce").astype("Int64")
df["solver"] = df.get("params.solver", "jacobi").fillna("jacobi")
df["strategy"] = df.get("params.strategy", "sliced").fillna("sliced")
df["numba_threads"] = pd.to_numeric(df.get("params.specified_numba_threads"), errors="coerce").fillna(1).astype(int)
df["wall_time"] = pd.to_numeric(df.get("metrics.wall_time"), errors="coerce")
df["mlups"] = pd.to_numeric(df.get("metrics.mlups"), errors="coerce")

# Filter valid runs
df = df.dropna(subset=["N", "n_ranks", "wall_time", "mlups"])

# Compute total parallel processes
df["n_procs"] = df["n_ranks"] * df["numba_threads"]

# Create solver mode labels
def get_mode(row):
    if row["solver"] == "jacobi":
        return "Jacobi"
    elif row["solver"] == "fmg" and row["numba_threads"] > 1:
        return f"FMG Hybrid ({row['numba_threads']}T)"
    else:
        return "FMG"

df["Mode"] = df.apply(get_mode, axis=1)
df["Strategy"] = df["strategy"].str.capitalize()

# Filter to strong scaling sizes (257 and 513)
df_strong = df[df["N"].isin([257, 513])].copy()

print(f"Total strong scaling runs: {len(df_strong)}")
print("\nBreakdown by Mode:")
print(df_strong.groupby(["Mode", "N"])["n_procs"].agg(["min", "max", "count"]))

# Output directory
fig_dir = get_project_root() / "figures" / "scaling"
fig_dir.mkdir(parents=True, exist_ok=True)

# Compute strong scaling metrics
def compute_speedup(group):
    """Compute speedup relative to single-process baseline."""
    results = []

    # Get baseline (n_procs=1) - use any mode's baseline
    baseline = group[group["n_procs"] == 1]
    if baseline.empty:
        # Try to find smallest n_procs as baseline
        min_procs = group["n_procs"].min()
        baseline = group[group["n_procs"] == min_procs]

    if baseline.empty:
        return pd.DataFrame()

    T_1 = baseline["wall_time"].mean()

    for _, row in group.iterrows():
        P = row["n_procs"]
        T_P = row["wall_time"]
        speedup = T_1 / T_P if T_P > 0 else np.nan

        results.append({
            "N": row["N"],
            "n_procs": P,
            "n_ranks": row["n_ranks"],
            "numba_threads": row["numba_threads"],
            "Mode": row["Mode"],
            "Strategy": row["Strategy"],
            "wall_time": T_P,
            "speedup": speedup,
            "efficiency": (speedup / P) * 100 if P > 0 else np.nan,
            "mlups": row["mlups"],
        })

    return pd.DataFrame(results)

# Compute per (N, Strategy) group
df_ss = pd.concat([
    compute_speedup(g) for _, g in df_strong.groupby(["N", "Strategy"])
], ignore_index=True)

print(f"\nStrong scaling points: {len(df_ss)}")

# Plot: Speedup comparison (col=N, hue=Mode, style=Strategy)
for N in [257, 513]:
    data = df_ss[(df_ss["N"] == N) & (df_ss["n_procs"] > 1)]
    if data.empty:
        continue

    fig, ax = plt.subplots(figsize=(8, 5))

    P_range = sorted(data["n_procs"].unique())

    # Plot each mode
    for mode in ["Jacobi", "FMG", "FMG Hybrid (4T)"]:
        mode_data = data[data["Mode"] == mode]
        if mode_data.empty:
            continue

        for strategy in ["Sliced", "Cubic"]:
            subset = mode_data[mode_data["Strategy"] == strategy].sort_values("n_procs")
            if subset.empty:
                continue

            # Aggregate by n_procs (mean of duplicates)
            subset = subset.groupby("n_procs").agg({"speedup": "mean"}).reset_index()

            label = f"{mode} ({strategy})"
            linestyle = "-" if strategy == "Cubic" else "--"
            marker = "o" if "Jacobi" in mode else ("s" if "Hybrid" in mode else "^")

            ax.plot(subset["n_procs"], subset["speedup"],
                   marker=marker, linestyle=linestyle, label=label, markersize=6)

    # Ideal scaling line
    ax.plot(P_range, P_range, "k--", alpha=0.4, linewidth=1, label="Ideal")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(P_range)
    ax.set_xticks([], minor=True)
    ax.set_xticklabels([str(p) for p in P_range], fontsize=8, rotation=45)
    ax.set_yticks(ax.get_yticks())
    ax.set_yticks([], minor=True)
    ax.set_xlabel("Parallel Processes (ranks × threads)")
    ax.set_ylabel("Speedup S(P)")
    ax.set_title(f"Strong Scaling Comparison: N={N}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    output_file = fig_dir / f"05_strong_comparison_N{N}.pdf"
    fig.savefig(output_file, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()

# Plot: Throughput comparison
for N in [257, 513]:
    data = df_ss[df_ss["N"] == N]
    if data.empty:
        continue

    fig, ax = plt.subplots(figsize=(8, 5))

    P_range = sorted(data["n_procs"].unique())

    for mode in ["Jacobi", "FMG", "FMG Hybrid (4T)"]:
        mode_data = data[data["Mode"] == mode]
        if mode_data.empty:
            continue

        for strategy in ["Sliced", "Cubic"]:
            subset = mode_data[mode_data["Strategy"] == strategy].sort_values("n_procs")
            if subset.empty:
                continue

            # Aggregate by n_procs (mean of duplicates)
            subset = subset.groupby("n_procs").agg({"mlups": "mean"}).reset_index()

            label = f"{mode} ({strategy})"
            linestyle = "-" if strategy == "Cubic" else "--"
            marker = "o" if "Jacobi" in mode else ("s" if "Hybrid" in mode else "^")

            ax.plot(subset["n_procs"], subset["mlups"],
                   marker=marker, linestyle=linestyle, label=label, markersize=6)

    ax.set_xscale("log")
    ax.set_xticks(P_range)
    ax.set_xticks([], minor=True)
    ax.set_xticklabels([str(p) for p in P_range], fontsize=8, rotation=45)
    ax.set_xlabel("Parallel Processes (ranks × threads)")
    ax.set_ylabel("Throughput (MLup/s)")
    ax.set_title(f"Strong Scaling Throughput: N={N}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    output_file = fig_dir / f"06_throughput_comparison_N{N}.pdf"
    fig.savefig(output_file, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()

print("\nDone!")

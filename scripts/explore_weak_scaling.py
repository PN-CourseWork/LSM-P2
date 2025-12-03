"""Explore all MLflow experiments to find weak scaling data."""

import mlflow
import pandas as pd
from utils.mlflow.io import setup_mlflow_tracking, get_mlflow_client

# Connect to Databricks
setup_mlflow_tracking("databricks")
client = get_mlflow_client()

# List all experiments in our project
print("=== All Experiments ===")
experiments = client.search_experiments(filter_string="name LIKE '/Shared/LSM-PoissonMPI%'")
for exp in experiments:
    print(f"  {exp.name} (id={exp.experiment_id})")

# Collect all runs from all experiments
print("\n=== Loading all runs ===")
all_runs = []
for exp in experiments:
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="",
        max_results=1000,
    )
    if not runs.empty:
        runs["experiment_name"] = exp.name
        all_runs.append(runs)
        print(f"  {exp.name}: {len(runs)} runs")

if not all_runs:
    print("No runs found!")
    exit()

df = pd.concat(all_runs, ignore_index=True)
print(f"\nTotal runs: {len(df)}")

# Extract key parameters
df["N"] = pd.to_numeric(df.get("params.N"), errors="coerce").astype("Int64")
df["n_ranks"] = pd.to_numeric(df.get("params.n_ranks"), errors="coerce").astype("Int64")
df["solver"] = df.get("params.solver", "jacobi").fillna("jacobi")
df["strategy"] = df.get("params.strategy", "sliced").fillna("sliced")
df["mlups"] = pd.to_numeric(df.get("metrics.mlups"), errors="coerce")
df["wall_time"] = pd.to_numeric(df.get("metrics.wall_time"), errors="coerce")

# Filter to valid runs
df = df.dropna(subset=["N", "n_ranks", "mlups"])
print(f"Valid runs with metrics: {len(df)}")

# Show all unique (N, n_ranks) combinations
print("\n=== All (N, n_ranks) combinations ===")
combos = df.groupby(["N", "n_ranks"]).agg({
    "solver": lambda x: list(x.unique()),
    "strategy": lambda x: list(x.unique()),
    "mlups": "count"
}).rename(columns={"mlups": "count"})
print(combos.to_string())

# Weak scaling detection: constant work per rank
# Series 1: 129@1, 257@8, 385@27, 513@64 (~129³/rank)
# Series 2: 257@1, 513@8, 769@27, 1025@64 (~257³/rank)
# Calculate points per rank for each combo
df["points_per_rank"] = (df["N"] ** 3) / df["n_ranks"]

print("\n=== Potential weak scaling series (grouped by ~points/rank) ===")
# Round to nearest million for grouping
df["ppr_millions"] = (df["points_per_rank"] / 1e6).round(0)
for ppr, group in df.groupby("ppr_millions"):
    if len(group["n_ranks"].unique()) >= 2:  # At least 2 different rank counts
        ranks = sorted(group["n_ranks"].unique())
        Ns = sorted(group["N"].unique())
        print(f"\n~{ppr}M points/rank:")
        print(f"  Ranks: {list(ranks)}")
        print(f"  N values: {list(Ns)}")
        print(f"  Runs: {len(group)}")

        # Show detailed breakdown
        detail = group.groupby(["N", "n_ranks", "solver", "strategy"])["mlups"].mean()
        print(detail.to_string())

# Try to construct weak scaling plot data
print("\n=== Attempting weak scaling plot ===")
# Find series with baseline (n_ranks=1)
baseline_Ns = df[df["n_ranks"] == 1]["N"].unique()
print(f"N values with baseline (n_ranks=1): {list(baseline_Ns)}")

# For each baseline N, find scaling series
for base_N in sorted(baseline_Ns):
    base_ppr = base_N ** 3  # points per rank at baseline
    # Find runs with similar points per rank (within 20%)
    mask = (df["points_per_rank"] > base_ppr * 0.8) & (df["points_per_rank"] < base_ppr * 1.2)
    series = df[mask].copy()
    if len(series["n_ranks"].unique()) >= 2:
        print(f"\nWeak scaling series starting at N={base_N} (~{base_ppr/1e6:.1f}M pts/rank):")
        summary = series.groupby(["N", "n_ranks"]).agg({
            "mlups": ["mean", "count"],
            "solver": lambda x: ",".join(sorted(set(x)))
        })
        print(summary.to_string())

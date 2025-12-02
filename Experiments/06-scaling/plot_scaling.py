"""
Scaling Experiment Visualization
================================
Strong and weak scaling analysis for Jacobi and FMG solvers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from Poisson import get_project_root

# Setup
sns.set_theme(style="whitegrid")
repo_root = get_project_root()
data_base = repo_root / "data"
fig_dir = repo_root / "figures" / "scaling"
fig_dir.mkdir(parents=True, exist_ok=True)

# Data directories
JACOBI_DIRS = [
    data_base / "06-scaling-strong_sliced",
    data_base / "06-scaling-strong_cubic",
    data_base / "06-scaling-weak_sliced",
    data_base / "06-scaling-weak_cubic",
]
FMG_DIRS = [
    data_base / "06-scaling-fmg_strong_sliced",
    data_base / "06-scaling-fmg_strong_cubic",
    data_base / "06-scaling-fmg_weak_sliced",
    data_base / "06-scaling-fmg_weak_cubic",
]


def load_data(data_dirs: list[Path], solver: str) -> pd.DataFrame:
    """Load HDF5 scaling results into DataFrame."""
    results = []
    for data_dir in data_dirs:
        if not data_dir.exists():
            continue
        for h5_file in data_dir.glob("*.h5"):
            try:
                df = pd.read_hdf(h5_file, key="results")
                df["solver"] = solver
                results.append(df)
            except Exception as e:
                print(f"Warning: Could not load {h5_file}: {e}")
    if not results:
        return pd.DataFrame()
    df = pd.concat(results, ignore_index=True)
    df["method"] = df["solver"] + "-" + df["decomposition"] + "/" + df["communicator"]
    return df


def compute_strong_scaling(df: pd.DataFrame) -> pd.DataFrame:
    """Compute strong scaling: fixed N, varying P."""
    df = df[df["wall_time"].apply(lambda x: x not in (None, 'None', ''))].copy()
    df["wall_time"] = df["wall_time"].astype(float)

    results = []
    for solver in df["solver"].unique():
        for N in df[df["solver"] == solver]["N"].unique():
            df_N = df[(df["solver"] == solver) & (df["N"] == N)]
            baseline = df_N[df_N["mpi_size"] == 1]
            if baseline.empty:
                baseline = df_N[df_N["mpi_size"] == df_N["mpi_size"].min()]
            if baseline.empty:
                continue
            T1 = float(baseline["wall_time"].values[0])

            for _, row in df_N.iterrows():
                P = int(row["mpi_size"])
                T_P = float(row["wall_time"])
                speedup = T1 / T_P if T_P > 0 else 0
                results.append({
                    "solver": solver, "N": N, "P": P,
                    "method": row["method"], "wall_time": T_P,
                    "speedup": speedup,
                    "efficiency": (speedup / P) * 100,
                })
    return pd.DataFrame(results)


def compute_weak_scaling(df: pd.DataFrame) -> pd.DataFrame:
    """Compute weak scaling: N grows with P."""
    df = df[df["wall_time"].apply(lambda x: x not in (None, 'None', ''))].copy()
    df["wall_time"] = df["wall_time"].astype(float)

    results = []
    for solver in df["solver"].unique():
        for method in df[df["solver"] == solver]["method"].unique():
            df_m = df[(df["solver"] == solver) & (df["method"] == method)]
            baseline = df_m[df_m["mpi_size"] == 1]
            if baseline.empty:
                continue
            T1 = float(baseline["wall_time"].values[0])

            for _, row in df_m.iterrows():
                P = int(row["mpi_size"])
                T_P = float(row["wall_time"])
                results.append({
                    "solver": solver, "N": int(row["N"]), "P": P,
                    "method": method, "wall_time": T_P,
                    "efficiency": (T1 / T_P) * 100,
                })
    return pd.DataFrame(results)


def plot_strong_scaling(df: pd.DataFrame, title: str, filename: str):
    """Plot strong scaling with seaborn."""
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    P_range = sorted(df["P"].unique())

    # Left: Speedup
    sns.lineplot(data=df, x="P", y="speedup", hue="method", style="method",
                 markers=True, dashes=False, ax=axes[0], markersize=8)
    axes[0].plot(P_range, P_range, "k--", alpha=0.5, label="Ideal (S=P)")
    axes[0].set(xlabel="Number of Ranks (P)", ylabel="Speedup S(P) = T(1)/T(P)",
                xscale="log", yscale="log", title=f"{title} - Speedup")
    # X-axis: show P values
    axes[0].set_xticks(P_range)
    axes[0].set_xticklabels([str(int(p)) for p in P_range])
    # Y-axis: show integer speedup values from 1 to max
    max_s = int(np.ceil(df["speedup"].max()))
    y_ticks = [1, 2, 4, 8, 16, 32, 64, 128]
    y_ticks = [y for y in y_ticks if y <= max_s * 1.2]
    axes[0].set_yticks(y_ticks)
    axes[0].set_yticklabels([str(y) for y in y_ticks])
    axes[0].legend(loc="upper left", fontsize=8)

    # Right: Efficiency
    sns.lineplot(data=df, x="P", y="efficiency", hue="method", style="method",
                 markers=True, dashes=False, ax=axes[1], markersize=8)
    axes[1].axhline(y=100, color="k", linestyle="--", alpha=0.5, label="Ideal")
    axes[1].set(xlabel="Number of Ranks (P)", ylabel="Parallel Efficiency (%)",
                xscale="log", ylim=(0, 110), title=f"{title} - Efficiency")
    axes[1].set_xticks(P_range)
    axes[1].set_xticklabels([str(int(p)) for p in P_range])
    axes[1].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    fig.savefig(fig_dir / filename)
    print(f"Saved: {fig_dir / filename}")
    plt.close()


def plot_weak_scaling(df: pd.DataFrame, title: str, filename: str):
    """Plot weak scaling with seaborn."""
    if df.empty or df["N"].nunique() <= 1:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    P_range = sorted(df["P"].unique())

    sns.lineplot(data=df, x="P", y="efficiency", hue="method", style="method",
                 markers=True, dashes=False, ax=ax, markersize=8)
    ax.axhline(y=100, color="k", linestyle="--", alpha=0.5, label="Ideal")
    ax.set(xlabel="Number of Ranks (P)", ylabel="Weak Scaling Efficiency (%)",
           xscale="log", ylim=(0, 110), title=title)
    ax.set_xticks(P_range)
    ax.set_xticklabels([str(int(p)) for p in P_range])
    ax.legend(loc="best", fontsize=8)

    plt.tight_layout()
    fig.savefig(fig_dir / filename)
    print(f"Saved: {fig_dir / filename}")
    plt.close()


# Load and process data
df_jacobi = load_data(JACOBI_DIRS, "Jacobi")
df_fmg = load_data(FMG_DIRS, "FMG")
df_all = pd.concat([df_jacobi, df_fmg], ignore_index=True)

print(f"Loaded {len(df_jacobi)} Jacobi, {len(df_fmg)} FMG results")

if df_all.empty:
    print("No data found.")
    import sys
    sys.exit(0)

# Compute scaling metrics
df_strong = compute_strong_scaling(df_all)
df_weak = compute_weak_scaling(df_all)

# Select N with most P values for strong scaling
N_strong = df_strong.groupby("N")["P"].nunique().idxmax()
print(f"Using N={N_strong} for strong scaling")

# Generate plots
df_jac_strong = df_strong[(df_strong["solver"] == "Jacobi") & (df_strong["N"] == N_strong)]
df_all_strong = df_strong[df_strong["N"] == N_strong]
df_jac_weak = df_weak[df_weak["solver"] == "Jacobi"]

plot_strong_scaling(df_jac_strong, f"Strong Scaling: Jacobi (N={N_strong})",
                    "01_strong_scaling_jacobi.pdf")
plot_strong_scaling(df_all_strong, f"Strong Scaling: Jacobi vs FMG (N={N_strong})",
                    "02_strong_scaling_comparison.pdf")
plot_weak_scaling(df_jac_weak, "Weak Scaling: Jacobi", "03_weak_scaling_jacobi.pdf")
plot_weak_scaling(df_weak, "Weak Scaling: Jacobi vs FMG", "04_weak_scaling_comparison.pdf")

# Summary
print("\n" + "=" * 50)
print("SCALING SUMMARY")
print("=" * 50)
for solver in ["Jacobi", "FMG"]:
    df_s = df_strong[(df_strong["solver"] == solver) & (df_strong["N"] == N_strong)]
    if not df_s.empty:
        print(f"\n{solver} (N={N_strong}):")
        print(df_s.groupby("method")[["P", "speedup", "efficiency"]].max().round(2).to_string())

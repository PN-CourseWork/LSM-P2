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

import hydra
from omegaconf import DictConfig

from Poisson import get_project_root
from utils import plotting  # Apply scientific style


def load_scaling_data(dirs_by_solver: dict) -> pd.DataFrame:
    """Load scaling data from directories, properly handling decomposition."""
    results = []
    for solver, dirs in dirs_by_solver.items():
        for data_dir in dirs:
            if not data_dir.exists():
                continue
            # Infer decomposition from directory name
            dir_decomp = "sliced" if "sliced" in data_dir.name else "cubic"
            for h5_file in data_dir.glob("*.h5"):
                try:
                    df = pd.read_hdf(h5_file, key="results")
                    df["solver"] = solver
                    # Fix "none" decomposition for P=1 runs
                    df.loc[df["decomposition"] == "none", "decomposition"] = dir_decomp
                    results.append(df)
                except Exception as e:
                    print(f"Warning: {h5_file}: {e}")
    if not results:
        return pd.DataFrame()
    df = pd.concat(results, ignore_index=True)
    df["method"] = df["solver"] + "-" + df["decomposition"] + "/" + df["communicator"]
    return df


def compute_strong_scaling(df: pd.DataFrame) -> pd.DataFrame:
    """Compute strong scaling: fixed N, varying P.

    Speedup S(P) = T(1) / T(P)
    Efficiency E(P) = S(P) / P * 100
    """
    df = df.copy()
    df["wall_time"] = pd.to_numeric(df["wall_time"], errors="coerce")
    df = df.dropna(subset=["wall_time"])

    results = []
    for solver in df["solver"].unique():
        for N in df[df["solver"] == solver]["N"].unique():
            df_N = df[(df["solver"] == solver) & (df["N"] == N)]

            # Baseline: P=1 (average if multiple)
            baseline = df_N[df_N["mpi_size"] == 1]["wall_time"].mean()
            if pd.isna(baseline):
                continue

            for _, row in df_N.iterrows():
                P = int(row["mpi_size"])
                T_P = float(row["wall_time"])
                speedup = baseline / T_P
                results.append({
                    "solver": solver, "N": N, "P": P,
                    "method": row["method"],
                    "speedup": speedup,
                    "efficiency": (speedup / P) * 100,
                })
    return pd.DataFrame(results)


def compute_weak_scaling(df: pd.DataFrame) -> pd.DataFrame:
    """Compute weak scaling: N grows with P.

    Efficiency E(P) = T(1) / T(P) * 100
    (Should stay ~100% for perfect weak scaling)

    Baseline matching: same solver + decomposition (communicator may differ at P=1).
    """
    df = df.copy()
    df["wall_time"] = pd.to_numeric(df["wall_time"], errors="coerce")
    df = df.dropna(subset=["wall_time"])

    results = []
    for solver in df["solver"].unique():
        for decomp in df[df["solver"] == solver]["decomposition"].unique():
            df_d = df[(df["solver"] == solver) & (df["decomposition"] == decomp)]

            # Baseline: P=1 (any communicator, use mean if multiple)
            baseline = df_d[df_d["mpi_size"] == 1]["wall_time"].mean()
            if pd.isna(baseline):
                continue

            for _, row in df_d.iterrows():
                P = int(row["mpi_size"])
                T_P = float(row["wall_time"])
                results.append({
                    "solver": solver, "N": int(row["N"]), "P": P,
                    "method": row["method"],
                    "efficiency": (baseline / T_P) * 100,
                })
    return pd.DataFrame(results)


def plot_strong_scaling(df: pd.DataFrame, title: str, filename: str, fig_dir: Path):
    """Plot strong scaling (speedup + efficiency) with shared legend."""
    df = df[df["P"] > 1].copy()  # Exclude baseline from plot
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    P_range = sorted(df["P"].unique())

    # Left: Speedup
    sns.lineplot(data=df, x="P", y="speedup", hue="method", style="method",
                 markers=True, dashes=False, ax=axes[0], markersize=8)
    axes[0].plot(P_range, P_range, "k--", alpha=0.5, label="Ideal")
    axes[0].set(xlabel="Number of Ranks (P)", ylabel="Speedup S(P) = T(1)/T(P)",
                xscale="log", yscale="log", title="Speedup")
    axes[0].set_xticks(P_range)
    axes[0].set_xticklabels([str(p) for p in P_range])
    max_s = max(int(np.ceil(df["speedup"].max())), max(P_range))
    y_ticks = [y for y in [1, 2, 4, 8, 16, 32, 64, 128] if y <= max_s * 1.2]
    axes[0].set_yticks(y_ticks)
    axes[0].set_yticklabels([str(y) for y in y_ticks])
    axes[0].get_legend().remove()

    # Right: Efficiency
    sns.lineplot(data=df, x="P", y="efficiency", hue="method", style="method",
                 markers=True, dashes=False, ax=axes[1], markersize=8)
    axes[1].axhline(y=100, color="k", linestyle="--", alpha=0.5)
    axes[1].set(xlabel="Number of Ranks (P)", ylabel="Parallel Efficiency (%)",
                xscale="log", ylim=(0, 110), title="Efficiency")
    axes[1].set_xticks(P_range)
    axes[1].set_xticklabels([str(p) for p in P_range])

    # Shared legend outside
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].get_legend().remove()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.15, 0.5), fontsize=9)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 0.88, 0.95])
    fig.savefig(fig_dir / filename, bbox_inches="tight")
    print(f"Saved: {fig_dir / filename}")
    plt.close()


def plot_weak_scaling(df: pd.DataFrame, title: str, filename: str, fig_dir: Path):
    """Plot weak scaling efficiency with legend outside."""
    df = df[df["P"] > 1].copy()  # Exclude baseline from plot
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    P_range = sorted(df["P"].unique())

    sns.lineplot(data=df, x="P", y="efficiency", hue="method", style="method",
                 markers=True, dashes=False, ax=ax, markersize=8)
    ax.axhline(y=100, color="k", linestyle="--", alpha=0.5)
    ax.set(xlabel="Number of Ranks (P)", ylabel="Weak Scaling Efficiency (%)",
           xscale="log", ylim=(0, 120))
    ax.set_xticks(P_range)
    ax.set_xticklabels([str(p) for p in P_range])

    # Legend outside
    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.2, 0.5), fontsize=9)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 0.8, 0.95])
    fig.savefig(fig_dir / filename, bbox_inches="tight")
    print(f"Saved: {fig_dir / filename}")
    plt.close()


@hydra.main(config_path="../hydra-conf", config_name="experiment/scaling", version_base=None)
def main(cfg: DictConfig):
    """Main plotting function with Hydra configuration."""
    # Setup paths
    repo_root = get_project_root()
    data_base = repo_root / "data"
    fig_dir = repo_root / "figures" / "scaling"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Data directories - SEPARATE strong vs weak
    STRONG_DIRS = {
        "Jacobi": [
            data_base / "06-scaling-strong_sliced",
            data_base / "06-scaling-strong_cubic",
        ],
        "FMG": [
            data_base / "06-scaling-fmg_strong_sliced",
            data_base / "06-scaling-fmg_strong_cubic",
        ],
    }
    WEAK_DIRS = {
        "Jacobi": [
            data_base / "06-scaling-weak_sliced",
            data_base / "06-scaling-weak_cubic",
        ],
        "FMG": [
            data_base / "06-scaling-fmg_weak_sliced",
            data_base / "06-scaling-fmg_weak_cubic",
        ],
    }

    # Load data separately for strong and weak scaling
    df_strong = load_scaling_data(STRONG_DIRS)
    df_weak = load_scaling_data(WEAK_DIRS)

    print(f"Strong scaling: {len(df_strong)} data points")
    print(f"Weak scaling: {len(df_weak)} data points")

    if df_strong.empty and df_weak.empty:
        print("No data found.")
        return

    # Compute metrics
    df_strong_metrics = compute_strong_scaling(df_strong)
    df_weak_metrics = compute_weak_scaling(df_weak)

    # Strong scaling plots
    N_strong = df_strong_metrics["N"].mode().values[0] if not df_strong_metrics.empty else 513
    print(f"Strong scaling N={N_strong}")

    df_jac_strong = df_strong_metrics[(df_strong_metrics["solver"] == "Jacobi") &
                                       (df_strong_metrics["N"] == N_strong)]
    df_all_strong = df_strong_metrics[df_strong_metrics["N"] == N_strong]

    plot_strong_scaling(df_jac_strong, f"Strong Scaling: Jacobi (N={N_strong})",
                        "01_strong_scaling_jacobi.pdf", fig_dir)
    plot_strong_scaling(df_all_strong, f"Strong Scaling: Jacobi vs FMG (N={N_strong})",
                        "02_strong_scaling_comparison.pdf", fig_dir)

    # Weak scaling plots
    df_jac_weak = df_weak_metrics[df_weak_metrics["solver"] == "Jacobi"]
    plot_weak_scaling(df_jac_weak, "Weak Scaling: Jacobi", "03_weak_scaling_jacobi.pdf", fig_dir)
    plot_weak_scaling(df_weak_metrics, "Weak Scaling: Jacobi vs FMG", "04_weak_scaling_comparison.pdf", fig_dir)

    # Summary
    print("\n" + "=" * 50)
    print("SCALING SUMMARY")
    print("=" * 50)
    if not df_strong_metrics.empty:
        print(f"\nStrong Scaling (N={N_strong}):")
        summary = df_strong_metrics[df_strong_metrics["N"] == N_strong].groupby("method").agg({
            "P": "max", "speedup": "max", "efficiency": lambda x: x.iloc[-1]
        }).round(2)
        print(summary.to_string())

    if not df_weak_metrics.empty:
        print(f"\nWeak Scaling:")
        summary = df_weak_metrics.groupby("method").agg({
            "P": "max", "efficiency": lambda x: x.iloc[-1]
        }).round(2)
        print(summary.to_string())


if __name__ == "__main__":
    main()

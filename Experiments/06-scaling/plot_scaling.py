"""
Scaling Experiment Visualization
================================
Strong and weak scaling analysis for Jacobi and FMG solvers.

Plots:
1. Decomposition comparison (sliced vs cubic) - Strong & Weak scaling
2. FMG Hybrid (MPI + Numba threads) - Strong & Weak scaling

Fetches data from MLflow experiments.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import hydra
from omegaconf import DictConfig

from Poisson import get_project_root
from utils.mlflow.io import setup_mlflow_tracking, load_runs
from utils import plotting  # Apply scientific style


def load_experiment_data(experiment_name: str, project_prefix: str) -> pd.DataFrame:
    """Load experiment data from MLflow."""
    df = load_runs(experiment_name, converged_only=False, project_prefix=project_prefix)

    if df.empty:
        return df

    # Extract parameters (handle missing columns gracefully)
    df["N"] = pd.to_numeric(df.get("params.N"), errors="coerce").astype("Int64")
    df["n_ranks"] = pd.to_numeric(df.get("params.n_ranks"), errors="coerce").astype("Int64")
    df["solver"] = df.get("params.solver", pd.Series("jacobi", index=df.index)).fillna("jacobi")
    df["strategy"] = df.get("params.strategy", pd.Series("sliced", index=df.index)).fillna("sliced")
    df["communicator"] = df.get("params.communicator", pd.Series("custom", index=df.index)).fillna("custom")
    df["numba_threads"] = pd.to_numeric(df.get("params.numba_threads"), errors="coerce").fillna(1).astype(int)

    # Extract metrics (handle missing columns gracefully)
    df["wall_time"] = pd.to_numeric(df.get("metrics.wall_time"), errors="coerce")
    df["mlups"] = pd.to_numeric(df.get("metrics.mlups"), errors="coerce")
    df["iterations"] = pd.to_numeric(df.get("metrics.iterations"), errors="coerce")

    # Computed columns
    df["total_processes"] = df["n_ranks"] * df["numba_threads"]
    df["Strategy"] = df["strategy"].str.title()  # For legend

    # Keep latest run per configuration
    df = df.sort_values("start_time").groupby(
        ["solver", "N", "n_ranks", "strategy", "communicator", "numba_threads"]
    ).last().reset_index()

    return df


def compute_strong_scaling(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """Compute strong scaling metrics.

    Speedup S(P) = T(1) / T(P)
    Efficiency E(P) = S(P) / P * 100
    """
    df = df.dropna(subset=["wall_time"]).copy()
    results = []

    for keys, group in df.groupby(group_cols):
        # Baseline: smallest rank count (usually 1)
        min_ranks = group["n_ranks"].min()
        baseline_rows = group[group["n_ranks"] == min_ranks]
        if baseline_rows.empty:
            continue
        baseline = baseline_rows["wall_time"].mean()

        for _, row in group.iterrows():
            P = int(row["n_ranks"])
            T_P = float(row["wall_time"])
            speedup = baseline / T_P
            results.append({
                **{col: row[col] for col in group_cols},
                "n_ranks": P,
                "total_processes": int(row["total_processes"]),
                "wall_time": T_P,
                "speedup": speedup,
                "efficiency": (speedup / P) * 100,
                "mlups": row.get("mlups", np.nan),
            })

    return pd.DataFrame(results)


def compute_weak_scaling(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """Compute weak scaling metrics.

    Efficiency E(P) = T(1) / T(P) * 100
    """
    df = df.dropna(subset=["wall_time"]).copy()
    results = []

    for keys, group in df.groupby(group_cols):
        # Baseline: P=1
        baseline_rows = group[group["n_ranks"] == 1]
        if baseline_rows.empty:
            continue
        baseline = baseline_rows["wall_time"].mean()

        for _, row in group.iterrows():
            P = int(row["n_ranks"])
            T_P = float(row["wall_time"])
            results.append({
                **{col: row[col] for col in group_cols},
                "N": int(row["N"]),
                "n_ranks": P,
                "total_processes": int(row["total_processes"]),
                "wall_time": T_P,
                "efficiency": (baseline / T_P) * 100,
                "mlups": row.get("mlups", np.nan),
            })

    return pd.DataFrame(results)


def plot_decomposition_scaling(df_strong: pd.DataFrame, df_weak: pd.DataFrame, fig_dir):
    """Plot decomposition comparison (sliced vs cubic).

    Strong scaling: Speedup & Efficiency vs Ranks, hue=Strategy
    Weak scaling: Efficiency vs Ranks, hue=Strategy
    """
    # Strong scaling
    if not df_strong.empty:
        df = df_strong[df_strong["n_ranks"] > 1].copy()
        if not df.empty:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            P_range = sorted(df["n_ranks"].unique())

            # Speedup
            sns.lineplot(data=df, x="n_ranks", y="speedup", hue="Strategy",
                         style="Strategy", markers=True, dashes=False,
                         ax=axes[0], markersize=8)
            axes[0].plot(P_range, P_range, "k--", alpha=0.5, label="Ideal", linewidth=1)
            axes[0].set(xlabel="Number of Ranks", ylabel="Speedup S(P)",
                        xscale="log", yscale="log")
            axes[0].set_xticks(P_range)
            axes[0].set_xticklabels([str(p) for p in P_range])
            axes[0].legend(title="Decomposition")
            axes[0].grid(True, alpha=0.3)

            # Efficiency
            sns.lineplot(data=df, x="n_ranks", y="efficiency", hue="Strategy",
                         style="Strategy", markers=True, dashes=False,
                         ax=axes[1], markersize=8)
            axes[1].axhline(y=100, color="k", linestyle="--", alpha=0.5, linewidth=1)
            axes[1].set(xlabel="Number of Ranks", ylabel="Parallel Efficiency (%)",
                        xscale="log", ylim=(0, 110))
            axes[1].set_xticks(P_range)
            axes[1].set_xticklabels([str(p) for p in P_range])
            axes[1].legend(title="Decomposition")
            axes[1].grid(True, alpha=0.3)

            N = df["N"].mode().values[0] if "N" in df.columns else "?"
            fig.suptitle(f"Strong Scaling: Decomposition Comparison (N={N})", fontweight="bold")
            plt.tight_layout()
            fig.savefig(fig_dir / "01_strong_scaling_decomposition.pdf", bbox_inches="tight")
            print(f"Saved: {fig_dir / '01_strong_scaling_decomposition.pdf'}")
            plt.close()

    # Weak scaling
    if not df_weak.empty:
        df = df_weak[df_weak["n_ranks"] > 1].copy()
        if not df.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            P_range = sorted(df["n_ranks"].unique())

            sns.lineplot(data=df, x="n_ranks", y="efficiency", hue="Strategy",
                         style="Strategy", markers=True, dashes=False,
                         ax=ax, markersize=8)
            ax.axhline(y=100, color="k", linestyle="--", alpha=0.5, linewidth=1, label="Ideal")
            ax.set(xlabel="Number of Ranks", ylabel="Weak Scaling Efficiency (%)",
                   xscale="log", ylim=(0, 120))
            ax.set_xticks(P_range)
            ax.set_xticklabels([str(p) for p in P_range])
            ax.legend(title="Decomposition")
            ax.grid(True, alpha=0.3)

            fig.suptitle("Weak Scaling: Decomposition Comparison", fontweight="bold")
            plt.tight_layout()
            fig.savefig(fig_dir / "02_weak_scaling_decomposition.pdf", bbox_inches="tight")
            print(f"Saved: {fig_dir / '02_weak_scaling_decomposition.pdf'}")
            plt.close()


def plot_fmg_hybrid_scaling(df_strong: pd.DataFrame, df_weak: pd.DataFrame, fig_dir):
    """Plot FMG hybrid MPI+Numba scaling.

    x-axis: Total Processes (MPI ranks × numba threads)
    Legend: Numba threads configuration
    """
    # Strong scaling
    if not df_strong.empty:
        df = df_strong[df_strong["n_ranks"] > 1].copy()
        df["Config"] = df["numba_threads"].apply(lambda x: f"{x} thread{'s' if x > 1 else ''}")
        if not df.empty:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            P_range = sorted(df["total_processes"].unique())

            # Speedup vs total processes
            sns.lineplot(data=df, x="total_processes", y="speedup", hue="Config",
                         style="Config", markers=True, dashes=False,
                         ax=axes[0], markersize=8)
            axes[0].plot(P_range, P_range, "k--", alpha=0.5, label="Ideal", linewidth=1)
            axes[0].set(xlabel="Total Processes (ranks × threads)", ylabel="Speedup S(P)",
                        xscale="log", yscale="log")
            axes[0].legend(title="Numba Threads")
            axes[0].grid(True, alpha=0.3)

            # Efficiency
            sns.lineplot(data=df, x="total_processes", y="efficiency", hue="Config",
                         style="Config", markers=True, dashes=False,
                         ax=axes[1], markersize=8)
            axes[1].axhline(y=100, color="k", linestyle="--", alpha=0.5, linewidth=1)
            axes[1].set(xlabel="Total Processes (ranks × threads)", ylabel="Parallel Efficiency (%)",
                        xscale="log", ylim=(0, 110))
            axes[1].legend(title="Numba Threads")
            axes[1].grid(True, alpha=0.3)

            fig.suptitle("FMG Strong Scaling: Hybrid MPI + Numba", fontweight="bold")
            plt.tight_layout()
            fig.savefig(fig_dir / "03_strong_scaling_fmg_hybrid.pdf", bbox_inches="tight")
            print(f"Saved: {fig_dir / '03_strong_scaling_fmg_hybrid.pdf'}")
            plt.close()

    # Weak scaling
    if not df_weak.empty:
        df = df_weak[df_weak["n_ranks"] > 1].copy()
        df["Config"] = df["numba_threads"].apply(lambda x: f"{x} thread{'s' if x > 1 else ''}")
        if not df.empty:
            fig, ax = plt.subplots(figsize=(8, 5))

            sns.lineplot(data=df, x="total_processes", y="efficiency", hue="Config",
                         style="Config", markers=True, dashes=False,
                         ax=ax, markersize=8)
            ax.axhline(y=100, color="k", linestyle="--", alpha=0.5, linewidth=1, label="Ideal")
            ax.set(xlabel="Total Processes (ranks × threads)",
                   ylabel="Weak Scaling Efficiency (%)",
                   xscale="log", ylim=(0, 120))
            ax.legend(title="Numba Threads")
            ax.grid(True, alpha=0.3)

            fig.suptitle("FMG Weak Scaling: Hybrid MPI + Numba", fontweight="bold")
            plt.tight_layout()
            fig.savefig(fig_dir / "04_weak_scaling_fmg_hybrid.pdf", bbox_inches="tight")
            print(f"Saved: {fig_dir / '04_weak_scaling_fmg_hybrid.pdf'}")
            plt.close()


def plot_mlups_comparison(df: pd.DataFrame, fig_dir):
    """Plot MLups performance comparison."""
    if df.empty or "mlups" not in df.columns:
        return

    df = df.dropna(subset=["mlups"]).copy()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.lineplot(data=df, x="n_ranks", y="mlups", hue="Strategy",
                 style="Strategy", markers=True, dashes=False,
                 ax=ax, markersize=8)
    ax.set(xlabel="Number of Ranks", ylabel="Performance (Mlup/s)", xscale="log")
    ax.legend(title="Decomposition")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Performance: Decomposition Comparison", fontweight="bold")
    plt.tight_layout()
    fig.savefig(fig_dir / "05_mlups_decomposition.pdf", bbox_inches="tight")
    print(f"Saved: {fig_dir / '05_mlups_decomposition.pdf'}")
    plt.close()


@hydra.main(config_path="../hydra-conf", config_name="experiment/scaling", version_base=None)
def main(cfg: DictConfig):
    """Main plotting function."""
    repo_root = get_project_root()
    fig_dir = repo_root / "figures" / "scaling"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load data from MLflow
    print("Loading data from MLflow...")
    setup_mlflow_tracking(mode=cfg.mlflow.mode)
    prefix = cfg.mlflow.databricks_project_prefix

    # Load from the three scaling experiments
    df_scaling = load_experiment_data("scaling", prefix)
    df_weak = load_experiment_data("weak_scaling", prefix)
    df_fmg = load_experiment_data("fmg_scaling", prefix)

    print(f"\nLoaded:")
    print(f"  Strong scaling (Jacobi): {len(df_scaling)} runs")
    print(f"  Weak scaling (Jacobi): {len(df_weak)} runs")
    print(f"  FMG scaling: {len(df_fmg)} runs")

    if df_scaling.empty and df_weak.empty and df_fmg.empty:
        print("\nNo data found. Run experiments first:")
        print("  ./Experiments/HPC-jobs/submit_all.sh")
        return

    # === Decomposition comparison (Jacobi) ===
    print("\n--- Decomposition Comparison ---")
    if not df_scaling.empty:
        df_strong_decomp = compute_strong_scaling(df_scaling, ["strategy", "N"])
        df_strong_decomp["Strategy"] = df_strong_decomp["strategy"].str.title()
        print(f"Strong scaling points: {len(df_strong_decomp)}")
    else:
        df_strong_decomp = pd.DataFrame()

    if not df_weak.empty:
        df_weak_decomp = compute_weak_scaling(df_weak, ["strategy"])
        df_weak_decomp["Strategy"] = df_weak_decomp["strategy"].str.title()
        print(f"Weak scaling points: {len(df_weak_decomp)}")
    else:
        df_weak_decomp = pd.DataFrame()

    plot_decomposition_scaling(df_strong_decomp, df_weak_decomp, fig_dir)

    # === FMG Hybrid scaling ===
    print("\n--- FMG Hybrid Scaling ---")
    if not df_fmg.empty:
        # Separate strong (single N) and weak (varying N) data
        N_counts = df_fmg.groupby("N").size()
        if len(N_counts) > 0:
            # For strong scaling, use the most common N
            N_strong = N_counts.idxmax()
            df_fmg_strong_data = df_fmg[df_fmg["N"] == N_strong]
            df_fmg_strong = compute_strong_scaling(df_fmg_strong_data, ["numba_threads", "N"])
            print(f"FMG strong scaling points (N={N_strong}): {len(df_fmg_strong)}")

            # For weak scaling, use all varying N data
            df_fmg_weak = compute_weak_scaling(df_fmg, ["numba_threads"])
            print(f"FMG weak scaling points: {len(df_fmg_weak)}")

            plot_fmg_hybrid_scaling(df_fmg_strong, df_fmg_weak, fig_dir)

    # === MLups comparison ===
    if not df_strong_decomp.empty:
        plot_mlups_comparison(df_strong_decomp, fig_dir)

    # === Summary ===
    print("\n" + "=" * 60)
    print("SCALING SUMMARY")
    print("=" * 60)

    if not df_strong_decomp.empty:
        print("\nStrong Scaling (Jacobi) - Max Speedup & Min Efficiency:")
        summary = df_strong_decomp.groupby("Strategy").agg({
            "n_ranks": "max",
            "speedup": "max",
            "efficiency": ["max", "min"]
        }).round(2)
        print(summary.to_string())

    if not df_weak_decomp.empty:
        print("\nWeak Scaling (Jacobi) - Efficiency Range:")
        summary = df_weak_decomp.groupby("Strategy").agg({
            "n_ranks": "max",
            "efficiency": ["max", "min"]
        }).round(2)
        print(summary.to_string())

    print("\nDone!")


if __name__ == "__main__":
    main()

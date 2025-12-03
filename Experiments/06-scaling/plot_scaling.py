"""
Scaling Experiment Visualization
================================

Strong and weak scaling analysis for Jacobi and FMG solvers.

Experiments (from v3):
- scaling: Strong scaling (Jacobi) - N=257,513, ranks=1-96
- fmg_scaling: Strong scaling (FMG) - N=257,513, ranks=1-96
- weak_scaling_jacobi: Weak scaling Jacobi - 129@1, 257@8, 513@64
- weak_scaling_fmg: Weak scaling FMG - 129@1, 257@8, 513@64

Usage:
    uv run python Experiments/06-scaling/plot_scaling.py mlflow=databricks
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


def load_scaling_data(experiment_name: str, prefix: str) -> pd.DataFrame:
    """Load and process scaling experiment data."""
    df = load_runs(experiment_name, converged_only=False, project_prefix=prefix)

    if df.empty:
        return df

    # Extract parameters
    df["N"] = pd.to_numeric(df.get("params.N"), errors="coerce").astype("Int64")
    df["n_ranks"] = pd.to_numeric(df.get("params.n_ranks"), errors="coerce").astype("Int64")
    df["solver"] = df.get("params.solver", "jacobi").fillna("jacobi")
    df["strategy"] = df.get("params.strategy", "sliced").fillna("sliced")

    # Extract metrics
    df["wall_time"] = pd.to_numeric(df.get("metrics.wall_time"), errors="coerce")
    df["mlups"] = pd.to_numeric(df.get("metrics.mlups"), errors="coerce")
    df["iterations"] = pd.to_numeric(df.get("metrics.iterations"), errors="coerce").astype("Int64")

    # Derived metrics
    df["wall_time_per_iter_ms"] = (df["wall_time"] / df["iterations"]) * 1e3
    df["total_points"] = df["N"] ** 3

    # Labels
    df["Solver"] = df["solver"].str.upper()
    df["Strategy"] = df["strategy"].str.capitalize()

    return df


def compute_strong_scaling(df: pd.DataFrame, baseline_col: str = "n_ranks") -> pd.DataFrame:
    """Compute strong scaling metrics (speedup and efficiency).

    Speedup S(P) = T(1) / T(P)
    Efficiency E(P) = S(P) / P * 100

    Note: For n_ranks=1, strategy doesn't matter (no decomposition).
    We use a common baseline per (N, solver) for all strategies.
    """
    results = []

    # Get baselines per (N, solver) - strategy irrelevant for sequential
    baselines = df[df["n_ranks"] == 1].groupby(["N", "solver"])["wall_time"].mean()

    # Process each run
    for _, row in df.iterrows():
        N = int(row["N"])
        solver = row["solver"]
        P = int(row["n_ranks"])
        T_P = float(row["wall_time"])

        # Get baseline for this (N, solver)
        try:
            T_1 = baselines.loc[(N, solver)]
        except KeyError:
            # No baseline for this configuration
            continue

        speedup = T_1 / T_P if T_P > 0 else np.nan

        results.append({
            "N": N,
            "n_ranks": P,
            "strategy": row["strategy"],
            "Strategy": row["strategy"].capitalize() if pd.notna(row["strategy"]) else "Unknown",
            "solver": solver,
            "Solver": solver.upper() if pd.notna(solver) else "JACOBI",
            "wall_time": T_P,
            "T_1": T_1,
            "speedup": speedup,
            "efficiency": (speedup / P) * 100 if P > 0 else np.nan,
            "mlups": row.get("mlups", np.nan),
        })

    return pd.DataFrame(results)


def compute_weak_scaling(df: pd.DataFrame) -> pd.DataFrame:
    """Compute weak scaling efficiency.

    Efficiency E(P) = T(1) / T(P) * 100
    (For weak scaling, work per rank is constant, so ideal T(P) = T(1))
    """
    results = []

    # Group by strategy only (N varies with ranks in weak scaling)
    for strategy, group in df.groupby("strategy"):
        # Get baseline (n_ranks=1)
        baseline_rows = group[group["n_ranks"] == 1]
        if baseline_rows.empty:
            continue
        T_1 = baseline_rows["wall_time"].mean()

        for _, row in group.iterrows():
            P = int(row["n_ranks"])
            T_P = float(row["wall_time"])

            results.append({
                "N": int(row["N"]),
                "n_ranks": P,
                "strategy": strategy,
                "Strategy": strategy.capitalize(),
                "solver": row["solver"],
                "Solver": row["solver"].upper() if pd.notna(row["solver"]) else "JACOBI",
                "wall_time": T_P,
                "efficiency": (T_1 / T_P) * 100 if T_P > 0 else np.nan,
                "mlups": row.get("mlups", np.nan),
            })

    return pd.DataFrame(results)


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Generate scaling plots from MLflow data."""

    repo_root = get_project_root()
    fig_dir = repo_root / "figures" / "scaling"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data from MLflow...")
    setup_mlflow_tracking(mode=cfg.mlflow.mode)
    prefix = cfg.mlflow.get("project_prefix", "")

    # Load all scaling data from the main experiments
    df_jacobi = load_scaling_data("scaling", prefix)
    df_fmg = load_scaling_data("fmg_scaling", prefix)

    print(f"\nLoaded:")
    print(f"  Jacobi runs: {len(df_jacobi)}")
    print(f"  FMG runs: {len(df_fmg)}")

    # Combine all data
    df_all = pd.concat([df_jacobi, df_fmg], ignore_index=True)

    # Weak scaling pairs: (N, n_ranks) with ~127³ points per rank
    # 129@1, 257@8, 513@64
    WEAK_SCALING_PAIRS = {(129, 1), (257, 8), (513, 64)}

    # Separate strong and weak scaling based on (N, n_ranks) pattern
    def is_weak_scaling(row):
        return (int(row["N"]), int(row["n_ranks"])) in WEAK_SCALING_PAIRS

    if not df_all.empty:
        df_all["is_weak"] = df_all.apply(is_weak_scaling, axis=1)
        df_strong = df_all[~df_all["is_weak"]].copy()
        df_weak = df_all[df_all["is_weak"]].copy()
    else:
        df_strong = df_all
        df_weak = pd.DataFrame()

    print(f"  Strong scaling runs: {len(df_strong)}")
    print(f"  Weak scaling runs: {len(df_weak)}")

    if df_strong.empty and df_weak.empty:
        print("\nNo data found. Run experiments first:")
        print("  bsub < jobs/scaling.sh")
        print("  bsub < jobs/baseline.sh")
        return

    # =========================================================================
    # Strong Scaling Plots (N as columns, Solver/Strategy as hue/style)
    # =========================================================================

    if not df_strong.empty:
        print("\n--- Strong Scaling Analysis ---")

        # Compute scaling metrics
        df_ss = compute_strong_scaling(df_strong)
        if df_ss.empty:
            print("No strong scaling data (missing n_ranks=1 baseline)")
        else:
            print(f"Strong scaling points: {len(df_ss)}")

            # Get unique rank counts for x-axis ticks
            P_range = sorted(df_ss["n_ranks"].unique())
            P_range_parallel = [p for p in P_range if p > 1]

            plot_num = 1

            # Speedup plot: N as columns, Solver/Strategy as hue/style
            g = sns.relplot(
                data=df_ss[df_ss["n_ranks"] > 1],
                x="n_ranks",
                y="speedup",
                hue="Solver",
                style="Strategy",
                col="N",
                kind="line",
                markers=True,
                markersize=8,
                facet_kws={"sharey": False},
                height=4,
                aspect=1.3,
            )
            for ax in g.axes.flat:
                # Ideal scaling line (only for parallel ranks)
                ax.plot(P_range_parallel, P_range_parallel, "k--", alpha=0.5, linewidth=1, label="Ideal")
                ax.set_xscale("log")
                ax.set_yscale("log")
                # Set explicit ticks at actual rank values
                ax.set_xticks(P_range_parallel)
                ax.set_xticklabels([str(p) for p in P_range_parallel], fontsize=8)
                ax.set_xlim(min(P_range_parallel) * 0.8, max(P_range_parallel) * 1.2)
                ax.grid(True, alpha=0.3)
            g.set_axis_labels("Number of Ranks", "Speedup S(P)")
            g.figure.suptitle("Strong Scaling: Speedup", y=1.02)

            output_file = fig_dir / f"{plot_num:02d}_strong_speedup.pdf"
            g.savefig(output_file, bbox_inches="tight")
            print(f"Saved: {output_file}")
            plt.close()
            plot_num += 1

            # Throughput plot: N as columns, Solver/Strategy as hue/style
            g = sns.relplot(
                data=df_ss,
                x="n_ranks",
                y="mlups",
                hue="Solver",
                style="Strategy",
                col="N",
                kind="line",
                markers=True,
                markersize=8,
                errorbar=("ci", 95),
                facet_kws={"sharey": False},
                height=4,
                aspect=1.3,
            )
            for ax in g.axes.flat:
                ax.set_xscale("log")
                # Set explicit ticks at actual rank values
                ax.set_xticks(P_range)
                ax.set_xticklabels([str(p) for p in P_range], fontsize=8)
                ax.set_xlim(min(P_range) * 0.8, max(P_range) * 1.2)
                ax.grid(True, alpha=0.3)
            g.set_axis_labels("Number of Ranks", "Throughput (MLup/s)")
            g.figure.suptitle("Strong Scaling: Throughput", y=1.02)

            output_file = fig_dir / f"{plot_num:02d}_strong_throughput.pdf"
            g.savefig(output_file, bbox_inches="tight")
            print(f"Saved: {output_file}")
            plt.close()
            plot_num += 1

    # =========================================================================
    # Weak Scaling Plots (single row, Solver/Strategy as hue/style)
    # =========================================================================

    # Track plot numbering across strong and weak scaling
    if 'plot_num' not in dir():
        plot_num = 1

    if not df_weak.empty:
        print("\n--- Weak Scaling Analysis ---")

        # Compute weak scaling metrics
        df_ws = compute_weak_scaling(df_weak)
        if df_ws.empty:
            print("No weak scaling data (missing n_ranks=1 baseline)")
        else:
            print(f"Weak scaling points: {len(df_ws)}")

            # Weak Scaling Throughput (single plot, Solver/Strategy as hue/style)
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.lineplot(
                data=df_ws,
                x="n_ranks",
                y="mlups",
                hue="Solver",
                style="Strategy",
                markers=True,
                markersize=8,
                errorbar=("ci", 95),
                ax=ax,
            )
            ax.set_xscale("log")
            ax.set_xlabel("Number of Ranks")
            ax.set_ylabel("Throughput (MLup/s)")
            ax.set_title("Weak Scaling: Throughput (~127³ points/rank)")
            ax.grid(True, alpha=0.3)
            ax.legend(title="Config")

            output_file = fig_dir / f"{plot_num:02d}_weak_throughput.pdf"
            fig.savefig(output_file, bbox_inches="tight")
            print(f"Saved: {output_file}")
            plt.close()
            plot_num += 1

    # =========================================================================
    # Summary
    # =========================================================================

    print("\n" + "=" * 70)
    print("SCALING SUMMARY")
    print("=" * 70)

    if not df_strong.empty and 'df_ss' in dir() and not df_ss.empty:
        print("\nStrong Scaling - Max Speedup by Configuration:")
        summary = df_ss.groupby(["Solver", "N", "Strategy"]).agg({
            "n_ranks": "max",
            "speedup": "max",
            "efficiency": "min"
        }).round(2)
        print(summary.to_string())

    if not df_weak.empty and 'df_ws' in dir() and not df_ws.empty:
        print("\nWeak Scaling - Efficiency at Max Ranks:")
        summary = df_ws.groupby(["Solver", "Strategy"]).agg({
            "n_ranks": "max",
            "efficiency": ["max", "min"]
        }).round(2)
        print(summary.to_string())

    print("\nDone!")


if __name__ == "__main__":
    main()

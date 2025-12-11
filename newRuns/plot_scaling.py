"""
Scaling Plots from Downloaded MLflow Data
==========================================

Creates strong and weak scaling plots from the downloaded CSV data.

Usage:
    uv run python newRuns/plot_scaling.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup
DATA_DIR = Path(__file__).parent
FIG_DIR = DATA_DIR / "figures" / "scaling"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Enable LaTeX rendering
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']
sns.set_theme()
plt.rcParams['text.usetex'] = True  # Re-enable after sns.set_theme()


def load_and_process_data():
    """Load and process scaling data from CSV."""
    df = pd.read_csv(DATA_DIR / "mlflow_scaling_runs.csv")

    # Convert columns
    df["N"] = pd.to_numeric(df["N"], errors="coerce")
    df["n_ranks"] = pd.to_numeric(df["n_ranks"], errors="coerce")
    df["wall_time"] = pd.to_numeric(df["wall_time"], errors="coerce")
    df["mlups"] = pd.to_numeric(df.get("mlups"), errors="coerce")
    df["iterations"] = pd.to_numeric(df.get("iterations"), errors="coerce")

    # Fill missing values
    df["solver"] = df["solver"].fillna("jacobi")
    df["strategy"] = df["strategy"].fillna("sliced")
    df["communicator"] = df["communicator"].fillna("custom")

    # Filter out rows with missing essential data
    df = df.dropna(subset=["N", "n_ranks", "wall_time"])

    # Create labels
    df["Solver"] = df["solver"].str.upper()
    df["Decomposition"] = df["strategy"].str.capitalize()
    df["Datatype"] = df["communicator"].str.capitalize()

    # Compute MLUPS if missing
    if df["mlups"].isna().all() and "iterations" in df.columns:
        df["mlups"] = (df["N"] ** 3 * df["iterations"]) / df["wall_time"] / 1e6

    return df


def compute_strong_scaling(df: pd.DataFrame) -> pd.DataFrame:
    """Compute strong scaling metrics (speedup and efficiency)."""
    results = []

    # Get baselines per (N, solver) - strategy irrelevant for sequential
    baselines = df[df["n_ranks"] == 1].groupby(["N", "solver"])["wall_time"].mean()

    for _, row in df.iterrows():
        try:
            N = int(row["N"])
            solver = row["solver"]
            P = int(row["n_ranks"])
            T_P = float(row["wall_time"])
        except (ValueError, TypeError):
            continue

        try:
            T_1 = baselines.loc[(N, solver)]
        except KeyError:
            continue

        speedup = T_1 / T_P if T_P > 0 else np.nan

        results.append({
            "N": N,
            "n_ranks": P,
            "strategy": row["strategy"],
            "Decomposition": str(row["strategy"]).capitalize() if pd.notna(row["strategy"]) else "Unknown",
            "solver": solver,
            "Solver": solver.upper() if pd.notna(solver) else "JACOBI",
            "Datatype": str(row["communicator"]).capitalize() if pd.notna(row.get("communicator")) else "Custom",
            "wall_time": T_P,
            "T_1": T_1,
            "speedup": speedup,
            "efficiency": (speedup / P) * 100 if P > 0 else np.nan,
            "mlups": row.get("mlups", np.nan),
        })

    return pd.DataFrame(results)


def compute_weak_scaling_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """Compute weak scaling efficiency based on MLUPS."""
    results = []

    # Group by solver, strategy, communicator
    for (solver, strategy, comm), group in df.groupby(["solver", "strategy", "communicator"]):
        group = group.sort_values("n_ranks")
        if len(group) < 2:
            continue

        # Get baseline (smallest rank count)
        baseline_row = group[group["n_ranks"] == group["n_ranks"].min()].iloc[0]
        baseline_mlups = baseline_row["mlups"]
        baseline_ranks = baseline_row["n_ranks"]

        if pd.isna(baseline_mlups) or baseline_mlups <= 0:
            continue

        for _, row in group.iterrows():
            P = row["n_ranks"]
            mlups = row["mlups"]

            if pd.isna(mlups) or mlups <= 0:
                continue

            # Weak scaling efficiency: actual MLUPS / ideal MLUPS
            # Ideal MLUPS scales linearly with ranks
            ideal_mlups = baseline_mlups * (P / baseline_ranks)
            efficiency = (mlups / ideal_mlups) * 100

            results.append({
                "N": int(row["N"]),
                "n_ranks": int(P),
                "Solver": solver.upper(),
                "Decomposition": str(strategy).capitalize(),
                "Datatype": str(comm).capitalize(),
                "mlups": mlups,
                "efficiency": efficiency,
                "source_experiment": row.get("source_experiment", ""),
            })

    return pd.DataFrame(results)


def main():
    print("Loading scaling data...")
    df = load_and_process_data()
    print(f"Loaded {len(df)} runs")

    # Identify strong vs weak scaling experiments
    # Strong scaling: same N, varying ranks
    # Weak scaling: N varies with ranks (constant work per rank)

    # Filter for v3 data (most complete)
    df_v3 = df[df["source_prefix"].str.contains("v3", na=False)]

    # Strong scaling from "scaling" and "fmg_scaling" experiments
    strong_exps = ["scaling", "fmg_scaling"]
    df_strong = df_v3[df_v3["source_experiment"].isin(strong_exps)]

    # Weak scaling from weak_scaling experiments
    weak_exps = ["weak_scaling_jacobi", "weak_scaling_fmg", "weak_scaling_v2", "weak_scaling_v2-LARGE", "Exam-weak_scaling"]
    df_weak = df[df["source_experiment"].isin(weak_exps)]

    print(f"Strong scaling runs: {len(df_strong)}")
    print(f"Weak scaling runs: {len(df_weak)}")

    plot_num = 1

    # ==========================================================================
    # Strong Scaling Plots
    # ==========================================================================

    if not df_strong.empty:
        print("\n--- Strong Scaling Analysis ---")
        df_ss = compute_strong_scaling(df_strong)

        if not df_ss.empty:
            print(f"Strong scaling data points: {len(df_ss)}")
            P_range = sorted(df_ss["n_ranks"].unique())
            P_range_parallel = [p for p in P_range if p > 1]

            # Group by N for separate facets
            N_values = sorted(df_ss["N"].unique())
            print(f"Problem sizes: {N_values}")

            # Plot 1: Strong Scaling Speedup
            df_plot = df_ss[df_ss["n_ranks"] > 1]
            if not df_plot.empty:
                g = sns.relplot(
                    data=df_plot,
                    x="n_ranks",
                    y="speedup",
                    hue="Solver",
                    style="Decomposition",
                    col="N",
                    kind="line",
                    markers=True,
                    markersize=8,
                    facet_kws={"sharey": False},
                    height=4,
                    aspect=1.2,
                )
                for ax in g.axes.flat:
                    ax.plot(P_range_parallel, P_range_parallel, "k--", alpha=0.5, linewidth=1, label="Ideal")
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    ax.set_xticks(P_range_parallel)
                    ax.set_xticks([], minor=True)
                    ax.set_xticklabels([str(int(p)) for p in P_range_parallel], fontsize=8)
                    ax.grid(True, alpha=0.3)
                g.set_axis_labels(r"Number of Ranks", r"Speedup $S(P)$")

                output_file = FIG_DIR / f"{plot_num:02d}_strong_speedup.pdf"
                g.savefig(output_file, bbox_inches="tight")
                print(f"Saved: {output_file}")
                plt.close()
                plot_num += 1

            # Plot 2: Strong Scaling Efficiency
            if not df_plot.empty:
                g = sns.relplot(
                    data=df_plot,
                    x="n_ranks",
                    y="efficiency",
                    hue="Solver",
                    style="Decomposition",
                    col="N",
                    kind="line",
                    markers=True,
                    markersize=8,
                    facet_kws={"sharey": True},
                    height=4,
                    aspect=1.2,
                )
                for ax in g.axes.flat:
                    ax.axhline(y=100, color="k", linestyle="--", alpha=0.5, linewidth=1)
                    ax.set_xscale("log")
                    ax.set_xticks(P_range_parallel)
                    ax.set_xticks([], minor=True)
                    ax.set_xticklabels([str(int(p)) for p in P_range_parallel], fontsize=8)
                    ax.set_ylim(0, 120)
                    ax.grid(True, alpha=0.3)
                g.set_axis_labels(r"Number of Ranks", r"Efficiency (\%)")

                output_file = FIG_DIR / f"{plot_num:02d}_strong_efficiency.pdf"
                g.savefig(output_file, bbox_inches="tight")
                print(f"Saved: {output_file}")
                plt.close()
                plot_num += 1

            # Plot 3: Strong Scaling Throughput
            g = sns.relplot(
                data=df_ss,
                x="n_ranks",
                y="mlups",
                hue="Solver",
                style="Decomposition",
                col="N",
                kind="line",
                markers=True,
                markersize=8,
                errorbar=("ci", 95),
                facet_kws={"sharey": False},
                height=4,
                aspect=1.2,
            )
            for ax in g.axes.flat:
                ax.set_xscale("log")
                ax.set_xticks(P_range)
                ax.set_xticks([], minor=True)
                ax.set_xticklabels([str(int(p)) for p in P_range], fontsize=8)
                ax.grid(True, alpha=0.3)
            g.set_axis_labels(r"Number of Ranks", r"Throughput (MLup/s)")

            output_file = FIG_DIR / f"{plot_num:02d}_strong_throughput.pdf"
            g.savefig(output_file, bbox_inches="tight")
            print(f"Saved: {output_file}")
            plt.close()
            plot_num += 1

    # ==========================================================================
    # Weak Scaling Plots
    # ==========================================================================

    if not df_weak.empty:
        print("\n--- Weak Scaling Analysis ---")
        df_ws = compute_weak_scaling_efficiency(df_weak)

        if not df_ws.empty:
            print(f"Weak scaling data points: {len(df_ws)}")

            # Calculate local grid size for grouping
            df_ws["local_size_raw"] = df_ws["N"] / np.cbrt(df_ws["n_ranks"])

            def bin_local_size(x):
                if 245 <= x <= 270:
                    return 257
                elif 500 <= x <= 530:
                    return 513
                elif 125 <= x <= 135:
                    return 129
                else:
                    return int(round(x))

            df_ws["local_size"] = df_ws["local_size_raw"].apply(bin_local_size)
            local_sizes = sorted(df_ws["local_size"].unique())
            print(f"Local grid sizes: {local_sizes}")

            # Create separate plots for each local grid size
            for local_size in local_sizes:
                df_local = df_ws[df_ws["local_size"] == local_size].copy()
                rank_ticks = sorted(df_local["n_ranks"].unique())

                if len(rank_ticks) < 2:
                    print(f"Skipping local_size={local_size} - only {len(rank_ticks)} rank value(s)")
                    continue

                # Aggregate duplicates
                df_agg = df_local.groupby(["n_ranks", "Solver", "Decomposition", "Datatype"]).agg({
                    "efficiency": "mean",
                    "mlups": "mean"
                }).reset_index()

                # Filter ticks to avoid overlap
                filtered_ticks = []
                for t in rank_ticks:
                    if not filtered_ticks or t > filtered_ticks[-1] * 1.15:
                        filtered_ticks.append(t)

                # Efficiency plot
                g = sns.relplot(
                    data=df_agg,
                    x="n_ranks",
                    y="efficiency",
                    col="Solver",
                    hue="Decomposition",
                    style="Datatype",
                    kind="line",
                    markers=True,
                    markersize=8,
                    linewidth=2,
                    height=5,
                    aspect=1.1,
                    facet_kws={"legend_out": False},
                )
                g.set_axis_labels(r"Number of Ranks", r"Weak Scaling Efficiency (\%)")

                for ax in g.axes.flat:
                    ax.axhline(y=100, color="k", linestyle="--", alpha=0.5, linewidth=1)
                    ax.set_xticks(filtered_ticks)
                    ax.set_xticklabels([str(int(r)) for r in filtered_ticks])
                    ax.set_ylim(0, 120)
                    ax.legend(loc="lower left", fontsize=9)

                plt.tight_layout()
                output_file = FIG_DIR / f"{plot_num:02d}_weak_efficiency_local{local_size}.pdf"
                g.savefig(output_file, bbox_inches="tight")
                print(f"Saved: {output_file}")
                plt.close()
                plot_num += 1

                # Throughput plot
                g = sns.relplot(
                    data=df_agg,
                    x="n_ranks",
                    y="mlups",
                    col="Solver",
                    hue="Decomposition",
                    style="Datatype",
                    kind="line",
                    markers=True,
                    markersize=8,
                    linewidth=2,
                    height=5,
                    aspect=1.1,
                    facet_kws={"legend_out": False},
                )
                g.set_axis_labels(r"Number of Ranks", r"Throughput (MLup/s)")

                for ax in g.axes.flat:
                    ax.set_xticks(filtered_ticks)
                    ax.set_xticklabels([str(int(r)) for r in filtered_ticks])
                    ax.legend(loc="upper left", fontsize=9)

                plt.tight_layout()
                output_file = FIG_DIR / f"{plot_num:02d}_weak_throughput_local{local_size}.pdf"
                g.savefig(output_file, bbox_inches="tight")
                print(f"Saved: {output_file}")
                plt.close()
                plot_num += 1

    print("\nDone!")


if __name__ == "__main__":
    main()

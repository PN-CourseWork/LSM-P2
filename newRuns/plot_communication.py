"""
Communication Analysis Plots from Downloaded MLflow Data
=========================================================

Creates communication analysis plots comparing placement strategies
and halo exchange methods.

Usage:
    uv run python newRuns/plot_communication.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup
DATA_DIR = Path(__file__).parent
FIG_DIR = DATA_DIR / "figures" / "communication"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Enable LaTeX rendering
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']
sns.set_theme()
plt.rcParams['text.usetex'] = True


def load_and_process_data():
    """Load and process communication data from CSV."""
    df = pd.read_csv(DATA_DIR / "mlflow_communication_runs.csv")

    # Convert columns
    df["N"] = pd.to_numeric(df["N"], errors="coerce")
    df["n_ranks"] = pd.to_numeric(df["n_ranks"], errors="coerce")
    df["wall_time"] = pd.to_numeric(df["wall_time"], errors="coerce")
    df["mlups"] = pd.to_numeric(df.get("mlups"), errors="coerce")
    df["iterations"] = pd.to_numeric(df.get("iterations"), errors="coerce")
    df["total_halo_time"] = pd.to_numeric(df.get("total_halo_time"), errors="coerce")
    df["total_compute_time"] = pd.to_numeric(df.get("total_compute_time"), errors="coerce")
    df["bandwidth_gb_s"] = pd.to_numeric(df.get("bandwidth_gb_s"), errors="coerce")

    # Fill missing values
    df["solver"] = df["solver"].fillna("jacobi")
    df["strategy"] = df["strategy"].fillna("sliced")
    df["communicator"] = df["communicator"].fillna("custom")

    # Add placement labels based on source experiment
    df["placement"] = df["source_experiment"].apply(
        lambda x: "Spread" if "spread" in str(x).lower() else
                  "Compact" if "compact" in str(x).lower() else "Unknown"
    )

    # Filter out rows with missing essential data
    df = df.dropna(subset=["N", "n_ranks"])

    # Create labels
    df["Decomposition"] = df["strategy"].str.capitalize()
    df["Communicator"] = df["communicator"].str.capitalize()
    df["Placement"] = df["placement"]

    # Compute derived metrics
    df["total_points"] = df["N"] ** 3
    df["points_per_rank"] = df["total_points"] / df["n_ranks"]
    df["local_size"] = df["points_per_rank"] ** (1/3)

    # Time per iteration
    df["wall_time_per_iter_ms"] = (df["wall_time"] / df["iterations"]) * 1e3
    df["halo_time_per_iter_ms"] = (df["total_halo_time"] / df["iterations"]) * 1e3
    df["compute_time_per_iter_ms"] = (df["total_compute_time"] / df["iterations"]) * 1e3
    df["halo_fraction"] = df["total_halo_time"] / df["wall_time"] * 100

    # Compute halo bandwidth if not present
    if df["bandwidth_gb_s"].isna().all():
        df["halo_bytes_per_iter"] = 6 * (df["local_size"] ** 2) * 8
        df["bandwidth_gb_s"] = (df["halo_bytes_per_iter"] * df["iterations"]) / df["total_halo_time"] / 1e9

    return df


def main():
    print("Loading communication data...")
    df = load_and_process_data()
    print(f"Loaded {len(df)} runs")

    # Filter to valid placement experiments
    df = df[df["Placement"].isin(["Spread", "Compact"])]
    print(f"After filtering to Spread/Compact: {len(df)} runs")

    if df.empty:
        print("No communication data with Spread/Compact placement found.")
        return

    print(f"Problem sizes N: {sorted(df['N'].dropna().unique())}")
    print(f"Placements: {df['Placement'].unique().tolist()}")

    plot_num = 1

    # ==========================================================================
    # Plot 1: Throughput (MLup/s)
    # ==========================================================================

    g = sns.relplot(
        data=df,
        x="N",
        y="mlups",
        hue="Decomposition",
        style="Communicator",
        col="Placement",
        kind="line",
        markers=True,
        markersize=8,
        errorbar=("ci", 95),
        facet_kws={"sharey": True},
        height=4,
        aspect=1.2,
    )
    g.set_axis_labels(r"Problem Size ($N$)", r"Throughput (MLup/s)")
    for ax in g.axes.flat:
        ax.grid(True, alpha=0.3)

    output_file = FIG_DIR / f"{plot_num:02d}_throughput.pdf"
    g.savefig(output_file, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()
    plot_num += 1

    # ==========================================================================
    # Plot 2: Wall Time per Iteration
    # ==========================================================================

    df_wall = df.dropna(subset=["wall_time_per_iter_ms"])
    if df_wall.empty:
        print("Skipping wall time plot - no data")
    else:
        g = sns.relplot(
            data=df_wall,
            x="N",
            y="wall_time_per_iter_ms",
            hue="Decomposition",
            style="Communicator",
            col="Placement",
            kind="line",
            markers=True,
            markersize=8,
            errorbar=("ci", 95),
            facet_kws={"sharey": True},
            height=4,
            aspect=1.2,
        )
        g.set_axis_labels(r"Problem Size ($N$)", r"Wall Time per Iteration (ms)")
        for ax in g.axes.flat:
            ax.grid(True, alpha=0.3)

        output_file = FIG_DIR / f"{plot_num:02d}_wall_time.pdf"
        g.savefig(output_file, bbox_inches="tight")
        print(f"Saved: {output_file}")
        plt.close()
        plot_num += 1

    # ==========================================================================
    # Plot 3: Halo Exchange Time per Iteration
    # ==========================================================================

    df_halo = df.dropna(subset=["halo_time_per_iter_ms"])
    if not df_halo.empty:
        g = sns.relplot(
            data=df_halo,
            x="N",
            y="halo_time_per_iter_ms",
            hue="Decomposition",
            style="Communicator",
            col="Placement",
            kind="line",
            markers=True,
            markersize=8,
            errorbar=("ci", 95),
            facet_kws={"sharey": True},
            height=4,
            aspect=1.2,
        )
        g.set_axis_labels(r"Problem Size ($N$)", r"Halo Time per Iteration (ms)")
        for ax in g.axes.flat:
            ax.grid(True, alpha=0.3)

        output_file = FIG_DIR / f"{plot_num:02d}_halo_time.pdf"
        g.savefig(output_file, bbox_inches="tight")
        print(f"Saved: {output_file}")
        plt.close()
        plot_num += 1

    # ==========================================================================
    # Plot 4: Halo Exchange Bandwidth
    # ==========================================================================

    df_bw = df.dropna(subset=["bandwidth_gb_s"])
    df_bw = df_bw[df_bw["bandwidth_gb_s"] > 0]
    if not df_bw.empty:
        g = sns.relplot(
            data=df_bw,
            x="N",
            y="bandwidth_gb_s",
            hue="Decomposition",
            style="Communicator",
            col="Placement",
            kind="line",
            markers=True,
            markersize=8,
            errorbar=("ci", 95),
            facet_kws={"sharey": True},
            height=4,
            aspect=1.2,
        )
        g.set_axis_labels(r"Problem Size ($N$)", r"Halo Bandwidth (GB/s)")
        for ax in g.axes.flat:
            ax.grid(True, alpha=0.3)

        output_file = FIG_DIR / f"{plot_num:02d}_halo_bandwidth.pdf"
        g.savefig(output_file, bbox_inches="tight")
        print(f"Saved: {output_file}")
        plt.close()
        plot_num += 1

    # ==========================================================================
    # Plot 5: Communication Overhead (% of total time)
    # ==========================================================================

    df_frac = df.dropna(subset=["halo_fraction"])
    if not df_frac.empty:
        g = sns.relplot(
            data=df_frac,
            x="N",
            y="halo_fraction",
            hue="Decomposition",
            style="Communicator",
            col="Placement",
            kind="line",
            markers=True,
            markersize=8,
            errorbar=("ci", 95),
            facet_kws={"sharey": True},
            height=4,
            aspect=1.2,
        )
        g.set_axis_labels(r"Problem Size ($N$)", r"Halo Fraction (\%)")
        for ax in g.axes.flat:
            ax.grid(True, alpha=0.3)

        output_file = FIG_DIR / f"{plot_num:02d}_halo_fraction.pdf"
        g.savefig(output_file, bbox_inches="tight")
        print(f"Saved: {output_file}")
        plt.close()
        plot_num += 1

    # ==========================================================================
    # Plot 6: Direct Placement Comparison
    # ==========================================================================

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: MLup/s comparison
    ax1 = axes[0]
    sns.lineplot(
        data=df,
        x="N",
        y="mlups",
        hue="Placement",
        style="Decomposition",
        markers=True,
        markersize=8,
        errorbar=("ci", 95),
        ax=ax1,
    )
    ax1.set_xlabel(r"Problem Size ($N$)")
    ax1.set_ylabel(r"Throughput (MLup/s)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right: Halo bandwidth comparison
    ax2 = axes[1]
    if not df_bw.empty:
        sns.lineplot(
            data=df_bw,
            x="N",
            y="bandwidth_gb_s",
            hue="Placement",
            style="Decomposition",
            markers=True,
            markersize=8,
            errorbar=("ci", 95),
            ax=ax2,
        )
    ax2.set_xlabel(r"Problem Size ($N$)")
    ax2.set_ylabel(r"Halo Bandwidth (GB/s)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = FIG_DIR / f"{plot_num:02d}_placement_comparison.pdf"
    fig.savefig(output_file, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()
    plot_num += 1

    # ==========================================================================
    # Summary Statistics
    # ==========================================================================

    print("\n" + "=" * 70)
    print("Summary: MLup/s by Placement x Decomposition")
    print("=" * 70)
    summary = df.pivot_table(
        index="N",
        columns=["Placement", "Decomposition"],
        values="mlups",
        aggfunc="mean"
    )
    print(summary.round(1).to_string())

    print("\nDone!")


if __name__ == "__main__":
    main()

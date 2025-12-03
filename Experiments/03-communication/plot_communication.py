"""
Communication Analysis
======================

Compares MPI rank placement strategies (spread vs compact) and
halo exchange methods (NumPy vs custom datatypes).

Experiments:
- comm_spread: Ranks spread across packages (ppr:6:package)
- comm_compact: Ranks packed per node (ppr:12:node)

Usage
-----

.. code-block:: bash

    # Run communication experiments first (on HPC)
    bsub < jobs/comm_spread.sh
    bsub < jobs/comm_compact.sh

    # Then plot results
    uv run python Experiments/03-communication/plot_communication.py mlflow=databricks
"""

# %%
# Setup
# -----

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import hydra
from omegaconf import DictConfig

from Poisson import get_project_root
from utils.mlflow.io import setup_mlflow_tracking, load_runs
from utils import plotting  # Apply scientific style


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Generate communication analysis plots from MLflow data."""

    repo_root = get_project_root()
    fig_dir = repo_root / "figures" / "communication"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # %%
    # Load Data from MLflow
    # ---------------------

    print("Loading data from MLflow...")
    setup_mlflow_tracking(mode=cfg.mlflow.mode)
    prefix = cfg.mlflow.get("project_prefix", "")

    # Load both placement experiments
    df_spread = load_runs("comm_spread", converged_only=False, project_prefix=prefix)
    df_compact = load_runs("comm_compact", converged_only=False, project_prefix=prefix)

    if df_spread.empty and df_compact.empty:
        print("No runs found in experiments 'comm_spread' or 'comm_compact'.")
        print("Run the experiments first on HPC:")
        print("  bsub < jobs/comm_spread.sh")
        print("  bsub < jobs/comm_compact.sh")
        return

    # Add placement labels
    df_spread["placement"] = "Spread"
    df_compact["placement"] = "Compact"

    # Combine datasets
    df = pd.concat([df_spread, df_compact], ignore_index=True)

    # Extract parameters and metrics from MLflow columns
    df["N"] = pd.to_numeric(df["params.N"], errors="coerce").astype(int)
    df["n_ranks"] = pd.to_numeric(df["params.n_ranks"], errors="coerce").astype(int)
    df["strategy"] = df["params.strategy"]
    df["communicator"] = df["params.communicator"]
    df["mlups"] = pd.to_numeric(df["metrics.mlups"], errors="coerce")
    df["wall_time"] = pd.to_numeric(df["metrics.wall_time"], errors="coerce")
    df["halo_time"] = pd.to_numeric(df["metrics.total_halo_time"], errors="coerce")
    df["compute_time"] = pd.to_numeric(df["metrics.total_compute_time"], errors="coerce")
    df["iterations"] = pd.to_numeric(df["metrics.iterations"], errors="coerce").astype(int)

    # Compute derived metrics
    df["wall_time_per_iter_ms"] = (df["wall_time"] / df["iterations"]) * 1e3
    df["halo_time_per_iter_ms"] = (df["halo_time"] / df["iterations"]) * 1e3
    df["compute_time_per_iter_ms"] = (df["compute_time"] / df["iterations"]) * 1e3
    df["halo_fraction"] = df["halo_time"] / df["wall_time"] * 100

    # Compute bandwidth (GB/s) - halo data volume per second
    # Each halo exchange: 6 faces × local_size² × 8 bytes (float64)
    # For 24 ranks with N points: local_size ≈ (N-2) / ranks^(1/3)
    df["total_points"] = df["N"] ** 3
    df["points_per_rank"] = df["total_points"] / df["n_ranks"]
    # Approximate halo surface area per rank (6 faces)
    df["local_size"] = df["points_per_rank"] ** (1/3)
    df["halo_bytes_per_iter"] = 6 * (df["local_size"] ** 2) * 8  # 8 bytes per double
    df["halo_bandwidth_GBs"] = (df["halo_bytes_per_iter"] * df["iterations"]) / df["halo_time"] / 1e9

    # Create labels
    df["Decomposition"] = df["strategy"].str.capitalize()
    df["Communicator"] = df["communicator"].str.capitalize()
    df["Placement"] = df["placement"]

    # Calculate total grid points in millions for x-axis
    df["Grid Points (M)"] = df["total_points"] / 1e6

    print(f"Loaded {len(df)} runs")
    print(f"Problem sizes N: {sorted(df['N'].unique())}")
    print(f"Grid points (M): {sorted(df['Grid Points (M)'].unique())}")
    print(f"Placements: {df['Placement'].unique().tolist()}")

    # %%
    # Plot 1: Throughput (MLup/s)
    # ---------------------------

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
    g.set_axis_labels("Problem Size (N)", "Throughput (MLup/s)")
    g.figure.suptitle("Solver Throughput", y=1.02)
    g.add_legend(title="Config")

    output_file = fig_dir / "01_throughput.pdf"
    g.savefig(output_file, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()

    # %%
    # Plot 2: Wall Time per Iteration
    # --------------------------------

    g = sns.relplot(
        data=df,
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
    g.set_axis_labels("Problem Size (N)", "Wall Time per Iteration (ms)")
    g.figure.suptitle("Total Time per Iteration", y=1.02)
    g.add_legend(title="Config")

    output_file = fig_dir / "02_wall_time.pdf"
    g.savefig(output_file, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()

    # %%
    # Plot 3: Halo Exchange Time per Iteration
    # -----------------------------------------

    g = sns.relplot(
        data=df,
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
    g.set_axis_labels("Problem Size (N)", "Halo Time per Iteration (ms)")
    g.figure.suptitle("Halo Exchange Latency", y=1.02)
    g.add_legend(title="Config")

    output_file = fig_dir / "03_halo_time.pdf"
    g.savefig(output_file, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()

    # %%
    # Plot 3: Halo Exchange Bandwidth
    # --------------------------------

    g = sns.relplot(
        data=df,
        x="N",
        y="halo_bandwidth_GBs",
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
    g.set_axis_labels("Problem Size (N)", "Halo Bandwidth (GB/s)")
    g.figure.suptitle("Halo Exchange Bandwidth", y=1.02)
    g.add_legend(title="Config")

    output_file = fig_dir / "04_halo_bandwidth.pdf"
    g.savefig(output_file, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()

    # %%
    # Plot 4: Communication Overhead (% of total time)
    # -------------------------------------------------

    g = sns.relplot(
        data=df,
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
    g.set_axis_labels("Problem Size (N)", "Halo Fraction (%)")
    g.figure.suptitle("Communication Overhead", y=1.02)
    g.add_legend(title="Config")

    output_file = fig_dir / "05_halo_fraction.pdf"
    g.savefig(output_file, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()

    # %%
    # Plot 5: Compute Time per Iteration
    # -----------------------------------

    g = sns.relplot(
        data=df,
        x="N",
        y="compute_time_per_iter_ms",
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
    g.set_axis_labels("Problem Size (N)", "Compute Time per Iteration (ms)")
    g.figure.suptitle("Compute Performance", y=1.02)
    g.add_legend(title="Config")

    output_file = fig_dir / "06_compute_time.pdf"
    g.savefig(output_file, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()

    # %%
    # Plot 6: Direct Placement Comparison (single plot)
    # --------------------------------------------------

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
    ax1.set_xlabel("Problem Size (N)")
    ax1.set_ylabel("Throughput (MLup/s)")
    ax1.set_title("Spread vs Compact: Throughput")
    ax1.legend(title="Config", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right: Halo bandwidth comparison
    ax2 = axes[1]
    sns.lineplot(
        data=df,
        x="N",
        y="halo_bandwidth_GBs",
        hue="Placement",
        style="Decomposition",
        markers=True,
        markersize=8,
        errorbar=("ci", 95),
        ax=ax2,
    )
    ax2.set_xlabel("Problem Size (N)")
    ax2.set_ylabel("Halo Bandwidth (GB/s)")
    ax2.set_title("Spread vs Compact: Bandwidth")
    ax2.legend(title="Config", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = fig_dir / "07_placement_direct.pdf"
    fig.savefig(output_file, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()

    # %%
    # Summary Statistics
    # ------------------

    print("\n" + "=" * 70)
    print("Summary: MLup/s by Placement × Decomposition")
    print("=" * 70)
    summary = df.pivot_table(
        index="N",
        columns=["Placement", "Decomposition"],
        values="mlups",
        aggfunc="mean"
    )
    print(summary.round(1).to_string())

    print("\n" + "=" * 70)
    print("Summary: Halo Bandwidth (GB/s) by Placement × Decomposition")
    print("=" * 70)
    summary = df.pivot_table(
        index="N",
        columns=["Placement", "Decomposition"],
        values="halo_bandwidth_GBs",
        aggfunc="mean"
    )
    print(summary.round(2).to_string())

    print("\n" + "=" * 70)
    print("Summary: Halo Fraction (%) by Placement × Communicator")
    print("=" * 70)
    summary = df.pivot_table(
        index="N",
        columns=["Placement", "Communicator"],
        values="halo_fraction",
        aggfunc="mean"
    )
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()

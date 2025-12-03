"""
Communication Analysis
======================

Compares NumPy array copies vs MPI custom datatypes for halo exchange,
and sliced vs cubic decomposition strategies.

Usage
-----

.. code-block:: bash

    # Run communication benchmark first
    mpiexec -n 4 uv run python run_solver.py +experiment=communication mlflow=databricks -m

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

    df = load_runs("communication", converged_only=False, project_prefix=prefix)

    if df.empty:
        print("No runs found in experiment 'communication'.")
        print("Run the experiment first:")
        print("  mpiexec -n 4 uv run python run_solver.py +experiment=communication mlflow=databricks -m")
        return

    # Extract parameters and metrics from MLflow columns
    df["N"] = pd.to_numeric(df["params.N"], errors="coerce").astype("Int64")
    df["strategy"] = df["params.strategy"]
    df["communicator"] = df["params.communicator"]
    df["mlups"] = pd.to_numeric(df["metrics.mlups"], errors="coerce")
    df["wall_time"] = pd.to_numeric(df["metrics.wall_time"], errors="coerce")
    df["halo_time"] = pd.to_numeric(df["metrics.total_halo_time"], errors="coerce")
    df["compute_time"] = pd.to_numeric(df["metrics.total_compute_time"], errors="coerce")
    df["iterations"] = pd.to_numeric(df["metrics.iterations"], errors="coerce").astype("Int64")

    # Compute derived metrics
    df["halo_time_per_iter_us"] = (df["halo_time"] / df["iterations"]) * 1e6
    df["compute_time_per_iter_ms"] = (df["compute_time"] / df["iterations"]) * 1e3

    # Create config label
    df["config"] = df["strategy"].str.capitalize() + " / " + df["communicator"].str.capitalize()

    # Keep only latest run per configuration
    df = df.sort_values("start_time").groupby(["N", "strategy", "communicator"]).last().reset_index()

    print(f"Loaded {len(df)} runs")
    print(f"Problem sizes: {sorted(df['N'].unique())}")
    print(f"Configurations: {df['config'].unique().tolist()}")

    # %%
    # Plot 1: Throughput Comparison
    # -----------------------------

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: MLup/s by configuration
    ax1 = axes[0]
    sns.barplot(
        data=df,
        x="N",
        y="mlups",
        hue="config",
        ax=ax1,
    )
    ax1.set_xlabel("Problem Size (N)")
    ax1.set_ylabel("Throughput (MLup/s)")
    ax1.set_title("Solver Throughput by Configuration")
    ax1.legend(title="Strategy / Communicator", fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")

    # Right: Halo time per iteration
    ax2 = axes[1]
    sns.barplot(
        data=df,
        x="N",
        y="halo_time_per_iter_us",
        hue="config",
        ax=ax2,
    )
    ax2.set_xlabel("Problem Size (N)")
    ax2.set_ylabel("Halo Time per Iteration (µs)")
    ax2.set_title("Halo Exchange Overhead")
    ax2.legend(title="Strategy / Communicator", fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_file = fig_dir / "01_throughput_comparison.pdf"
    fig.savefig(output_file, bbox_inches="tight")
    print(f"Saved: {output_file}")

    # %%
    # Plot 2: Time Breakdown
    # ----------------------

    fig, ax = plt.subplots(figsize=(10, 6))

    # Melt data for stacked bar
    df_melt = df.melt(
        id_vars=["N", "config"],
        value_vars=["compute_time", "halo_time"],
        var_name="component",
        value_name="time",
    )
    df_melt["component"] = df_melt["component"].map({
        "compute_time": "Compute",
        "halo_time": "Halo Exchange"
    })

    # Create pivot for stacked bar
    pivot = df.pivot_table(
        index=["N", "config"],
        values=["compute_time", "halo_time"],
        aggfunc="mean"
    ).reset_index()

    # Simple grouped bar showing compute vs halo fraction
    df["halo_fraction"] = df["halo_time"] / df["wall_time"] * 100

    sns.barplot(
        data=df,
        x="N",
        y="halo_fraction",
        hue="config",
        ax=ax,
    )
    ax.set_xlabel("Problem Size (N)")
    ax.set_ylabel("Halo Exchange Fraction (%)")
    ax.set_title("Communication Overhead as Percentage of Total Time")
    ax.legend(title="Strategy / Communicator", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_file = fig_dir / "02_halo_fraction.pdf"
    fig.savefig(output_file, bbox_inches="tight")
    print(f"Saved: {output_file}")

    # %%
    # Summary Statistics
    # ------------------

    print("\n" + "=" * 70)
    print("Summary: MLup/s by configuration")
    print("=" * 70)
    summary = df.pivot_table(index="N", columns="config", values="mlups", aggfunc="mean")
    print(summary.to_string())

    print("\n" + "=" * 70)
    print("Summary: Halo fraction (%) by configuration")
    print("=" * 70)
    summary = df.pivot_table(index="N", columns="config", values="halo_fraction", aggfunc="mean")
    print(summary.to_string())


if __name__ == "__main__":
    main()

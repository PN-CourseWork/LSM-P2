"""
Communication Analysis: Contiguous vs Non-Contiguous
=====================================================

Compares NumPy array copies vs MPI custom datatypes for halo exchange,
demonstrating the benefit of zero-copy communication for non-contiguous data.

Fetches data from MLflow. Run experiment first:
    uv run python Experiments/03-communication/compute_communication.py
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import hydra
from omegaconf import DictConfig

from Poisson import get_project_root
from utils.mlflow.io import setup_mlflow_tracking, load_runs


@hydra.main(config_path="../hydra-conf", config_name="03-communication", version_base=None)
def main(cfg: DictConfig):
    """Generate communication analysis plots from MLflow data."""

    # Setup
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 100

    repo_root = get_project_root()
    experiment_name = cfg.get("experiment_name", "03-communication")
    fig_dir = repo_root / "figures" / experiment_name.replace("03-", "")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load from MLflow
    print("Loading data from MLflow...")
    setup_mlflow_tracking(mode=cfg.mlflow.mode)

    df = load_runs(experiment_name, converged_only=False)

    if df.empty:
        print(f"No runs found in experiment '{experiment_name}'.")
        print("Run the experiment first: uv run python Experiments/03-communication/compute_communication.py")
        return

    # Extract parameters and metrics from MLflow columns
    df["N"] = df["params.N"].astype(int)
    df["local_N"] = df["params.local_N"].astype(int)
    df["halo_time_us"] = df["metrics.halo_time_mean_us"].astype(float)
    df["label"] = df["params.label"]

    print(f"Loaded {len(df)} measurements")
    print(f"Configurations: {df['label'].unique()}")
    print(f"Problem sizes: {sorted(df['N'].unique())}")

    # Halo Exchange Time vs Problem Size
    fig, ax = plt.subplots(figsize=(10, 6))

    palette = {
        "NumPy (sliced, contiguous)": "#1f77b4",
        "Custom (sliced, contiguous)": "#2ca02c",
        "NumPy (cubic, mixed)": "#d62728",
        "Custom (cubic, mixed)": "#ff7f0e",
    }

    sns.lineplot(
        data=df,
        x="local_N",
        y="halo_time_us",
        hue="label",
        style="label",
        markers=True,
        dashes=False,
        palette=palette,
        ax=ax,
        markersize=8,
        linewidth=2,
    )

    ax.set_xlabel("Local Subdomain Size (N / nprocs)", fontsize=12)
    ax.set_ylabel(r"Halo Exchange Time ($\mu$s)", fontsize=12)
    ax.set_title("Halo Exchange Performance: Sliced vs Cubic Decomposition", fontsize=13)
    ax.legend(title="Configuration", fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = fig_dir / "communication_comparison.pdf"
    plt.savefig(output_file, bbox_inches="tight")
    print(f"Saved: {output_file}")

    # Summary Statistics
    print("\n" + "=" * 70)
    print("Summary: Mean halo time (us) by local subdomain size")
    print("=" * 70)
    summary = df.groupby(["local_N", "label"])["halo_time_us"].mean()
    print(summary.unstack().to_string())


if __name__ == "__main__":
    main()

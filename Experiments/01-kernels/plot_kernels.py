"""
Visualization of Kernel Experiments
====================================

Comprehensive analysis and visualization of NumPy vs Numba kernel benchmarks.
Fetches data directly from MLflow.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hydra
from omegaconf import DictConfig

from Poisson import get_project_root
from utils.mlflow.io import load_runs, setup_mlflow_tracking
from utils import plotting  # Apply scientific style


@hydra.main(config_path="../hydra-conf", config_name="experiment/kernel", version_base=None)
def main(cfg: DictConfig):
    # Get experiment name from config or use default
    exp_name = cfg.experiment_name or "kernel"

    # Setup MLflow tracking based on config
    setup_mlflow_tracking(mode=cfg.mlflow.mode)

    repo_root = get_project_root()
    fig_dir = repo_root / "figures" / "kernels"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load data from MLflow
    print(f"Loading data from MLflow experiment '{exp_name}'...")
    df = load_runs(exp_name, converged_only=False)

    if df.empty:
        print(f"No runs found for experiment '{exp_name}'.")
        print("Run the experiment first:")
        print("  uv run python run_solver.py -cn experiment/kernel")
        return

    # Extract parameters
    df["N"] = df["params.N"].astype(int)
    df["use_numba"] = df["params.use_numba"].fillna("False").astype(str).str.lower() == "true"
    df["numba_threads"] = df["params.numba_threads"].fillna(1).astype(int)
    df["mlups"] = df["metrics.mlups"].astype(float)
    df["wall_time"] = df["metrics.wall_time"].astype(float)
    df["iterations"] = df["metrics.iterations"].astype(int)

    # Compute time per iteration
    df["time_per_iter_ms"] = (df["wall_time"] / df["iterations"]) * 1000

    # Create configuration labels
    df["config"] = df.apply(
        lambda row: "NumPy" if not row["use_numba"]
        else f"Numba ({row['numba_threads']} threads)",
        axis=1
    )

    print(f"Loaded {len(df)} runs")
    print(f"Problem sizes: {sorted(df['N'].unique())}")
    print(f"Configurations: {df['config'].unique()}")

    # %%
    # Plot 1: Performance (time per iteration)
    # -----------------------------------------

    fig, ax = plt.subplots()
    sns.lineplot(
        data=df,
        x="N",
        y="time_per_iter_ms",
        hue="config",
        style="config",
        markers=True,
        dashes=False,
        ax=ax,
    )
    N_values = sorted(df["N"].unique())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(N_values, labels=[f"${n}^3$" for n in N_values])
    ax.minorticks_off()
    ax.set_xlabel("Problem Size")
    ax.set_ylabel("Time per Iteration (ms)")
    ax.set_title("Kernel Performance Comparison")
    ax.legend(title="Kernel")

    output_file = fig_dir / "01_performance.pdf"
    fig.savefig(output_file, bbox_inches="tight")
    print(f"Saved: {output_file}")

    # %%
    # Plot 2: Throughput (Mlup/s)
    # ---------------------------

    fig, ax = plt.subplots()
    sns.lineplot(
        data=df,
        x="N",
        y="mlups",
        hue="config",
        style="config",
        markers=True,
        dashes=False,
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_xticks(N_values, labels=[f"${n}^3$" for n in N_values])
    ax.minorticks_off()
    ax.set_xlabel("Problem Size")
    ax.set_ylabel("Throughput (Mlup/s)")
    ax.set_title("Kernel Throughput Comparison")
    ax.legend(title="Kernel")

    output_file = fig_dir / "02_throughput.pdf"
    fig.savefig(output_file, bbox_inches="tight")
    print(f"Saved: {output_file}")

    # %%
    # Plot 3: Speedup (Numba vs NumPy)
    # --------------------------------

    # Get NumPy baseline times per N
    df_numpy = df[~df["use_numba"]].groupby("N")["time_per_iter_ms"].mean().reset_index()
    df_numpy = df_numpy.rename(columns={"time_per_iter_ms": "numpy_time"})

    # Compute speedup for Numba runs
    df_numba = df[df["use_numba"]].merge(df_numpy, on="N", how="left")
    df_numba["speedup"] = df_numba["numpy_time"] / df_numba["time_per_iter_ms"]

    if not df_numba.empty:
        fig, ax = plt.subplots()
        sns.lineplot(
            data=df_numba,
            x="N",
            y="speedup",
            hue="numba_threads",
            style="numba_threads",
            markers=True,
            dashes=False,
            ax=ax,
            palette="viridis",
        )
        N_vals = sorted(df_numba["N"].unique())
        ax.axhline(y=1, color="k", linestyle="--", alpha=0.5, label="NumPy baseline")
        ax.set_xscale("log")
        ax.set_xticks(N_vals, labels=[f"${n}^3$" for n in N_vals])
        ax.minorticks_off()
        ax.set_xlabel("Problem Size")
        ax.set_ylabel("Speedup vs NumPy")
        ax.set_title("Numba Speedup")
        ax.legend(title="Threads")

        output_file = fig_dir / "03_speedup.pdf"
        fig.savefig(output_file, bbox_inches="tight")
        print(f"Saved: {output_file}")
    else:
        print("No Numba runs found for speedup plot.")


if __name__ == "__main__":
    main()

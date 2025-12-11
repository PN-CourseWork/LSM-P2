"""
Visualization of Kernel Experiments
====================================

Comprehensive analysis and visualization of NumPy vs Numba kernel benchmarks.
Fetches data directly from MLflow.

Usage:
    uv run python Experiments/01-kernels/plot_kernels.py mlflow=databricks
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hydra
from omegaconf import DictConfig

from Poisson import get_project_root
from utils.mlflow.io import load_runs, setup_mlflow_tracking
from utils import plotting  # Apply scientific style


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Setup MLflow tracking based on config
    setup_mlflow_tracking(mode=cfg.mlflow.mode)
    prefix = cfg.mlflow.get("project_prefix", "")

    repo_root = get_project_root()
    fig_dir = repo_root / "figures" / "kernels"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load data from MLflow
    exp_name = "kernel"
    print(f"Loading data from MLflow experiment '{exp_name}'...")
    df = load_runs(exp_name, converged_only=False, project_prefix=prefix)

    if df.empty:
        print(f"No runs found for experiment '{exp_name}'.")
        print("Run the experiments first:")
        print("  uv run python run_solver.py +experiment=kernel/convergence_numpy mlflow=databricks -m")
        print("  uv run python run_solver.py +experiment=kernel/convergence_numba mlflow=databricks -m")
        print("  uv run python run_solver.py +experiment=kernel/benchmark_numpy mlflow=databricks -m")
        print("  uv run python run_solver.py +experiment=kernel/benchmark_numba mlflow=databricks -m")
        return

    # Extract parameters (handle missing columns)
    df["N"] = pd.to_numeric(df["params.N"], errors="coerce").astype("Int64")
    df["max_iter"] = pd.to_numeric(df["params.max_iter"], errors="coerce").astype("Int64")
    # Handle use_numba as either bool, int (0/1), or string
    if "params.use_numba" in df.columns:
        df["use_numba"] = pd.to_numeric(df["params.use_numba"], errors="coerce").fillna(0).astype(bool)
    else:
        df["use_numba"] = False

    # Use observed numba threads (actual runtime value)
    if "metrics.observed_numba_threads" in df.columns:
        df["numba_threads"] = pd.to_numeric(df["metrics.observed_numba_threads"], errors="coerce").fillna(1).astype(int)
    elif "params.specified_numba_threads" in df.columns:
        df["numba_threads"] = pd.to_numeric(df["params.specified_numba_threads"], errors="coerce").fillna(1).astype(int)
    else:
        df["numba_threads"] = 1

    df["mlups"] = pd.to_numeric(df["metrics.mlups"], errors="coerce")
    df["wall_time"] = pd.to_numeric(df["metrics.wall_time"], errors="coerce")
    df["iterations"] = pd.to_numeric(df["metrics.iterations"], errors="coerce").astype("Int64")
    df["final_error"] = pd.to_numeric(df["metrics.final_error"], errors="coerce")

    # Compute time per iteration
    df["time_per_iter_ms"] = (df["wall_time"] / df["iterations"]) * 1000

    # Create configuration labels
    df["config"] = df.apply(
        lambda row: "NumPy" if not row["use_numba"]
        else f"Numba ({row['numba_threads']}T)",
        axis=1
    )

    print(f"Loaded {len(df)} runs")
    print(f"Problem sizes: {sorted(df['N'].dropna().unique())}")
    print(f"Configurations: {df['config'].unique()}")

    # %%
    # Plot 1: Convergence (error vs iterations) - N=17 runs only
    # ----------------------------------------------------------

    df_conv = df[df["N"] == 17].copy()
    if not df_conv.empty:
        # Keep only latest run per (config, max_iter)
        df_conv = df_conv.sort_values("start_time").groupby(["config", "max_iter"]).last().reset_index()

        fig, ax = plt.subplots()
        sns.lineplot(
            data=df_conv,
            x="max_iter",
            y="final_error",
            hue="config",
            style="config",
            markers=True,
            dashes=False,
            ax=ax,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Algebraic Error")
        ax.set_title("Kernel Convergence Comparison (N=17)")
        ax.legend(title="Kernel")
        ax.grid(True, alpha=0.3)

        output_file = fig_dir / "01_convergence.pdf"
        fig.savefig(output_file, bbox_inches="tight")
        print(f"Saved: {output_file}")
    else:
        print("No convergence runs found (N=17).")

    # %%
    # Plot 2: Performance (MLup/s vs N) - benchmark runs only (N > 17)
    # ----------------------------------------------------------------

    df_bench = df[df["N"] > 17].copy()
    if not df_bench.empty:
        # Keep only latest run per (config, N)
        df_bench = df_bench.sort_values("start_time").groupby(["config", "N"]).last().reset_index()

        fig, ax = plt.subplots()
        sns.lineplot(
            data=df_bench,
            x="N",
            y="mlups",
            hue="config",
            style="config",
            markers=True,
            dashes=False,
            ax=ax,
        )
        N_values = sorted(df_bench["N"].unique())
        ax.set_xscale("log")
        ax.set_xticks(N_values, labels=[f"${n}^3$" for n in N_values])
        ax.minorticks_off()
        ax.set_xlabel("Problem Size")
        ax.set_ylabel("Throughput (Mlup/s)")
        ax.set_title("Kernel Throughput Comparison")
        ax.legend(title="Kernel")
        ax.grid(True, alpha=0.3)

        output_file = fig_dir / "02_throughput.pdf"
        fig.savefig(output_file, bbox_inches="tight")
        print(f"Saved: {output_file}")

        # %%
        # Plot 3: Speedup (Numba vs NumPy)
        # --------------------------------

        # Get NumPy baseline times per N
        df_numpy = df_bench[~df_bench["use_numba"]].groupby("N")["time_per_iter_ms"].mean().reset_index()
        df_numpy = df_numpy.rename(columns={"time_per_iter_ms": "numpy_time"})

        # Compute speedup for Numba runs
        df_numba = df_bench[df_bench["use_numba"]].merge(df_numpy, on="N", how="left")
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
            ax.grid(True, alpha=0.3)

            output_file = fig_dir / "03_speedup.pdf"
            fig.savefig(output_file, bbox_inches="tight")
            print(f"Saved: {output_file}")
        else:
            print("No Numba runs found for speedup plot.")

        # %%
        # Plot 4: Time per iteration (with confidence intervals)
        # -------------------------------------------------------

        # Use ALL benchmark runs (not deduplicated) for CI calculation
        df_bench_all = df[df["N"] > 17].copy()

        fig, ax = plt.subplots()
        sns.lineplot(
            data=df_bench_all,
            x="N",
            y="time_per_iter_ms",
            hue="config",
            style="config",
            markers=True,
            dashes=False,
            errorbar="ci",  # 95% confidence interval
            ax=ax,
        )
        N_values = sorted(df_bench_all["N"].unique())
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(N_values, labels=[f"${n}^3$" for n in N_values])
        ax.minorticks_off()
        ax.set_xlabel("Problem Size")
        ax.set_ylabel("Time per Iteration (ms)")
        ax.set_title("Kernel Timing Comparison")
        ax.legend(title="Kernel")
        ax.grid(True, alpha=0.3)

        output_file = fig_dir / "04_time_per_iter.pdf"
        fig.savefig(output_file, bbox_inches="tight")
        print(f"Saved: {output_file}")
    else:
        print("No benchmark runs found (N > 17).")


if __name__ == "__main__":
    main()

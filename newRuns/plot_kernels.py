"""
Kernel Benchmark Plots from Downloaded MLflow Data
===================================================

Creates kernel benchmark plots comparing NumPy vs Numba performance.

Usage:
    uv run python newRuns/plot_kernels.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup
DATA_DIR = Path(__file__).parent
FIG_DIR = DATA_DIR / "figures" / "kernels"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Enable LaTeX rendering
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']
sns.set_theme()
plt.rcParams['text.usetex'] = True


def load_and_process_data():
    """Load and process kernel data from CSV."""
    df = pd.read_csv(DATA_DIR / "mlflow_kernels_runs.csv")

    # Convert columns
    df["N"] = pd.to_numeric(df["N"], errors="coerce")
    df["wall_time"] = pd.to_numeric(df["wall_time"], errors="coerce")
    df["mlups"] = pd.to_numeric(df.get("mlups"), errors="coerce")
    df["iterations"] = pd.to_numeric(df.get("iterations"), errors="coerce")
    df["max_iter"] = pd.to_numeric(df.get("max_iter"), errors="coerce")
    df["final_error"] = pd.to_numeric(df.get("final_error"), errors="coerce")

    # Handle use_numba
    if "use_numba" in df.columns:
        df["use_numba"] = pd.to_numeric(df["use_numba"], errors="coerce").fillna(0).astype(bool)
    else:
        df["use_numba"] = False

    # Handle numba threads
    if "observed_numba_threads" in df.columns:
        df["numba_threads"] = pd.to_numeric(df["observed_numba_threads"], errors="coerce").fillna(1).astype(int)
    elif "specified_numba_threads" in df.columns:
        df["numba_threads"] = pd.to_numeric(df["specified_numba_threads"], errors="coerce").fillna(1).astype(int)
    elif "threads" in df.columns:
        df["numba_threads"] = pd.to_numeric(df["threads"], errors="coerce").fillna(1).astype(int)
    else:
        df["numba_threads"] = 1

    # Filter out rows with missing essential data
    df = df.dropna(subset=["N"])

    # Compute time per iteration
    df["time_per_iter_ms"] = (df["wall_time"] / df["iterations"]) * 1000

    # Create configuration labels
    df["config"] = df.apply(
        lambda row: "NumPy" if not row["use_numba"]
        else f"Numba ({int(row['numba_threads'])}T)",
        axis=1
    )

    return df


def main():
    print("Loading kernel data...")
    df = load_and_process_data()
    print(f"Loaded {len(df)} runs")
    print(f"Problem sizes: {sorted(df['N'].dropna().unique())}")
    print(f"Configurations: {df['config'].unique()}")

    plot_num = 1

    # ==========================================================================
    # Plot 1: Convergence (error vs iterations) - N=17 runs only
    # ==========================================================================

    df_conv = df[df["N"] == 17].copy()
    if not df_conv.empty and "max_iter" in df_conv.columns:
        # Keep only latest run per (config, max_iter)
        df_conv = df_conv.sort_values("start_time").groupby(["config", "max_iter"]).last().reset_index()
        df_conv = df_conv.dropna(subset=["final_error", "max_iter"])

        if not df_conv.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
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
            ax.set_xlabel(r"Iterations")
            ax.set_ylabel(r"Algebraic Error")
            ax.legend(title="Kernel")
            ax.grid(True, alpha=0.3)

            output_file = FIG_DIR / f"{plot_num:02d}_convergence.pdf"
            fig.savefig(output_file, bbox_inches="tight")
            print(f"Saved: {output_file}")
            plt.close()
            plot_num += 1
    else:
        print("No convergence runs found (N=17).")

    # ==========================================================================
    # Plot 2: Performance (MLup/s vs N) - benchmark runs only (N > 17)
    # ==========================================================================

    df_bench = df[df["N"] > 17].copy()
    df_bench = df_bench.dropna(subset=["mlups"])

    if not df_bench.empty:
        # Keep only latest run per (config, N)
        df_bench_latest = df_bench.sort_values("start_time").groupby(["config", "N"]).last().reset_index()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.lineplot(
            data=df_bench_latest,
            x="N",
            y="mlups",
            hue="config",
            style="config",
            markers=True,
            dashes=False,
            ax=ax,
        )
        N_values = sorted(df_bench_latest["N"].unique())
        ax.set_xscale("log")
        ax.set_xticks(N_values)
        ax.set_xticklabels([f"${int(n)}^3$" for n in N_values])
        ax.minorticks_off()
        ax.set_xlabel(r"Problem Size")
        ax.set_ylabel(r"Throughput (MLup/s)")
        ax.legend(title="Kernel")
        ax.grid(True, alpha=0.3)

        output_file = FIG_DIR / f"{plot_num:02d}_throughput.pdf"
        fig.savefig(output_file, bbox_inches="tight")
        print(f"Saved: {output_file}")
        plt.close()
        plot_num += 1

        # ==================================================================
        # Plot 3: Speedup (Numba vs NumPy)
        # ==================================================================

        # Get NumPy baseline times per N
        df_numpy = df_bench_latest[~df_bench_latest["use_numba"]].copy()
        if not df_numpy.empty:
            df_numpy_baseline = df_numpy.groupby("N")["time_per_iter_ms"].mean().reset_index()
            df_numpy_baseline = df_numpy_baseline.rename(columns={"time_per_iter_ms": "numpy_time"})

            # Compute speedup for Numba runs
            df_numba = df_bench_latest[df_bench_latest["use_numba"]].merge(df_numpy_baseline, on="N", how="left")
            df_numba["speedup"] = df_numba["numpy_time"] / df_numba["time_per_iter_ms"]

            if not df_numba.empty:
                fig, ax = plt.subplots(figsize=(8, 6))
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
                ax.set_xticks(N_vals)
                ax.set_xticklabels([f"${int(n)}^3$" for n in N_vals])
                ax.minorticks_off()
                ax.set_xlabel(r"Problem Size")
                ax.set_ylabel(r"Speedup vs NumPy")
                ax.legend(title="Threads")
                ax.grid(True, alpha=0.3)

                output_file = FIG_DIR / f"{plot_num:02d}_speedup.pdf"
                fig.savefig(output_file, bbox_inches="tight")
                print(f"Saved: {output_file}")
                plt.close()
                plot_num += 1

        # ==================================================================
        # Plot 4: Time per iteration (with confidence intervals)
        # ==================================================================

        # Use ALL benchmark runs for CI calculation
        df_bench_all = df[df["N"] > 17].copy()
        df_bench_all = df_bench_all.dropna(subset=["time_per_iter_ms"])

        if not df_bench_all.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.lineplot(
                data=df_bench_all,
                x="N",
                y="time_per_iter_ms",
                hue="config",
                style="config",
                markers=True,
                dashes=False,
                errorbar="ci",
                ax=ax,
            )
            N_values = sorted(df_bench_all["N"].unique())
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks(N_values)
            ax.set_xticklabels([f"${int(n)}^3$" for n in N_values])
            ax.minorticks_off()
            ax.set_xlabel(r"Problem Size")
            ax.set_ylabel(r"Time per Iteration (ms)")
            ax.legend(title="Kernel")
            ax.grid(True, alpha=0.3)

            output_file = FIG_DIR / f"{plot_num:02d}_time_per_iter.pdf"
            fig.savefig(output_file, bbox_inches="tight")
            print(f"Saved: {output_file}")
            plt.close()
            plot_num += 1

    else:
        print("No benchmark runs found (N > 17).")

    # ==========================================================================
    # Summary Statistics
    # ==========================================================================

    print("\n" + "=" * 70)
    print("Summary: MLup/s by Configuration")
    print("=" * 70)
    if not df_bench.empty:
        summary = df_bench.pivot_table(
            index="N",
            columns="config",
            values="mlups",
            aggfunc="mean"
        )
        print(summary.round(1).to_string())

    print("\nDone!")


if __name__ == "__main__":
    main()

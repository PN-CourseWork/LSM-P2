
"""
Visualization of Kernel Experiments
====================================

Comprehensive analysis and visualization of NumPy vs Numba kernel benchmarks.
Fetches data directly from MLflow.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import hydra
from omegaconf import DictConfig

from Poisson import get_project_root
from utils.mlflow.io import load_runs, get_mlflow_client, setup_mlflow_tracking

# %%
# Setup
# -----

sns.set_theme()

@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Setup MLflow tracking based on config
    setup_mlflow_tracking(mode=cfg.mlflow.mode)

    repo_root = get_project_root()
    fig_dir = repo_root / "figures" / "kernels"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # %%
    # Plot 1: Convergence Validation
    # -------------------------------

    print("Fetching convergence data from MLflow...")
    try:
        # Fetch all runs from the unified '01-kernels' experiment
        # and filter for 'convergence' mode
        all_runs = load_runs(exp_name_base, converged_only=False)
        df_conv_runs = all_runs[all_runs["params.run_mode"] == "convergence"]
        
        if df_conv_runs.empty:
            print(f"No convergence runs found for experiment '{exp_name_base}'.")
        else:
            client = get_mlflow_client()
            history_list = []
            
            for _, run in df_conv_runs.iterrows():
                run_id = run.run_id
                kernel = run["params.kernel"]
                # Fetch metric history for 'residual_history'
                try:
                    history = client.get_metric_history(run_id, "residual_history")
                    for m in history:
                        history_list.append({
                            "iteration": m.step,
                            "residual": m.value,
                            "kernel": kernel,
                            "N": int(run["params.N"])
                        })
                except Exception:
                    continue
                    
            if history_list:
                df_history = pd.DataFrame(history_list)
                
                g = sns.relplot(
                    data=df_history,
                    x="iteration",
                    y="residual",
                    col="N",
                    hue="kernel",
                    style="kernel",
                    kind="line",
                    dashes=True,
                    markers=False,
                    facet_kws={"sharey": True, "sharex": False},
                )

                g.set(xscale="log", yscale="log")
                g.set_axis_labels("Iteration", r"Algebraic Residual $||Au - f||_\infty$")
                g.set_titles(col_template="N={col_name}")
                g.fig.suptitle(r"Kernel Convergence Validation", y=1.02)
                g.savefig(fig_dir / "01_convergence_validation.pdf")
                print("Saved convergence plot.")
            else:
                 print("No metric history found for convergence runs.")

    except Exception as e:
        print(f"Skipping convergence plot: {e}")


    # %%
    # Plot 2 & 3: Benchmark Performance
    # ----------------------------------

    print("Fetching benchmark data from MLflow...")
    try:
        # Fetch all runs from the unified '01-kernels' experiment
        # and filter for 'benchmark' mode
        all_runs = load_runs(exp_name_base, converged_only=False)
        df_bench = all_runs[all_runs["params.run_mode"] == "benchmark"]
        
        if df_bench.empty:
            print(f"No benchmark runs found for experiment '{exp_name_base}'.")
        else:
            # Convert columns
            df_bench["N"] = df_bench["params.N"].astype(int)
            df_bench["mlups"] = df_bench["metrics.mlups"]
            # Use metrics.wall_time / metrics.iterations to get per-iter time
            df_bench["time_per_iter_ms"] = (df_bench["metrics.wall_time"] / df_bench["params.max_iter"].astype(float)) * 1000
            df_bench["num_threads"] = df_bench["params.threads"]
            df_bench["kernel"] = df_bench["params.kernel"]

            # Prepare configuration labels
            df_bench["config"] = df_bench.apply(
                lambda row: "NumPy"
                if row["kernel"] == "numpy"
                else f"Numba ({int(row['num_threads'])} threads)",
                axis=1,
            )

            # -- Plot 2: Performance --
            fig, ax = plt.subplots()
            sns.lineplot(
                data=df_bench,
                x="N",
                y="time_per_iter_ms",
                hue="config",
                style="config",
                markers=True,
                dashes=False,
                errorbar="ci",
                ax=ax,
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Problem Size (N)")
            ax.set_ylabel("Time per Iteration (ms)")
            ax.set_title("Kernel Performance Comparison")
            fig.savefig(fig_dir / "02_performance.pdf")
            print("Saved performance plot.")

            # -- Plot 3: Speedup --
            # Filter for Numba runs and compute speedup vs average NumPy time for that N
            df_numpy = df_bench[df_bench["kernel"] == "numpy"].groupby("N")["time_per_iter_ms"].mean().reset_index()
            df_numpy = df_numpy.rename(columns={"time_per_iter_ms": "numpy_mean_time"})
            
            df_speedup = df_bench[df_bench["kernel"] == "numba"].merge(df_numpy, on="N", how="left")
            df_speedup["speedup"] = df_speedup["numpy_mean_time"] / df_speedup["time_per_iter_ms"]
            df_speedup["thread_label"] = df_speedup["num_threads"].astype(str) + " threads"

            fig, ax = plt.subplots()
            sns.lineplot(
                data=df_speedup,
                x="N",
                y="speedup",
                hue="thread_label",
                style="thread_label",
                markers=True,
                dashes=False,
                errorbar="ci",
                ax=ax,
            )
            ax.set_xlabel("Problem Size (N)")
            ax.set_ylabel("Speedup vs NumPy")
            ax.set_title("Speedup (Numba vs NumPy)")
            fig.savefig(fig_dir / "03_speedup.pdf")
            print("Saved speedup plot.")

    except Exception as e:
        print(f"Skipping benchmark plots: {e}")

if __name__ == "__main__":
    main()

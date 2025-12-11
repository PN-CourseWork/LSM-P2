"""
FMG Spatial Convergence
=======================

Fetches FMG validation results from MLflow and plots spatial convergence
against the analytical solution.

Usage
-----

.. code-block:: bash

    # Run FMG experiment first
    uv run python run_solver.py --config-name=experiment/05-multigrid-fmg

    # Then plot results
    uv run python Experiments/05-multigrid/plot_multigrid_fmg.py
"""

# %%
# Setup
# -----

import hydra
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig

from Poisson import get_project_root
from utils.mlflow.io import setup_mlflow_tracking, load_runs
from utils import plotting  # Apply scientific style


@hydra.main(config_path="../hydra-conf", config_name="experiment/multigrid", version_base=None)
def main(cfg: DictConfig) -> None:
    """Plot FMG spatial convergence with data from MLflow."""

    repo_root = get_project_root()
    fig_dir = repo_root / "figures" / "multigrid"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # %%
    # Load Data from MLflow
    # ---------------------

    print("Loading data from MLflow...")
    setup_mlflow_tracking(mode=cfg.mlflow.mode)

    experiment_name = cfg.get("experiment_name", "05-multigrid-fmg")
    df = load_runs(experiment_name, converged_only=False)

    if df.empty:
        print(f"No runs found in experiment '{experiment_name}'.")
        print("Run the experiment first:")
        print("  uv run python Experiments/run_experiment.py --config-name=experiment/05-multigrid-fmg")
        return

    # Extract parameters and metrics from MLflow columns (handle missing columns)
    df["N"] = pd.to_numeric(df.get("params.N"), errors="coerce").astype("Int64")
    df["final_error"] = pd.to_numeric(df.get("metrics.final_error"), errors="coerce")
    df["Decomposition"] = df.get("params.strategy", pd.Series("sliced", index=df.index)).fillna("sliced").str.capitalize()

    print(f"Loaded {len(df)} FMG results")
    print(f"Decompositions: {df['Decomposition'].unique()}")
    print(f"Problem sizes: {sorted(df['N'].unique())}")

    # %%
    # Plot Convergence
    # ----------------

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.lineplot(
        data=df,
        x="N",
        y="final_error",
        hue="Decomposition",
        style="Decomposition",
        markers=True,
        dashes=True,
        ax=ax,
    )

    # Reference O(N^-2) based on first decomposition
    first_decomp = df["Decomposition"].iloc[0]
    ref_df = df[df["Decomposition"] == first_decomp]
    N_min, N_max = ref_df["N"].min(), ref_df["N"].max()
    err_at_N_min = ref_df[ref_df["N"] == N_min]["final_error"].iloc[0]
    ax.plot(
        [N_min, N_max],
        [err_at_N_min, err_at_N_min * (N_min / N_max) ** 2],
        "k:",
        alpha=0.6,
        label=r"$O(N^{-2})$",
    )

    N_vals = sorted(df["N"].unique())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(N_vals, labels=[f"${n}^3$" for n in N_vals])
    ax.minorticks_off()
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Grid Size")
    ax.set_ylabel("L2 Error")
    ax.set_title("Spatial Convergence: FMG")
    ax.legend()

    fig.tight_layout()
    output_file = fig_dir / "fmg_convergence.pdf"
    fig.savefig(output_file)
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()

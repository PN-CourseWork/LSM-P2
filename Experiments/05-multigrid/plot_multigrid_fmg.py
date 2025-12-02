"""
Plot FMG Spatial Convergence
============================

Fetches FMG validation results from MLflow and plots spatial convergence
against the analytical solution. Run experiment first with:

    uv run python Experiments/run_experiment.py --config-name=05-multigrid-fmg
"""

import hydra
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig

from Poisson import get_project_root
from utils.mlflow.io import setup_mlflow_tracking, load_runs


@hydra.main(config_path="../hydra-conf", config_name="05-multigrid-fmg", version_base=None)
def main(cfg: DictConfig) -> None:
    """Plot FMG spatial convergence with data from MLflow."""

    # Setup
    sns.set_style()

    repo_root = get_project_root()
    fig_dir = repo_root / "figures" / "multigrid"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load from MLflow
    print("Loading data from MLflow...")
    setup_mlflow_tracking(mode=cfg.mlflow.mode)

    experiment_name = cfg.get("experiment_name", "05-multigrid-fmg")
    df = load_runs(experiment_name, converged_only=False)

    if df.empty:
        print(f"No runs found in experiment '{experiment_name}'.")
        print("Run the experiment first: uv run python Experiments/run_experiment.py --config-name=05-multigrid-fmg")
        return

    # Extract parameters and metrics from MLflow columns
    df["N"] = df["params.N"].astype(int)
    df["final_error"] = df["metrics.final_error"].astype(float)
    df["Decomposition"] = df["params.strategy"].str.capitalize()

    print(f"Loaded {len(df)} FMG results")
    print(f"Decompositions: {df['Decomposition'].unique()}")
    print(f"Problem sizes: {sorted(df['N'].unique())}")

    # Plot
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

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlabel("Grid Size N")
    ax.set_ylabel("L2 Error")
    ax.set_title("Spatial Convergence: FMG")
    ax.legend()

    fig.tight_layout()
    output_file = fig_dir / "fmg_convergence.pdf"
    fig.savefig(output_file)
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()

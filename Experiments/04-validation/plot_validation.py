"""
Validation Analysis and Visualization
======================================

Analyze and visualize spatial convergence for solver validation.
Verifies O(h²) spatial accuracy by comparing numerical solutions
against the analytical solution u(x,y,z) = sin(πx)sin(πy)sin(πz).

Usage
-----

.. code-block:: bash

    # Run validation experiment first
    uv run python run_solver.py +experiment=validation

    # Then plot results
    uv run python Experiments/04-validation/plot_validation.py
"""

# %%
# Setup
# -----

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import hydra
from omegaconf import DictConfig

from Poisson import get_project_root
from utils.mlflow.io import setup_mlflow_tracking, load_runs
from utils import plotting  # Apply scientific style


def compute_order_of_accuracy(N_values, errors):
    """Compute order of accuracy from consecutive grid refinements.

    Order p satisfies: error ~ N^(-p) ~ h^p
    So p = log(e1/e2) / log(N2/N1)
    """
    orders = []
    for i in range(len(N_values) - 1):
        N_ratio = N_values[i + 1] / N_values[i]  # Fine/coarse
        error_ratio = errors[i] / errors[i + 1]
        order = np.log(error_ratio) / np.log(N_ratio)
        orders.append(order)
    return orders


@hydra.main(config_path="../hydra-conf", config_name="experiment/validation", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run validation plotting with Hydra configuration."""

    # %%
    # Initialize
    # ----------

    repo_root = get_project_root()
    fig_dir = repo_root / "figures" / "validation"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # %%
    # Load Data from MLflow
    # ---------------------

    print("Loading data from MLflow...")
    setup_mlflow_tracking(mode=cfg.mlflow.mode)

    experiment_name = cfg.get("experiment_name", "validation")
    df = load_runs(experiment_name, converged_only=False)

    if df.empty:
        print(f"No runs found in experiment '{experiment_name}'.")
        print("Run the experiment first:")
        print("  uv run python run_solver.py +experiment=validation")
        return

    # Extract parameters and metrics
    df["N"] = df["params.N"].astype(int)
    df["final_error"] = df["metrics.final_error"].astype(float)
    df["solver"] = df["params.solver"].fillna("jacobi")
    df["n_ranks"] = df["params.n_ranks"].astype(int)
    df["h"] = 2.0 / (df["N"] - 1)  # Grid spacing

    # Get strategy and communicator from params
    df["strategy"] = df["params.strategy"].fillna("sliced")
    df["communicator"] = df["params.communicator"].fillna("custom")

    # Keep only the latest run per (solver, N, strategy, communicator) combination
    df = df.sort_values("start_time").groupby(["solver", "N", "strategy", "communicator"]).last().reset_index()

    # Filter to expected grid sizes per solver
    expected_N = {
        "jacobi": [17, 33, 65],
        "fmg": [65, 129, 257],
    }
    df = df[df.apply(lambda r: r["N"] in expected_N.get(r["solver"], []), axis=1)]

    print(f"\nLoaded {len(df)} validation results")
    print(f"Solvers: {df['solver'].unique()}")
    print(f"Grid sizes per solver:")
    for solver in df["solver"].unique():
        sizes = sorted(df[df["solver"] == solver]["N"].values)
        print(f"  {solver}: {sizes}")

    # %%
    # Compute Order of Accuracy
    # -------------------------

    print("\n" + "=" * 50)
    print("Order of Accuracy Analysis")
    print("=" * 50)

    for solver in df["solver"].unique():
        print(f"\n{solver.upper()}:")
        solver_df = df[df["solver"] == solver]
        for strategy in solver_df["strategy"].unique():
            for communicator in solver_df["communicator"].unique():
                method_df = solver_df[
                    (solver_df["strategy"] == strategy) & (solver_df["communicator"] == communicator)
                ].sort_values("N")
                if method_df.empty:
                    continue
                N_vals = method_df["N"].values
                errors = method_df["final_error"].values

                print(f"  {strategy} / {communicator}:")
                print(f"    {'N':>6} {'L2 Error':>12} {'Order':>8}")
                print(f"    {'-'*6} {'-'*12} {'-'*8}")

                orders = compute_order_of_accuracy(N_vals, errors)
                for i, (N, err) in enumerate(zip(N_vals, errors)):
                    order_str = f"{orders[i]:.2f}" if i < len(orders) else "-"
                    print(f"    {N:>6} {err:>12.4e} {order_str:>8}")

                avg_order = np.mean(orders) if orders else 0
                print(f"    Average order: {avg_order:.2f} (expected: 2.00)")

    # %%
    # Plot Convergence (separate panels per solver)
    # ----------------------------------------------

    # Prepare data for plotting
    df["Solver"] = df["solver"].str.upper()
    df["Strategy"] = df["strategy"]
    df["Communicator"] = df["communicator"]

    g = sns.relplot(
        data=df,
        x="N",
        y="final_error",
        col="Solver",
        hue="Strategy",
        style="Communicator",
        markers=True,
        kind="line",
        height=3.5,
        aspect=1.0,
        facet_kws={"sharex": False, "sharey": False, "legend_out": True},
    )

    # Add O(N^-2) reference line to each panel and format x-axis as N³
    for ax, solver in zip(g.axes.flat, df["Solver"].unique()):
        solver_df = df[df["Solver"] == solver]
        N_min, N_max = solver_df["N"].min(), solver_df["N"].max()
        N_ref = np.array([N_min * 0.8, N_max * 1.2])
        # Scale reference to pass through middle of data
        err_mid = solver_df["final_error"].median()
        N_mid = solver_df["N"].median()
        scale = err_mid * (N_mid**2)
        err_ref = scale / (N_ref**2)
        ax.loglog(N_ref, err_ref, "k--", alpha=0.5, linewidth=1, label=r"$O(N^{-2})$")
        N_vals = sorted(solver_df["N"].unique())
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(N_vals, labels=[f"${n}^3$" for n in N_vals])
        ax.minorticks_off()
        ax.grid(True, alpha=0.3)

    # Shared legend is automatically placed outside by legend_out=True
    g.set_axis_labels("Grid Size", r"$L_2$ error")
    g.figure.suptitle("Spatial Convergence Validation", y=1.02)
    g.tight_layout()

    output_file = fig_dir / "spatial_convergence.pdf"
    g.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()

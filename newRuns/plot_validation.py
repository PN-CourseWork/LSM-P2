"""
Validation Plots from Downloaded MLflow Data
=============================================

Creates spatial convergence validation plots.

Usage:
    uv run python newRuns/plot_validation.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup
DATA_DIR = Path(__file__).parent
FIG_DIR = DATA_DIR / "figures" / "validation"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Enable LaTeX rendering
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']
sns.set_theme()
plt.rcParams['text.usetex'] = True


def compute_order_of_accuracy(N_values, errors):
    """Compute order of accuracy from consecutive grid refinements."""
    orders = []
    for i in range(len(N_values) - 1):
        N_ratio = N_values[i + 1] / N_values[i]
        error_ratio = errors[i] / errors[i + 1]
        if error_ratio > 0 and N_ratio > 0:
            order = np.log(error_ratio) / np.log(N_ratio)
            orders.append(order)
    return orders


def load_and_process_data():
    """Load and process validation data from CSV."""
    df = pd.read_csv(DATA_DIR / "mlflow_validation_runs.csv")

    # Convert columns
    df["N"] = pd.to_numeric(df["N"], errors="coerce")
    df["final_error"] = pd.to_numeric(df.get("final_error"), errors="coerce")
    df["n_ranks"] = pd.to_numeric(df.get("n_ranks"), errors="coerce")

    # Fill missing values
    df["solver"] = df["solver"].fillna("jacobi")
    df["strategy"] = df["strategy"].fillna("sliced")
    df["communicator"] = df["communicator"].fillna("custom")

    # Filter out rows with missing essential data
    df = df.dropna(subset=["N", "final_error"])

    # Create labels
    df["Solver"] = df["solver"].str.upper()
    df["Strategy"] = df["strategy"].str.capitalize()
    df["Communicator"] = df["communicator"].str.capitalize()

    # Compute grid spacing
    df["h"] = 2.0 / (df["N"] - 1)

    return df


def main():
    print("Loading validation data...")
    df = load_and_process_data()
    print(f"Loaded {len(df)} runs")

    if df.empty:
        print("No validation data found.")
        return

    # Keep only latest run per (solver, N, strategy, communicator)
    df = df.sort_values("start_time").groupby(["solver", "N", "strategy", "communicator"]).last().reset_index()

    print(f"Solvers: {df['solver'].unique()}")
    print(f"Grid sizes per solver:")
    for solver in df["solver"].unique():
        sizes = sorted(df[df["solver"] == solver]["N"].values)
        print(f"  {solver}: {sizes}")

    # ==========================================================================
    # Compute Order of Accuracy
    # ==========================================================================

    print("\n" + "=" * 50)
    print("Order of Accuracy Analysis")
    print("=" * 50)

    for solver in df["solver"].unique():
        print(f"\n{solver.upper()}:")
        solver_df = df[df["solver"] == solver]
        for strategy in solver_df["strategy"].unique():
            for communicator in solver_df["communicator"].unique():
                method_df = solver_df[
                    (solver_df["strategy"] == strategy) &
                    (solver_df["communicator"] == communicator)
                ].sort_values("N")

                if method_df.empty or len(method_df) < 2:
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

    # ==========================================================================
    # Plot Convergence
    # ==========================================================================

    plot_num = 1

    # Separate panels per solver
    solvers = df["Solver"].unique()

    if len(solvers) > 0:
        g = sns.relplot(
            data=df,
            x="N",
            y="final_error",
            col="Solver",
            hue="Strategy",
            style="Communicator",
            markers=True,
            dashes=True,
            kind="line",
            height=4,
            aspect=1.0,
            facet_kws={"sharex": False, "sharey": False},
        )

        # Add O(N^-2) reference line
        for ax, solver in zip(g.axes.flat, solvers):
            solver_df = df[df["Solver"] == solver]
            if solver_df.empty:
                continue

            N_min, N_max = solver_df["N"].min(), solver_df["N"].max()
            N_ref = np.array([N_min * 0.8, N_max * 1.2])

            # Scale reference to pass through middle of data
            err_mid = solver_df["final_error"].median()
            N_mid = solver_df["N"].median()
            scale = err_mid * (N_mid ** 2)
            err_ref = scale / (N_ref ** 2)

            ax.loglog(N_ref, err_ref, "k--", alpha=0.5, linewidth=1, label=r"$O(N^{-2})$")

            N_vals = sorted(solver_df["N"].unique())
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks(N_vals)
            ax.set_xticklabels([f"${int(n)}^3$" for n in N_vals])
            ax.minorticks_off()
            ax.grid(True, alpha=0.3)

        g.set_axis_labels(r"Grid Size", r"$L_2$ Error")

        output_file = FIG_DIR / f"{plot_num:02d}_spatial_convergence.pdf"
        g.savefig(output_file, bbox_inches="tight")
        print(f"\nSaved: {output_file}")
        plt.close()
        plot_num += 1

    # ==========================================================================
    # Single combined plot
    # ==========================================================================

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.lineplot(
        data=df,
        x="N",
        y="final_error",
        hue="Solver",
        style="Strategy",
        markers=True,
        dashes=True,
        ax=ax,
    )

    # Reference line
    N_min, N_max = df["N"].min(), df["N"].max()
    N_ref = np.array([N_min * 0.8, N_max * 1.2])
    err_mid = df["final_error"].median()
    N_mid = df["N"].median()
    scale = err_mid * (N_mid ** 2)
    err_ref = scale / (N_ref ** 2)
    ax.loglog(N_ref, err_ref, "k--", alpha=0.5, linewidth=1, label=r"$O(N^{-2})$")

    N_vals = sorted(df["N"].unique())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(N_vals)
    ax.set_xticklabels([f"${int(n)}^3$" for n in N_vals])
    ax.minorticks_off()
    ax.set_xlabel(r"Grid Size")
    ax.set_ylabel(r"$L_2$ Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_file = FIG_DIR / f"{plot_num:02d}_convergence_combined.pdf"
    fig.savefig(output_file, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()

    print("\nDone!")


if __name__ == "__main__":
    main()

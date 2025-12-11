"""
Plotting script for Jacobi vs FMG timings.
Reveals grid traversal patterns using data from validation experiment.
"""

import hydra
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from Poisson import get_project_root
from utils import plotting  # Apply scientific style
from utils.mlflow.io import setup_mlflow_tracking, load_runs, get_mlflow_client


@hydra.main(config_path="../hydra-conf", config_name="experiment/validation", version_base=None)
def main(cfg: DictConfig) -> None:
    """Plot timing comparison using validation experiment data."""

    # Setup paths
    repo_root = get_project_root()
    figure_dir = repo_root / "figures" / "multigrid"
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Load data from MLflow
    print("Loading timing data from MLflow validation experiment...")
    setup_mlflow_tracking(mode=cfg.mlflow.mode)

    df = load_runs("validation", converged_only=False)
    if df.empty:
        print("Warning: No validation data found.")
        return

    # Get one Jacobi and one FMG run (largest N for each)
    df["solver"] = df["params.solver"].fillna("jacobi")
    df["N"] = df["params.N"].astype(int)

    client = get_mlflow_client()
    history_data = []

    for solver in ["jacobi", "fmg"]:
        solver_df = df[df["solver"] == solver].sort_values("N", ascending=False)
        if solver_df.empty:
            continue

        # Get the largest N run
        run = solver_df.iloc[0]
        run_id = run.run_id
        N = run["N"]
        print(f"  Loading {solver.upper()} timeseries (N={N})...")

        # Fetch compute_times and halo_exchange_times metrics
        for metric_name in ["compute_times", "halo_exchange_times"]:
            try:
                history = client.get_metric_history(run_id, metric_name)
                for m in history:
                    history_data.append({
                        "step": m.step,
                        "time_sec": m.value,
                        "timing_type": "Compute" if "compute" in metric_name else "Halo Exchange",
                        "method": solver.upper(),
                    })
            except Exception as e:
                print(f"    Warning: Could not load {metric_name}: {e}")

    if not history_data:
        print("Warning: No timing data found in validation runs.")
        return

    df_timings = pd.DataFrame(history_data)
    print(f"Loaded {len(df_timings)} timing measurements")

    # Plot
    print("Generating plot...")

    g = sns.relplot(
        data=df_timings,
        x="step",
        y="time_sec",
        hue="timing_type",
        col="method",
        kind="line",
        height=4,
        aspect=1.3,
        facet_kws={"sharex": False, "sharey": False}
    )

    g.set_titles("{col_name}")
    g.set_axis_labels("Step / Iteration", "Time (s)")
    g.figure.suptitle("Grid Traversal Patterns: Jacobi vs FMG", y=1.02)
    g.tight_layout()

    output_path = figure_dir / "timings_traversal_pattern.pdf"
    g.savefig(output_path, bbox_inches="tight")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

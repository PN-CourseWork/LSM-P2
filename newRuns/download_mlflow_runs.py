"""Download MLflow runs from Databricks workspace.

Downloads runs from scaling and communication experiments to CSV files.

Usage:
    uv run python newRuns/download_mlflow_runs.py
"""

import pandas as pd
from pathlib import Path
import mlflow

# Setup Databricks MLflow tracking
try:
    mlflow.login(backend="databricks", interactive=False)
    mlflow.set_tracking_uri("databricks")
    print("Connected to Databricks MLflow tracking.")
except Exception as e:
    print(f"Failed to connect to Databricks: {e}")
    print("Make sure you have configured databricks-cli credentials.")
    exit(1)

# Output directory
OUTPUT_DIR = Path(__file__).parent

# All project prefixes to search
PROJECT_PREFIXES = [
    "/Shared/LSM-PoissonMPI-ExamPrep",
    "/Shared/LSM-PoissonMPI-v3",
    "/Shared/LSM-PoissonMPI-v2",
    "/Shared/LSM-PoissonMPI",
]

# Experiments to download by category
EXPERIMENT_CATEGORIES = {
    # Scaling experiments
    "scaling": [
        "scaling",
        "fmg_scaling",
        "weak_scaling_jacobi",
        "weak_scaling_fmg",
        "weak_scaling",
        "weak_scaling_v2",
        "weak_scaling_v2-LARGE",
        "Exam-weak_scaling",
        "06-scaling-strong_cubic",
        "06-scaling-strong_sliced",
        "06-scaling-weak_cubic",
        "06-scaling-weak_sliced",
        "06-scaling-FMG_strong_cubic",
        "06-scaling-FMG_strong_sliced",
        "06-scaling-FMG_weak_cubic",
        "06-scaling-FMG_weak_sliced",
        "Experiment-06-Scaling",
        "jacobi_strong_1node",
    ],
    # Communication experiments
    "communication": [
        "comm_spread",
        "comm_compact",
        "communication",
        "Experiment-03-Communication",
    ],
    # Kernel experiments
    "kernels": [
        "kernel",
        "kernels",
        "kernels_benchmark",
        "kernels_convergence",
        "Experiment-01-Kernels",
    ],
    # Validation experiments
    "validation": [
        "validation",
        "Experiment-04-Validation",
    ],
    # Multigrid experiments
    "multigrid": [
        "multigrid",
        "Experiment-05-Multigrid",
    ],
}


def download_experiment(experiment_name: str, prefix: str) -> pd.DataFrame:
    """Download all runs from an experiment."""
    full_name = f"{prefix}/{experiment_name}"

    client = mlflow.tracking.MlflowClient()

    # Find all experiments with this name (there can be duplicates)
    experiments = client.search_experiments(
        filter_string=f"name = '{full_name}'"
    )

    if not experiments:
        return pd.DataFrame()

    exp_ids = [exp.experiment_id for exp in experiments]

    # Fetch all runs
    df = mlflow.search_runs(
        experiment_ids=exp_ids,
        order_by=["start_time DESC"],
    )

    # Filter out parent runs
    if "tags.is_parent" in df.columns:
        df = df[df["tags.is_parent"] != "true"]

    if not df.empty:
        df["source_experiment"] = experiment_name
        df["source_prefix"] = prefix

    return df


def main():
    client = mlflow.tracking.MlflowClient()

    # Download runs grouped by category
    results = {}

    print("\n" + "=" * 70)
    print("Downloading runs from all prefixes...")
    print("=" * 70)

    for category, exp_names in EXPERIMENT_CATEGORIES.items():
        print(f"\n--- {category.upper()} ---")

        if category not in results:
            results[category] = []

        for exp_name in exp_names:
            for prefix in PROJECT_PREFIXES:
                df = download_experiment(exp_name, prefix)
                if not df.empty:
                    results[category].append(df)
                    print(f"  {prefix}/{exp_name}: {len(df)} runs")

    # Combine and save by category
    print("\n" + "=" * 70)
    print("Saving to CSV...")
    print("=" * 70)

    for category, dfs in results.items():
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)

            # Remove exact duplicates (same run_id)
            combined = combined.drop_duplicates(subset=["run_id"])

            # Select relevant columns
            param_cols = [c for c in combined.columns if c.startswith("params.")]
            metric_cols = [c for c in combined.columns if c.startswith("metrics.")]
            tag_cols = ["tags.mlflow.runName", "source_experiment", "source_prefix"]
            meta_cols = ["run_id", "start_time", "end_time", "status"]

            available_cols = [c for c in meta_cols + tag_cols + param_cols + metric_cols
                            if c in combined.columns]

            output_df = combined[available_cols].copy()

            # Simplify column names
            output_df.columns = [c.replace("params.", "").replace("metrics.", "").replace("tags.", "")
                                for c in output_df.columns]

            output_file = OUTPUT_DIR / f"mlflow_{category}_runs.csv"
            output_df.to_csv(output_file, index=False)
            print(f"Saved {len(output_df)} runs to {output_file.name}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total = 0
    for category, dfs in results.items():
        if dfs:
            count = sum(len(df) for df in dfs)
            total += count
            print(f"  {category}: {count} runs")
    print(f"  TOTAL: {total} runs")

    print("\nDone!")


if __name__ == "__main__":
    main()

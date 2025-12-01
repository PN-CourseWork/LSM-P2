"""
Solver Validation
=================

Run solver across configurations and save results.
"""

from pathlib import Path
import mlflow

from Poisson import run_solver, get_project_root
from utils.mlflow.io import setup_mlflow_tracking

# --- MLflow Setup ---
setup_mlflow_tracking()

# --- Script Setup ---
repo_root = get_project_root()
data_dir = repo_root / "data" / "04-validation"
data_dir.mkdir(parents=True, exist_ok=True)

# Parameters
problem_sizes = [16, 32, 48]
rank_counts = [4, 8] # Added 8 ranks for more coverage
configurations = [
    ("sliced", "numpy"),
    ("sliced", "custom"),
    ("cubic", "numpy"),
    ("cubic", "custom"),
]
max_iterations = 50000 # Increased from 5000
tolerance = 1e-10

print("Solver Validation")
print("=" * 60)

# --- MLflow Logging ---
# To disable MLflow logging, comment out the following lines.
try:
    mlflow.set_experiment("/Shared/LSM-PoissonMPI/Experiment-04-Validation")
    with mlflow.start_run(run_name="Validation-Results-Set") as run:
        print(f"INFO: Started MLflow run '{run.info.run_name}' for artifact logging.")
        
        # Log main parameters for the validation set
        mlflow.log_params({
            "problem_sizes": problem_sizes,
            "rank_counts": rank_counts,
            "configurations": [f"{s}_{c}" for s,c in configurations],
            "max_iterations": max_iterations,
            "tolerance": tolerance,
        })

        for N in problem_sizes:
            print(f"\nN={N}")
            for n_ranks in rank_counts:
                for strategy, comm in configurations:
                    output_file = data_dir / f"N{N}_np{n_ranks}_{strategy}_{comm}.h5"
                    print(f"  {output_file.name}...", end=" ", flush=True)

                    result = run_solver(
                        N=N,
                        n_ranks=n_ranks,
                        strategy=strategy,
                        communicator=comm,
                        max_iter=max_iterations,
                        tol=tolerance,
                        validate=True,
                        output=str(output_file),
                    )

                    if "error" in result:
                        print("ERROR")
                    else:
                        print("done")
                        # Log each generated H5 file as an artifact
                        if output_file.exists():
                            mlflow.log_artifact(str(output_file), artifact_path=f"N{N}")
                            print(f"    ✓ Logged artifact: {output_file.name}")
        
except Exception as e:
    print(f"  ✗ WARNING: MLflow logging failed: {e}")

print(f"\nSaved results to: {data_dir}")
"""
Full Multigrid (FMG) Spatial Convergence Study
----------------------------------------------

Runs the FMG solver on several mesh sizes and decomposition strategies
to measure spatial accuracy (expected ~O(N^-2)).

MultigridPoisson uses DistributedGrid internally for unified halo exchange.
"""

import mlflow
from Poisson import get_project_root, run_solver
from utils.mlflow.io import setup_mlflow_tracking

# --- MLflow Setup ---
setup_mlflow_tracking()

# Paths
repo_root = get_project_root()
data_dir = repo_root / "data" / "05-multigrid"
data_dir.mkdir(parents=True, exist_ok=True)

# Parameters
problem_sizes = [65, 129, 257]
rank_counts = [8]
decompositions = ["sliced", "cubic"]

n_smooth = 3
omega = 2.0 / 3.0  # Standard 3D Jacobi relaxation
max_iterations = 3000
tolerance = 1e-16
fmg_cycles = 1  # Single FMG cycle (discretization accuracy achieved in 1 cycle)

print("FMG Spatial Convergence")
print("=" * 60)

# --- MLflow Logging ---
try:
    mlflow.set_experiment("/Shared/LSM-PoissonMPI/Experiment-05-Multigrid")
    with mlflow.start_run(run_name="FMG-Spatial-Convergence") as run:
        print(f"INFO: Started MLflow run '{run.info.run_name}' for artifact logging.")

        # Log main parameters
        mlflow.log_params({
            "problem_sizes": problem_sizes,
            "rank_counts": rank_counts,
            "decompositions": decompositions,
            "n_smooth": n_smooth,
            "omega": omega,
            "fmg_cycles": fmg_cycles,
            "max_iterations": max_iterations,
            "tolerance": tolerance,
        })

        for N in problem_sizes:
            print(f"\nN={N}")
            for n_ranks in rank_counts:
                for decomp in decompositions:
                    output = data_dir / f"FMG_N{N}_r{n_ranks}_{decomp}.h5"
                    print(f"  {output.name}...", end=" ", flush=True)

                    result = run_solver(
                        N=N,
                        n_ranks=n_ranks,
                        solver_type="fmg",
                        strategy=decomp,
                        n_smooth=n_smooth,
                        omega=omega,
                        fmg_cycles=fmg_cycles,
                        max_iter=max_iterations,
                        tol=tolerance,
                        validate=True,
                        output=str(output),
                    )

                    if "error" in result:
                        print("ERROR")
                        print(f"    {result.get('error', '')[:200]}")
                    else:
                        print("done")
                        # Log each generated H5 file as an artifact
                        if output.exists():
                            mlflow.log_artifact(str(output), artifact_path=f"N{N}")
                            print(f"    ✓ Logged artifact: {output.name}")

except Exception as e:
    print(f"  ✗ WARNING: MLflow logging failed: {e}")

print(f"\nSaved results to: {data_dir}")

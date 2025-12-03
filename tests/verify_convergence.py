"""
Convergence Verification Script.

Loads results from the 'TEST-convergence' MLflow experiments and verifies
that the order of accuracy is approximately 2 (second-order) for all solvers.

Usage:
    uv run python tests/verify_convergence.py
"""

import shutil
import subprocess
import sys
import mlflow
import numpy as np
import pandas as pd
from scipy.stats import linregress

def run_simulations():
    """Run the Hydra sweeps to generate data."""
    print("🚀 Starting Convergence Sweeps...")

    # Clean up old test experiments (permanently delete directories + trash)
    mlflow.set_tracking_uri("./mlruns")
    client = mlflow.MlflowClient()
    from mlflow.entities import ViewType
    for exp in client.search_experiments(view_type=ViewType.ALL):
        if exp.name.startswith("TEST-convergence"):
            print(f"   🗑️  Deleting old experiment: {exp.name}")
            shutil.rmtree(f"./mlruns/{exp.experiment_id}", ignore_errors=True)
            shutil.rmtree(f"./mlruns/.trash/{exp.experiment_id}", ignore_errors=True)

    experiments = [
        "test/jacobi_convergence",
        "test/fmg_convergence"
    ]

    for exp in experiments:
        print(f"   ▶ Running {exp}...")
        cmd = [
            "uv", "run", "python", "run_solver.py",
            f"+experiment={exp}",
            "--multirun"
        ]
        
        try:
            subprocess.run(cmd, check=True) #, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"❌ Simulation {exp} failed with exit code {e.returncode}")
            sys.exit(1)
            
    print("✅ All simulations completed successfully.")

def check_convergence():
    # Run the physics first
    run_simulations()

    print("\n📊 Analyzing Results...")
    
    # Connect to local MLflow
    mlflow.set_tracking_uri("./mlruns")
    client = mlflow.MlflowClient()
    
    # Search for all experiments matching our pattern
    experiments = [e for e in client.search_experiments() if e.name.startswith("TEST-convergence")]
    
    if not experiments:
        print("Error: No 'TEST-convergence' experiments found.")
        sys.exit(1)
        
    # Fetch all runs from these experiments
    runs = mlflow.search_runs(
        experiment_ids=[e.experiment_id for e in experiments],
        filter_string="status = 'FINISHED'"
    )
    
    if runs.empty:
        print("No finished runs found.")
        sys.exit(1)

    print(f"Found {len(runs)} runs. Analyzing convergence...\n")
    
    # Group by Solver, Strategy, and Communicator
    # We handle n_ranks as fixed (or just another grouping col if it varies)
    group_cols = ["params.solver", "params.strategy", "params.communicator"]
    # Check if columns exist (they might be NaN if not logged properly, but they should be)
    for col in group_cols:
        if col not in runs.columns:
            runs[col] = "unknown"

    grouped = runs.groupby(group_cols)
    
    all_passed = True
    
    for (solver, strategy, comm), group in grouped:
        # Extract N and Error
        # Ensure N is sorted
        group["N"] = group["params.N"].astype(int)
        group = group.sort_values("N")
        
        Ns = group["N"].values
        errors = group["metrics.final_error"].values
        
        if len(Ns) < 2:
            print(f"⚠️  {solver} ({strategy}/{comm}): Not enough data points (N={Ns})")
            continue
            
        # Calculate Grid Spacing h ~ 1/(N-1)
        hs = 1.0 / (Ns - 1)
        
        
        # L2 norm
        errors_l2 = group["metrics.final_error"].values
        
        # Log-Log Regression for L2 Error: log(E_L2) = p_L2 * log(h) + C
        slope_l2, intercept_l2, r_value_l2, p_value_l2, std_err_l2 = linregress(np.log(hs), np.log(errors_l2))
        
        # Check Order of Accuracy (expecting ~2.0 for L2)
        is_good_l2 = 1.8 <= slope_l2 <= 2.2
        status_l2 = "✅ PASS" if is_good_l2 else "❌ FAIL"
        if not is_good_l2:
            all_passed = False
            
        print(f"{status_l2} {solver} ({strategy}/{comm}) L2: Order p = {slope_l2:.2f} (R²={r_value_l2**2:.4f})")
        print(f"      Ns: {Ns}")
        print(f"      L2 Errors: {['{:.2e}'.format(e) for e in errors_l2]}")
        print("-" * 60)

    if all_passed:
        print("\nAll convergence tests passed!")
        sys.exit(0)
    else:
        print("\nSome tests failed.")
        sys.exit(1)

if __name__ == "__main__":
    check_convergence()

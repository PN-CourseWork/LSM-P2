"""
Compute script for comparing Jacobi and FMG timings.
Target: N=257, 8 ranks.
"""

import hydra
from omegaconf import DictConfig
from Poisson import get_project_root, run_solver


@hydra.main(config_path="../hydra-conf", config_name="05-multigrid-timings", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run timing comparison with Hydra configuration."""

    # Setup paths
    repo_root = get_project_root()
    data_dir = repo_root / "data" / "05-multigrid"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Get configuration from Hydra
    N = cfg.N
    n_ranks = cfg.n_ranks
    strategy = cfg.strategy
    communicator = cfg.communicator

    print("Running Timing Comparison Experiment")
    print(f"Configuration: N={N}, Ranks={n_ranks}, Strategy={strategy}, Communicator={communicator}")

    # --- Run Jacobi ---
    print("\nRunning Jacobi (200 iterations)...")
    jacobi_out = data_dir / "timings_jacobi.h5"
    # We use run_solver which wraps the CLI/subprocess
    res_jacobi = run_solver(
        N=N,
        n_ranks=n_ranks,
        solver_type="jacobi",
        strategy=strategy,
        communicator=communicator,
        max_iter=200,
        tol=1e-20,  # Force it to run all iterations
        output=str(jacobi_out)
    )

    if "error" in res_jacobi:
        print(f"Jacobi failed: {res_jacobi['error']}")
        exit(1)
    else:
        print(f"Jacobi finished. Saved to {jacobi_out}")

    # --- Run FMG ---
    print("\nRunning FMG (sliced custom)...")
    fmg_out = data_dir / "timings_fmg.h5"
    res_fmg = run_solver(
        N=N,
        n_ranks=n_ranks,
        solver_type="fmg",
        strategy=strategy,
        communicator=communicator,
        n_smooth=3,  # Standard
        fmg_cycles=1,
        # We want to see the FMG pattern, 1 cycle is enough to show the hierarchy traversal
        output=str(fmg_out)
    )

    if "error" in res_fmg:
        print(f"FMG failed: {res_fmg['error']}")
        exit(1)
    else:
        print(f"FMG finished. Saved to {fmg_out}")

    print("\nDone.")


if __name__ == "__main__":
    main()

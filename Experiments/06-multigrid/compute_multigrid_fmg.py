"""
Full Multigrid (FMG) Spatial rgence Study
----------------------------------------------

Runs the FMG solver on several mesh sizes, communicators, and decomposition
strategies to measure spatial accuracy (expected ~O(N^-2)).
"""

from Poisson import get_project_root, run_solver

# Paths
repo_root = get_project_root()
data_dir = repo_root / "data" / "multigrid_fmg"
data_dir.mkdir(parents=True, exist_ok=True)

# Parameters
problem_sizes = [65, 129, 257]
rank_counts = [8]
communicators = ["numpy", "custom"]
decompositions = ["sliced", "cubic"]

n_smooth = 5
omega = 2.0 / 3.0  # Standard 3D Jacobi relaxation
max_iterations = 3000
tolerance = 1e-16
fmg_post_cycles = 2  # More cycles needed for cubic decomposition

print("FMG Spatial Convergence")
print("=" * 60)

for N in problem_sizes:
    print(f"\nN={N}")
    for n_ranks in rank_counts:
        for decomp in decompositions:
            for comm in communicators:
                output = data_dir / f"FMG_N{N}_r{n_ranks}_{decomp}_{comm}.h5"
                print(f"  {output.name}...", end=" ", flush=True)

                result = run_solver(
                    N=N,
                    n_ranks=n_ranks,
                    solver_type="fmg",
                    strategy=decomp,
                    communicator=comm,
                    n_smooth=n_smooth,
                    omega=omega,
                    fmg_post_cycles=fmg_post_cycles,
                    max_iter=max_iterations,
                    tol=tolerance,
                    validate=True,
                    output=str(output),
                )

                if "error" in result:
                    print("ERROR")
                else:
                    print("done")

print(f"\nSaved results to: {data_dir}")

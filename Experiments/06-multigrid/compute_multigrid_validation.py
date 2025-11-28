"""
Multigrid Spatial Convergence Study
-----------------------------------

Runs the MultigridPoisson solver on several mesh sizes and communicators
to measure spatial accuracy (expected ~O(N^-2)).
"""

from Poisson import get_project_root, run_solver

# Paths
repo_root = get_project_root()
data_dir = repo_root / "data" / "multigrid"
data_dir.mkdir(parents=True, exist_ok=True)

# Parameters
problem_sizes = [33, 65, 129, 257, 513]  # Power-of-two grids + 1 for clean coarsening
rank_counts = [4]
communicators = ["numpy", "custom"]

levels = 4
n_smooth = 3
max_iterations = 30
tolerance = 1e-8

print("Multigrid Spatial Convergence")
print("=" * 60)

for N in problem_sizes:
    print(f"\nN={N}")
    for n_ranks in rank_counts:
        for comm in communicators:
            output = data_dir / f"MG_N{N}_r{n_ranks}_{comm}.h5"
            print(f"  {output.name}...", end=" ", flush=True)

            result = run_solver(
                N=N,
                n_ranks=n_ranks,
                solver_type="multigrid",
                strategy="sliced",
                communicator=comm,
                n_smooth=n_smooth,
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

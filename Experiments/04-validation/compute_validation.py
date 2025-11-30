"""
Solver Validation
=================

Run solver across configurations and save results.
"""

from Poisson import run_solver, get_project_root

# Setup
repo_root = get_project_root()
data_dir = repo_root / "data" / "validation"
data_dir.mkdir(parents=True, exist_ok=True)

# Parameters
problem_sizes = [16, 32, 48]
rank_counts = [4]
configurations = [
    ("sliced", "numpy"),
    ("sliced", "custom"),
    ("cubic", "numpy"),
    ("cubic", "custom"),
]
max_iterations = 5000
tolerance = 1e-10  # Need tight tolerance to reach discretization floor

print("Solver Validation")
print("=" * 60)

for N in problem_sizes:
    print(f"\nN={N}")
    for n_ranks in rank_counts:
        for strategy, comm in configurations:
            output = data_dir / f"N{N}_np{n_ranks}_{strategy}_{comm}.h5"
            print(f"  {output.name}...", end=" ", flush=True)

            result = run_solver(
                N=N,
                n_ranks=n_ranks,
                strategy=strategy,
                communicator=comm,
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

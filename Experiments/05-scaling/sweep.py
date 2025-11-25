"""
Scaling Sweep: Strong and Weak Scaling Experiments
===================================================

Executes run_solver.py with varying ranks and problem sizes to measure
strong scaling (fixed N, varying ranks) and weak scaling (constant work per rank).
"""

import subprocess
from pathlib import Path

script_dir = Path(__file__).parent
run_solver = script_dir / "run_solver.py"


def run_experiment(n_ranks, N, strategy="sliced", communicator="numpy", **kwargs):
    """Run a single scaling experiment."""
    cmd = [
        "mpiexec", "-n", str(n_ranks),
        "uv", "run", "python", str(run_solver),
        "--N", str(N),
        "--strategy", strategy,
        "--communicator", communicator,
    ]

    for key, value in kwargs.items():
        arg_name = f"--{key.replace('_', '-')}"
        # Handle boolean flags (store_true arguments)
        if isinstance(value, bool):
            if value:
                cmd.append(arg_name)
        else:
            cmd.extend([arg_name, str(value)])

    print(f"\n{'='*60}")
    print(f"Running: N={N}, ranks={n_ranks}, {strategy}+{communicator}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, cwd=script_dir.parent.parent)
    return result.returncode == 0


def strong_scaling(N, rank_counts, **kwargs):
    """Strong scaling: fixed problem size, increasing ranks."""
    print("\n" + "#" * 60)
    print(f"# STRONG SCALING: N={N}")
    print("#" * 60)

    for n_ranks in rank_counts:
        run_experiment(n_ranks, N, **kwargs)


def weak_scaling(N_per_rank, rank_counts, **kwargs):
    """Weak scaling: constant work per rank (N scales with ranks)."""
    print("\n" + "#" * 60)
    print(f"# WEAK SCALING: {N_per_rank}^3 points per rank")
    print("#" * 60)

    for n_ranks in rank_counts:
        # Scale N so each rank has approximately N_per_rank^3 points
        # For cubic: N^3/P ~ N_per_rank^3 => N ~ N_per_rank * P^(1/3)
        # For sliced: (N-2)/P * N^2 ~ N_per_rank^3 => approximate
        N = int(N_per_rank * (n_ranks ** (1/3))) + 2
        run_experiment(n_ranks, N, **kwargs)


if __name__ == "__main__":
    # Configuration
    strategies = ["sliced", "cubic"]
    communicators = ["numpy"]
    rank_counts = [4, 8]

    # Common solver settings
    solver_opts = {
        "max_iter": 500,
        "tolerance": 0.0,
        "experiment": "poisson-scaling",
        "no_mlflow": False,  # Set to False to enable MLflow
    }

    for strategy in strategies:
        for comm in communicators:
            # Strong scaling: N=64 with 1,2,4,8 ranks
            strong_scaling(
                N=64,
                rank_counts=rank_counts,
                strategy=strategy,
                communicator=comm,
                **solver_opts
            )

            # Weak scaling: ~32^3 points per rank
            weak_scaling(
                N_per_rank=32,
                rank_counts=rank_counts,
                strategy=strategy,
                communicator=comm,
                **solver_opts
            )

    print("\n" + "=" * 60)
    print("Scaling sweep complete!")
    print("=" * 60)

"""
Run Validation Experiments
===========================

Helper script to run validation experiments and generate plots.
"""
import subprocess
import sys
from pathlib import Path


def run_validation(n_ranks=4):
    """Run validation experiment with specified number of ranks."""
    print(f"\n{'='*60}")
    print(f"Running validation with {n_ranks} ranks")
    print(f"{'='*60}\n")

    cmd = [
        "mpiexec",
        "-n", str(n_ranks),
        "uv", "run", "python",
        "compute_validation.py"
    ]

    result = subprocess.run(cmd, cwd=Path(__file__).parent)

    if result.returncode != 0:
        print(f"Error: Validation with {n_ranks} ranks failed")
        return False

    return True


def run_plotting():
    """Run plotting script to generate visualizations."""
    print(f"\n{'='*60}")
    print(f"Generating validation plots and visualizations")
    print(f"{'='*60}\n")

    cmd = ["uv", "run", "python", "plot_validation.py"]
    result = subprocess.run(cmd, cwd=Path(__file__).parent)

    return result.returncode == 0


if __name__ == '__main__':
    print("Solver Validation Suite")
    print("=" * 60)
    print("Tests spatial convergence O(N^-2) for all decomposition")
    print("and communication strategies")
    print("Testing with 4 and 8 ranks")
    print("=" * 60)

    # Run validation with different rank counts
    rank_counts = [4, 8]
    success_count = 0

    for n_ranks in rank_counts:
        if run_validation(n_ranks=n_ranks):
            success_count += 1
        else:
            print(f"Warning: Validation with {n_ranks} ranks failed")

    print(f"\n{'='*60}")
    print(f"Completed {success_count}/{len(rank_counts)} validation runs")
    print(f"{'='*60}")

    # Generate combined plots
    if success_count > 0:
        if run_plotting():
            print("\n✓ All validations complete and plots generated!")
        else:
            print("\n✗ Plotting failed")
            sys.exit(1)
    else:
        print("\n✗ All validations failed")
        sys.exit(1)

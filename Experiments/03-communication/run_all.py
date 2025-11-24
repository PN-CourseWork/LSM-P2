"""
Run All Communication Experiments
===================================

Helper script to run communication benchmarks across different rank counts.
"""
import subprocess
import sys
from pathlib import Path


def run_experiment(n_ranks):
    """Run communication benchmark with specified number of ranks.

    Parameters
    ----------
    n_ranks : int
        Number of MPI ranks to use
    """
    print(f"\n{'='*60}")
    print(f"Running with {n_ranks} ranks")
    print(f"{'='*60}\n")

    cmd = [
        "mpiexec",
        "-n", str(n_ranks),
        "uv", "run", "python",
        "compute_communication.py"
    ]

    result = subprocess.run(cmd, cwd=Path(__file__).parent)

    if result.returncode != 0:
        print(f"Error: Experiment with {n_ranks} ranks failed")
        return False

    return True


def run_plotting():
    """Run plotting script to generate visualizations."""
    print(f"\n{'='*60}")
    print(f"Generating plots")
    print(f"{'='*60}\n")

    cmd = ["uv", "run", "python", "plot_communication.py"]

    result = subprocess.run(cmd, cwd=Path(__file__).parent)

    return result.returncode == 0


if __name__ == '__main__':
    # Rank counts to test
    # 2, 4: sliced decomposition (contiguous planes, both methods work)
    # 8: cubic decomposition (non-contiguous, only NumPy tested)
    rank_counts = [2, 4, 8]

    print("Communication Benchmark Suite")
    print("=" * 60)
    print(f"Will run experiments with: {rank_counts} ranks")
    print(f"Sliced (2, 4 ranks): MPI datatypes vs NumPy arrays")
    print(f"Cubic (8 ranks): NumPy arrays only")
    print(f"Running 20 repetitions per configuration for statistics")
    print("=" * 60)

    # Run all experiments
    success_count = 0
    for n_ranks in rank_counts:
        if run_experiment(n_ranks):
            success_count += 1
        else:
            print(f"Warning: Skipping {n_ranks} ranks due to error")

    print(f"\n{'='*60}")
    print(f"Completed {success_count}/{len(rank_counts)} experiments")
    print(f"{'='*60}")

    # Generate plots
    if success_count > 0:
        if run_plotting():
            print("\n✓ All experiments complete and plots generated!")
        else:
            print("\n✗ Plotting failed")
            sys.exit(1)
    else:
        print("\n✗ No successful experiments to plot")
        sys.exit(1)

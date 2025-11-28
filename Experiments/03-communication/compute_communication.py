"""
Communication Method Benchmark
==============================

Compare NumPy vs Custom MPI datatype halo exchange for contiguous (Z-axis)
and non-contiguous (X-axis) memory layouts.

Uses per-iteration timeseries data for statistical analysis.
"""

import subprocess

from Poisson import get_project_root


def main():
    """Entry point - spawns MPI if needed."""
    try:
        from mpi4py import MPI

        if MPI.COMM_WORLD.Get_size() > 1:
            _run_benchmark()
            return
    except ImportError:
        pass

    # Spawn MPI
    script = (
        get_project_root()
        / "Experiments"
        / "03-communication"
        / "compute_communication.py"
    )
    subprocess.run(["mpiexec", "-n", "4", "uv", "run", "python", str(script)])


def _run_benchmark():
    """MPI worker - collects per-iteration timings."""
    import pandas as pd
    from mpi4py import MPI
    from Poisson import (
        JacobiPoisson,
        DomainDecomposition,
        NumpyHaloExchange,
        CustomHaloExchange,
        get_project_root,
    )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Config
    PROBLEM_SIZES = [32, 48, 64, 80, 100, 120, 140, 160, 180, 200]
    ITERATIONS = 500  # Per-iteration data gives us 500 samples each
    WARMUP = 50

    CONFIGS = [
        ("z", "numpy", "NumPy (Z-axis, contiguous)"),
        ("z", "custom", "Custom (Z-axis, contiguous)"),
        ("x", "numpy", "NumPy (X-axis, non-contiguous)"),
        ("x", "custom", "Custom (X-axis, non-contiguous)"),
    ]

    repo_root = get_project_root()
    data_dir = repo_root / "data" / "communication"
    data_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        print("Communication Benchmark: Per-Iteration Timings")
        print(f"Ranks: {size}, Iterations: {ITERATIONS}, Warmup: {WARMUP}")
        print("=" * 60)

    dfs = []

    for N in PROBLEM_SIZES:
        if rank == 0:
            print(f"\nN={N}")

        for axis, comm_type, label in CONFIGS:
            if rank == 0:
                print(f"  {label}...", end=" ", flush=True)

            # Create and run solver
            decomp = DomainDecomposition(N=N, size=size, strategy="sliced", axis=axis)
            halo = (
                CustomHaloExchange() if comm_type == "custom" else NumpyHaloExchange()
            )
            solver = JacobiPoisson(
                N=N,
                decomposition=decomp,
                communicator=halo,
                max_iter=WARMUP + ITERATIONS,
                tolerance=0,
            )
            solver.solve()

            # Get max halo time across ranks per iteration (skip warmup)
            local_times = solver.timeseries.halo_exchange_times[WARMUP:]
            max_times = comm.allreduce(local_times, op=MPI.MAX)

            if rank == 0:
                local_N = N // size  # Local subdomain size along decomposed axis
                print(f"mean={sum(max_times) / len(max_times) * 1e6:.1f} μs/iter")
                dfs.append(
                    pd.DataFrame(
                        {
                            "N": N,
                            "local_N": local_N,
                            "axis": axis,
                            "communicator": comm_type,
                            "label": label,
                            "iteration": range(len(max_times)),
                            "halo_time_us": [t * 1e6 for t in max_times],
                        }
                    )
                )

    if rank == 0:
        df = pd.concat(dfs, ignore_index=True)
        output = data_dir / f"communication_np{size}.parquet"
        df.to_parquet(output, index=False)
        print(f"\nSaved {len(df)} measurements to: {output}")


if __name__ == "__main__":
    main()

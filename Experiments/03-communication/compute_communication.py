"""
Communication Method Benchmark
================================

Unit test for halo exchange communicators.
"""
import numpy as np
import pandas as pd
import time
from pathlib import Path
from mpi4py import MPI

from Poisson import DomainDecomposition, NumpyHaloExchange, DatatypeCommunicator


def benchmark_communicator(communicator, u, rank_info, comm, n_iterations=100):
    """Benchmark a communicator implementation.

    Parameters
    ----------
    communicator : Communicator
        Communicator instance to benchmark
    u : np.ndarray
        Local array with ghost zones
    rank_info : RankInfo
        Decomposition information
    comm : MPI.Comm
        MPI communicator
    n_iterations : int
        Number of times to repeat exchange for timing accuracy

    Returns
    -------
    float
        Average time per exchange (seconds)
    """
    neighbors = rank_info.neighbors

    # Warmup
    for _ in range(10):
        communicator.exchange_halos(u, neighbors, comm)

    # Timed exchanges
    comm.Barrier()
    t_start = time.perf_counter()

    for _ in range(n_iterations):
        communicator.exchange_halos(u, neighbors, comm)

    comm.Barrier()
    t_end = time.perf_counter()

    return (t_end - t_start) / n_iterations


def run_benchmark(N, size, strategy, n_repetitions=100):
    """Run communication benchmark with multiple repetitions.

    Parameters
    ----------
    N : int
        Global grid size
    size : int
        Number of MPI ranks
    strategy : str
        Decomposition strategy
    n_repetitions : int
        Number of independent timing runs for statistics

    Returns
    -------
    list of dict
        Results for all ranks and repetitions
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_iterations = 50  # Repeat each timing run for accuracy

    # Setup decomposition
    decomp = DomainDecomposition(N=N, size=size, strategy=strategy)
    rank_info = decomp.get_rank_info(rank)

    # Allocate array
    u = np.zeros(rank_info.ghosted_shape, dtype=np.float64)

    # Create communicators
    numpy_comm = NumpyHaloExchange()
    datatype_comm = DatatypeCommunicator()

    # Run multiple independent repetitions
    all_local_results = []
    for rep in range(n_repetitions):
        # Reinitialize data each time
        u[:] = rank + np.random.rand(*rank_info.ghosted_shape)

        # Benchmark NumPy method (always)
        time_numpy = benchmark_communicator(numpy_comm, u.copy(), rank_info, comm, n_iterations)
        all_local_results.append(
            {'N': N, 'size': size, 'strategy': strategy, 'rank': rank, 'repetition': rep, 'method': 'numpy', 'time': time_numpy}
        )

        # Benchmark datatype method (both strategies now supported!)
        time_datatype = benchmark_communicator(datatype_comm, u.copy(), rank_info, comm, n_iterations)
        all_local_results.append(
            {'N': N, 'size': size, 'strategy': strategy, 'rank': rank, 'repetition': rep, 'method': 'datatype', 'time': time_datatype}
        )

    # Gather all results to rank 0
    all_results = comm.gather(all_local_results, root=0)

    if rank == 0:
        # Flatten list of lists
        return [item for sublist in all_results for item in sublist]
    return None


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Configuration 
    problem_sizes = [20, 40, 60, 80, 100, 120,  140]

    # Determine strategies to test based on size
    if size == 8:
        # For 8 ranks, test both strategies for comparison
        strategies = ['sliced', 'cubic']
    elif size in [2, 4]:
        strategies = ['sliced']
    else:
        if rank == 0:
            print(f"Warning: size={size} not configured, using cubic")
        strategies = ['cubic']

    # Run benchmarks for each strategy
    all_data = []
    for strategy in strategies:
        for N in problem_sizes:
            if rank == 0:
                print(f"Benchmarking N={N}, size={size}, strategy={strategy}...")

            results = run_benchmark(N, size, strategy)

            if rank == 0 and results is not None:
                all_data.extend(results)

                # Print summary
                df_subset = pd.DataFrame(results)
                numpy_mean = df_subset[df_subset['method'] == 'numpy']['time'].mean()

                # Check if datatype results exist
                datatype_results = df_subset[df_subset['method'] == 'datatype']
                if len(datatype_results) > 0:
                    datatype_mean = datatype_results['time'].mean()
                    print(f"  NumPy:     {numpy_mean*1e6:.2f} μs")
                    print(f"  Datatype:  {datatype_mean*1e6:.2f} μs")
                    print(f"  Speedup:   {numpy_mean/datatype_mean:.2f}x")
                else:
                    print(f"  NumPy:     {numpy_mean*1e6:.2f} μs")
                    print(f"  Datatype:  (not tested for {strategy})")

    # Save results as parquet
    if rank == 0:
        df = pd.DataFrame(all_data)

        repo_root = Path(__file__).resolve().parent.parent.parent
        data_dir = repo_root / "data" / "communication"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Use descriptive filename
        if len(strategies) > 1:
            output_file = data_dir / f"communication_size{size}_both.parquet"
        else:
            output_file = data_dir / f"communication_size{size}_{strategies[0]}.parquet"

        df.to_parquet(output_file, index=False)

        print(f"\nResults saved to: {output_file}")

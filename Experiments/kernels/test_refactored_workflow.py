"""Test the refactored HDF5 workflow.

This script tests:
1. Solver with new datastructures
2. Parallel HDF5 I/O
3. PostProcessor for loading and aggregating results
"""

from pathlib import Path
from mpi4py import MPI
from Poisson import JacobiPoisson, PostProcessor

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    print("=" * 60)
    print("Testing Refactored Workflow")
    print("=" * 60)

# Create output directory
output_dir = Path("test_output")
output_dir.mkdir(exist_ok=True)

# Test 1: Sequential solver
if rank == 0:
    print("\nTest 1: Sequential Solver")
    print("-" * 60)

if size == 1:
    solver = JacobiPoisson(N=25, omega=0.75, max_iter=100, use_numba=False)
    solver.solve()
    solver.summary()

    h5_file = output_dir / "test_sequential.h5"
    solver.save_hdf5(h5_file)

    if rank == 0:
        print(f"✓ Solver executed successfully")
        print(f"✓ Iterations: {solver.results.iterations}")
        print(f"✓ Converged: {solver.results.converged}")
        print(f"✓ Final error: {solver.results.final_error:.4e}")
        print(f"✓ Saved to: {h5_file}")
elif rank == 0:
    print("Skipping (requires single rank)")

# Test 2: Distributed solver
if rank == 0:
    print("\nTest 2: Distributed Solver")
    print("-" * 60)

if size > 1:
    solver = JacobiPoisson(
        decomposition="sliced",
        communicator="numpy",
        N=25,
        omega=0.75,
        max_iter=100
    )
    solver.solve()
    solver.summary()

    h5_file = output_dir / f"test_distributed_np{size}.h5"
    solver.save_hdf5(h5_file)

    if rank == 0:
        print(f"✓ Distributed solver executed (np={size})")
        print(f"✓ Iterations: {solver.results.iterations}")
        print(f"✓ Converged: {solver.results.converged}")
        print(f"✓ Final error: {solver.results.final_error:.4e}")
        print(f"✓ Saved to: {h5_file}")
elif rank == 0:
    print("Skipping (requires multiple ranks)")

# Test 3: PostProcessor (rank 0 only)
if rank == 0:
    print("\nTest 3: PostProcessor")
    print("-" * 60)

    # Find all HDF5 files
    h5_files = list(output_dir.glob("test_*.h5"))

    if h5_files:
        # Load with PostProcessor
        pp = PostProcessor(h5_files)

        # Test methods
        config = pp.get_config(0)
        print(f"✓ Loaded config: N={config['N']}, omega={config['omega']}")

        convergence = pp.get_convergence(0)
        print(f"✓ Convergence: {convergence['iterations']} iterations")

        timings = pp.aggregate_timings(0)
        print(f"✓ Aggregated timings: compute={timings['total_compute_time']:.4f}s")

        # Export to DataFrame
        df = pp.to_dataframe()
        print(f"✓ Converted to DataFrame: {len(df)} rows, {len(df.columns)} columns")
    else:
        print("⚠ No HDF5 files found to test PostProcessor")

if rank == 0:
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)

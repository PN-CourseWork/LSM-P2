"""MPI worker - invoked via: mpiexec -n X uv run python -m Poisson.helpers.runner_helper '{config}'"""

import sys
import json
from mpi4py import MPI
from Poisson import (
    JacobiPoisson,
    DomainDecomposition,
    NumpyHaloExchange,
    CustomHaloExchange,
)

config = json.loads(sys.argv[1])
comm = MPI.COMM_WORLD

decomp = DomainDecomposition(
    N=config["N"],
    size=comm.Get_size(),
    strategy=config.get("strategy", "sliced"),
    axis=config.get("axis", "z"),
)
halo = (
    CustomHaloExchange()
    if config.get("communicator") == "custom"
    else NumpyHaloExchange()
)

solver = JacobiPoisson(
    N=config["N"],
    decomposition=decomp,
    communicator=halo,
    max_iter=config.get("max_iter", 10000),
    tolerance=config.get("tol", 1e-6),
)

t0 = MPI.Wtime()
solver.solve()
wall_time = MPI.Wtime() - t0

# Store timing metrics in results (rank 0 only)
if comm.Get_rank() == 0:
    solver.results.wall_time = wall_time
    solver.results.total_compute_time = sum(solver.timeseries.compute_times)
    solver.results.total_halo_time = sum(solver.timeseries.halo_exchange_times)
    solver.results.total_mpi_comm_time = sum(solver.timeseries.mpi_comm_times)

# Compute L2 error if requested (not timed)
if config.get("validate"):
    solver.compute_l2_error()

# Save results to HDF5
output_path = config.get("output")
if output_path:
    solver.save_hdf5(output_path)

if comm.Get_rank() == 0:
    # Just print the path - runner.py will load the HDF5
    print(f"RESULT:{output_path}")

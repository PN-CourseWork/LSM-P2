"""MPI worker - invoked via: mpiexec -n X uv run python -m Poisson.helpers.runner_helper '{config}'"""

import sys
import json
from mpi4py import MPI
from Poisson import (
    JacobiPoisson,
    MultigridPoisson, # Import new solver
    DomainDecomposition,
    NumpyHaloExchange,
    CustomHaloExchange,
)

config = json.loads(sys.argv[1])
comm = MPI.COMM_WORLD

# Determine solver type
solver_type = config.get("solver_type", "jacobi") # Default to jacobi

if solver_type == "jacobi":
    decomp_N = config["N"]
    decomp = DomainDecomposition(
        N=decomp_N,
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
        use_numba=config.get("use_numba", False),
    )
elif solver_type == "multigrid":
    # MultigridPoisson handles its own decomposition internally
    communicator_choice = config.get("communicator", "numpy")
    solver = MultigridPoisson(
        N=config["N"],
        levels=config.get("levels"),  # auto-infer if None
        n_smooth=config.get("n_smooth", 3),
        omega=config.get("omega", 2.0 / 3.0),
        max_iter=config.get("max_iter", 20), # Multigrid converges faster
        tolerance=config.get("tol", 1e-6),
        use_numba=config.get("use_numba", False),
        decomposition_strategy=config.get("strategy", "sliced"),
        communicator=communicator_choice,
    )
elif solver_type == "fmg":
    communicator_choice = config.get("communicator", "numpy")
    solver = MultigridPoisson(
        N=config["N"],
        levels=config.get("levels"),  # auto-infer if None
        min_coarse_size=config.get("min_coarse_size", 9),
        n_smooth=config.get("n_smooth", 3),
        omega=config.get("omega", 2.0 / 3.0),
        fmg_post_cycles=config.get("fmg_post_cycles", 50),
        max_iter=config.get("max_iter", 20),
        tolerance=config.get("tol", 1e-6),
        use_numba=config.get("use_numba", False),
        decomposition_strategy=config.get("strategy", "sliced"),
        communicator=communicator_choice,
    )
else:
    raise ValueError(f"Unknown solver type: {solver_type}")


t0 = MPI.Wtime()
if solver_type == "fmg":
    solver.fmg_solve()
else:
    solver.solve()
wall_time = MPI.Wtime() - t0

# Store timing metrics in results (rank 0 only)
if comm.Get_rank() == 0:
    solver.results.wall_time = wall_time

# Compute L2 error if requested (not timed)
if config.get("validate"):
    # MultigridPoisson doesn't have compute_l2_error directly yet
    # We should add this method to MultigridPoisson
    if hasattr(solver, 'compute_l2_error'):
        solver.compute_l2_error()
    else:
        if comm.Get_rank() == 0:
            print("Warning: MultigridPoisson does not have compute_l2_error method yet.")

# Save results to HDF5
output_path = config.get("output")
if output_path:
    solver.save_hdf5(output_path)

if comm.Get_rank() == 0:
    # Just print the path - runner.py will load the HDF5
    print(f"RESULT:{output_path}")

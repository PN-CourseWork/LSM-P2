import math
from time import perf_counter as time
import numpy as np
from pathlib import Path
import pandas as pd
from mpi4py import MPI
from utils import datatools, cli
from Poisson import MPIJacobiSliced

# MPI imports

# Create the argument parser using shared utility
parser = cli.create_parser(
    methods=["sliced"],
    default_method="sliced",
    description="MPI sliced Poisson problem solver",
)

# Grab options!
options = parser.parse_args()
N: int = options.N
method: str = options.method
N_iter: int = options.iter
tolerance: float = options.tolerance

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

"""
In the below code, we'll have the axes aligned as z, y, x.
The MPIJacobiSliced solver decomposes along the Z-axis.
"""

# Create solver instance (MPI sliced decomposition)
# For numba acceleration, use: solver = MPIJacobiSliced(omega=0.75, use_numba=True)
solver = MPIJacobiSliced(omega=0.75)

# Set up the test problem
u1, u2, f, h = solver.setup_problem(N, initial_value=options.value0)

# Get the exact solution for validation
u_true = solver.get_exact_solution(N)

# Optional: warmup for numba (if use_numba=True)
# if rank == 0:
#     solver.warmup(N=10)

# Start MLflow logging (only on rank 0)
#if rank == 0:
#    solver.mlflow_start_log("/Shared/mpi_sliced_poisson_solver", N, N_iter, tolerance)

# Run the solver
solver.solve(u1, u2, f, h, N_iter, tolerance, u_true=u_true)

# End MLflow logging (only on rank 0)
#if rank == 0:
#    solver.mlflow_end_log()

# Only rank 0 prints summary and saves results
if rank == 0:
    solver.print_summary()
    data_dir = datatools.get_data_dir()
#    solver.save_results(data_dir, N, method, output_name=options.output)

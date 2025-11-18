import math
from time import perf_counter as time
import numpy as np
from pathlib import Path
import pandas as pd

from utils import datatools, cli
from Poisson import SequentialJacobi


# Create the argument parser using shared utility
parser = cli.create_parser(
    methods=["jacobi", "view"],  # "view" is alias for "jacobi"
    default_method="jacobi",
    description="Sequential Poisson problem solver",
)

# Grab options!
options = parser.parse_args()
N: int = options.N
method: str = options.method
N_iter: int = options.iter
tolerance: float = options.tolerance


"""
In the below code, we'll have the axes aligned as z, y, x.
"""

# Create solver instance (using pure numpy version)
# For numba acceleration, use: solver = SequentialJacobi(omega=0.75, use_numba=True)
solver = SequentialJacobi(omega=0.75)

# Set up the test problem
u1, u2, f, h = solver.setup_problem(N, initial_value=options.value0)

# Get the exact solution for validation
u_true = solver.get_exact_solution(N)

# Optional: warmup for numba (if use_numba=True)
# solver.warmup(N=10)

# Start MLflow logging
#solver.mlflow_start_log("/Shared/sequential_poisson_solver", N, N_iter, tolerance)

# Run the solver
solver.solve(u1, u2, f, h, N_iter, tolerance, u_true=u_true)

# End MLflow logging
#solver.mlflow_end_log()

# Print summary
solver.print_summary()

# Save results
#data_dir = datatools.get_data_dir()
#solver.save_results(data_dir, N, method, output_name=options.output)

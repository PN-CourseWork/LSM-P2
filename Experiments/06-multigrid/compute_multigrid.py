"""
Multigrid Convergence Experiment
================================

Runs the MultigridPoisson solver to demonstrate V-Cycle convergence.
"""

import time
from mpi4py import MPI
from Poisson import MultigridPoisson # Direct import for debugging

def main():
    # Configuration
    N = 65  # Power of 2 + 1 (64 + 1)
    levels = 4  # V-Cycle depth (65 -> 33 -> 17 -> 9)
    max_iter = 20
    tolerance = 1e-6
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"Running Multigrid Solver on N={N}^3 grid with {levels} levels.")
        print(f"MPI Size: {size}")
    
    # Instantiate Solver directly for debugging
    solver = MultigridPoisson(
        N=N,
        levels=levels,
        pre_smooth=2,
        post_smooth=2,
        max_iter=max_iter,
        tolerance=tolerance,
        use_numba=True,
        omega=0.8,
        decomposition_strategy='sliced'
    )
    
    start_time = time.time()
    solver.solve() # Call solve directly
    end_time = time.time()
    
    if rank == 0:
        print(f"Total Time: {end_time - start_time:.4f} seconds")
        # Access results directly for debugging convergence
        if solver.results.converged:
            print(f"Solver Converged in {solver.results.iterations} iterations.")
        else:
            print(f"Solver Did NOT Converge in {solver.results.iterations} iterations.")
        
        # Also print final error if available
        if solver.results.final_error is not None:
            print(f"Final L2 Error: {solver.results.final_error}")


if __name__ == "__main__":
    main()

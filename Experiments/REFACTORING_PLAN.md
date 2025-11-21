# Solver Unification Plan

## Overview

This document outlines the plan to unify all Poisson solver implementations under a common base class architecture. The goal is to enable fair performance comparisons between different decomposition strategies and MPI communication approaches while maintaining consistent bookkeeping, logging, and data management.

## Investigation goals 
### Benchmark Numba vs Numpy kernels 
- Different number of threads 

### Different types of domain decompositions
- Cubic 
- Sliced 
Note: How does the communication and computations scale here with problem size?
- plot communication and computation timings as a function of N (fixed number of ranks) 



### Different types of datastructures
- Numpy based
- Custom MPI data types
Note: Can we reduce some of the communication overhead when using Custom MPI data types? Can the code be cleaner and more readable?

### Analysis
- Strong Scaling
- Weak Scaling
Note: Can we relate the results from the previous questions to the results of the scaling plots?

## Target Architecture

### Solver Implementations (5 total)

1. **`SequentialJacobi`** 
   - Single-node baseline
   - No domain decomposition
   - Location: `src/Poisson/sequential.py`

2. **`MPIJacobiSliced`** 
   - 1D domain decomposition 
   - **Custom MPI datatypes** for communication
   - Location: `src/Poisson/mpi_sliced.py`

3. **`MPIJacobiSlicedNumpy`** 
   - 1D domain decomposition 
   - **Numpy contiguous arrays** (`ascontiguousarray`) for communication
   - Source: `Experiments/slice/slice.py`
   - Target: `src/Poisson/mpi_sliced_numpy.py`

4. **`MPIJacobiCubic`** 
   - 3D domain decomposition (Cartesian topology)
   - **Custom MPI datatypes** for face exchanges
   - Target: `src/Poisson/mpi_cubic.py`

5. **`MPIJacobiCubicNumpy`** 
   - 3D domain decomposition (Cartesian topology)
   - **Numpy contiguous arrays** (`ascontiguousarray`) for communication
   - Source: `Experiments/cubic/compute_cubic.py`
   - Target: `src/Poisson/mpi_cubic_numpy.py`

### Unified Base Class Pattern

All solvers extend `PoissonSolver` and follow this interface:

```python
# Initialization
solver = SolverClass(omega=0.75, use_numba=True, N=20, use_numba=False)

# Optional logging to MlFlow
solver.mlflow_start_log(experiment_name, N, max_iter, tolerance)

# Solve (no arguments - uses internal GlobalFields)
solver.solve(max_iter=1000, tolerance=1e-5)

# Results
solver.print_summary()
solver.save_results(data_dir)
solver.mlflow_end_log()
```

### Unified Data Structures

All solvers use:
- **`GlobalConfig`** - Runtime configuration (same across all ranks)
- **`GlobalFields`** - Problem definition (u1, u2, f, h, u_exact)
- **`LocalFields`** - Local domain data (MPI solvers only)
- **`GlobalResults`** - Aggregated results (iterations, convergence, timings)
- **`LocalResults`** - Per-rank performance data
- **`TimeSeriesGlobal`** - Time series data (compute_times, mpi_comm_times, residual_history)
- **`TimeSeriesLocal`** - Per-rank time series data



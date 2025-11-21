# Solver Unification Plan

## Overview

This document outlines the plan to unify all Poisson solver implementations under a common base class architecture with a **modular composition pattern**. The goal is to enable fair performance comparisons between different decomposition strategies and MPI communication approaches while maintaining consistent bookkeeping, logging, and data management.

The modular design allows us to independently study:
- **Kernels** (Numba vs Numpy)
- **Decomposition strategies** (Cubic vs Sliced)
- **Communication methods** (Custom MPI datatypes vs Numpy contiguous arrays)

## Investigation Goals

### Benchmark Numba vs Numpy kernels
- Different number of threads
- Performance comparison for same problem size
- Scalability with N

### Different types of domain decompositions
- **Cubic** (3D Cartesian decomposition)
- **Sliced** (1D decomposition along Z-axis)

**Key Questions**:
- How does communication and computation scale with problem size?
- Plot communication and computation timings as a function of N (fixed number of ranks)
- Which decomposition is more efficient for different problem sizes?

### Different types of communication methods
- **Numpy-based** (`ascontiguousarray`)
- **Custom MPI datatypes**

**Key Questions**:
- Can we reduce communication overhead with custom MPI datatypes?
- Is the code cleaner and more readable?
- What's the performance trade-off?

### Analysis
- **Strong Scaling**: Fixed problem size, increasing ranks
- **Weak Scaling**: Problem size grows with ranks (constant work per rank)

**Key Questions**:
- Can we relate decomposition/communication results to scaling behavior?
- Where are the bottlenecks?

## Target Architecture

### Modular Composition Pattern with Factory

We use a **Factory + Strategy pattern** where users specify strategies via strings, but internally the solver uses pluggable strategy objects:

```
SequentialJacobi (baseline)
MPIJacobi (modular with factory)
    ├── Factory creates strategies from strings
    ├── DecompositionStrategy (duck-typed)
    │   ├── SlicedDecomposition
    │   └── CubicDecomposition
    └── CommunicatorStrategy (duck-typed)
        ├── CustomMPICommunicator
        └── NumpyCommunicator
```

**Note**: We use **duck typing** instead of abstract base classes. Each strategy is a regular class that implements the expected interface (documented in docstrings).

### Solver Classes (2 total)

1. **`SequentialJacobi`** ✅
   - Single-node baseline
   - No domain decomposition
   - Location: `src/Poisson/sequential.py`

2. **`MPIJacobi`** (New modular class with factory)
   - Accepts string identifiers for decomposition and communicator
   - Creates strategy objects internally via factory methods
   - Location: `src/Poisson/mpi.py`

### Strategy Classes (4 total)

**Decomposition Strategies**:

3. **`SlicedDecomposition`**
   - 1D domain decomposition along Z-axis
   - Creates 1D Cartesian topology
   - Exchanges 2 ghost planes (top/bottom)
   - String identifier: `"sliced"`
   - Location: `src/Poisson/decomposition.py`

4. **`CubicDecomposition`**
   - 3D Cartesian decomposition
   - Creates 3D Cartesian topology
   - Exchanges 6 ghost faces (±X, ±Y, ±Z)
   - String identifier: `"cubic"`
   - Location: `src/Poisson/decomposition.py`

**Communicator Strategies**:

5. **`CustomMPICommunicator`**
   - Uses custom MPI datatypes (Create_contiguous, Create_subarray, etc.)
   - Pre-commits datatypes for efficiency
   - String identifier: `"custom-mpi"`
   - Location: `src/Poisson/communicator.py`

6. **`NumpyCommunicator`**
   - Uses `np.ascontiguousarray()` to create temporary buffers
   - Explicit memory copies before communication
   - String identifier: `"numpy"`
   - Location: `src/Poisson/communicator.py`

### Unified Interface

All solvers extend `PoissonSolver` and follow this interface:

```python
# Sequential baseline
solver = SequentialJacobi(omega=0.75, use_numba=True, N=100)

# MPI with sliced decomposition + custom MPI datatypes
solver = MPIJacobi(
    decomposition="sliced",
    communicator="custom-mpi",
    omega=0.75,
    use_numba=True,
    N=100
)

# MPI with cubic decomposition + numpy arrays
solver = MPIJacobi(
    decomposition="cubic",
    communicator="numpy",
    omega=0.75,
    use_numba=False,
    N=100
)

# Optional logging to MLflow
solver.mlflow_start_log(experiment_name)

# Solve
solver.solve(max_iter=1000, tolerance=1e-5)

# Results
solver.print_summary()
solver.save_results(data_dir)
solver.mlflow_end_log()
```

### Strategy String Identifiers

**Decomposition options**:
- `"sliced"` → `SlicedDecomposition`
- `"cubic"` → `CubicDecomposition`

**Communicator options**:
- `"custom-mpi"` → `CustomMPICommunicator`
- `"numpy"` → `NumpyCommunicator`

### 2×2 Combinations

The modular design gives us **4 MPI configurations**:

| Decomposition | Communicator | String Args | Method Name |
|--------------|--------------|-------------|-------------|
| Sliced | Custom MPI | `"sliced"`, `"custom-mpi"` | `mpi_sliced_custom` |
| Sliced | Numpy | `"sliced"`, `"numpy"` | `mpi_sliced_numpy` |
| Cubic | Custom MPI | `"cubic"`, `"custom-mpi"` | `mpi_cubic_custom` |
| Cubic | Numpy | `"cubic"`, `"numpy"` | `mpi_cubic_numpy` |

Plus the sequential baseline = **5 total configurations** from **6 classes** (instead of 5 separate solver classes).

### Unified Data Structures

All solvers use:
- **`GlobalConfig`** - Runtime configuration (same across all ranks)
- **`GlobalFields`** - Problem definition (u1, u2, f, h, u_exact)
- **`LocalFields`** - Local domain data (MPI solvers only)
- **`GlobalResults`** - Aggregated results (iterations, convergence, timings)
- **`LocalResults`** - Per-rank performance data
- **`TimeSeriesGlobal`** - Time series data (compute_times, mpi_comm_times, residual_history)
- **`TimeSeriesLocal`** - Per-rank time series data


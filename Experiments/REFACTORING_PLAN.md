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

### Modular Composition Pattern

Instead of creating 5 separate solver classes (which tightly couple decomposition + communication), we use a **strategy pattern** to make components pluggable:

```
SequentialJacobi (baseline)
MPIJacobi (modular)
    ├── DecompositionStrategy (pluggable)
    │   ├── SlicedDecomposition
    │   └── CubicDecomposition
    └── CommunicatorStrategy (pluggable)
        ├── MPIDatatypeCommunicator
        └── NumpyContiguousCommunicator
```

### Solver Classes (2 total)

1. **`SequentialJacobi`** ✅
   - Single-node baseline
   - No domain decomposition
   - Location: `src/Poisson/sequential.py`

2. **`MPIJacobi`** (New modular class)
   - Accepts decomposition and communicator strategies
   - Location: `src/Poisson/mpi.py`

### Strategy Classes (4 total)

**Decomposition Strategies**:

3. **`SlicedDecomposition`**
   - 1D domain decomposition along Z-axis
   - Creates 1D Cartesian topology
   - Exchanges 2 ghost planes (top/bottom)
   - Location: `src/Poisson/decomposition.py`

4. **`CubicDecomposition`**
   - 3D Cartesian decomposition
   - Creates 3D Cartesian topology
   - Exchanges 6 ghost faces (±X, ±Y, ±Z)
   - Location: `src/Poisson/decomposition.py`

**Communicator Strategies**:

5. **`MPIDatatypeCommunicator`**
   - Uses custom MPI datatypes (Create_contiguous, Create_subarray, etc.)
   - Pre-commits datatypes for efficiency
   - Location: `src/Poisson/communicator.py`

6. **`NumpyContiguousCommunicator`**
   - Uses `np.ascontiguousarray()` to create temporary buffers
   - Explicit memory copies before communication
   - Location: `src/Poisson/communicator.py`

### Unified Interface

All solvers extend `PoissonSolver` and follow this interface:

```python
# Sequential baseline
solver1 = SequentialJacobi(omega=0.75, use_numba=True, N=100)

# MPI with sliced decomposition + MPI datatypes
solver2 = MPIJacobi(
    decomposition="sliced",
    communicator="custom-mpi",
    omega=0.75,
    use_numba=True,
    N=100
)

# MPI with cubic decomposition + numpy arrays
solver3 = MPIJacobi(
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

### 2×2 Combinations

The modular design gives us **4 MPI configurations**:

| Decomposition | Communicator | Method Name |
|--------------|--------------|-------------|
| Sliced | MPI Datatypes | `mpi_sliced_custom` |
| Sliced | Numpy Contiguous | `mpi_sliced_numpy` |
| Cubic | MPI Datatypes | `mpi_cubic_custom` |
| Cubic | Numpy Contiguous | `mpi_cubic_numpy` |

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

## Implementation Plan

### Phase 1: Design Strategy Interfaces

**Define abstract base classes for strategies**

Location: `src/Poisson/strategies.py`

```python
from abc import ABC, abstractmethod

class DecompositionStrategy(ABC):
    @abstractmethod
    def decompose_domain(self, N, comm):
        """Returns local shape, global indices, neighbors."""
        pass

    @abstractmethod
    def setup_topology(self, comm):
        """Creates MPI Cartesian topology, returns cart comm and neighbors."""
        pass

    @abstractmethod
    def get_interior_slice(self):
        """Returns slice for interior points (excluding ghosts)."""
        pass

    @abstractmethod
    def gather_solution(self, u_local, N, comm):
        """Gathers distributed solution to rank 0."""
        pass

class CommunicatorStrategy(ABC):
    @abstractmethod
    def setup(self, local_shape, neighbors):
        """Initialize datatypes or buffers."""
        pass

    @abstractmethod
    def exchange_ghosts(self, u, cart, neighbors):
        """Exchange ghost layers with neighbors."""
        pass

    @abstractmethod
    def cleanup(self):
        """Free MPI datatypes if needed."""
        pass
```

### Phase 2: Implement Decomposition Strategies

**Location**: `src/Poisson/decomposition.py`

#### SlicedDecomposition
- Extract logic from `Experiments/slice/slice.py` and existing `mpi_sliced.py`
- 1D split along Z-axis
- Local shape: `(local_nz + 2, N, N)` (ghosts in Z direction)
- Neighbors: top and bottom ranks (from 1D Cartesian)
- Interior slice: `[1:-1, :, :]`

#### CubicDecomposition
- Extract logic from `Experiments/cubic/compute_cubic.py`
- 3D split using `MPI.Compute_dims(size, 3)`
- Local shape: `(local_nz + 2, local_ny + 2, local_nx + 2)`
- Neighbors: 6 faces (±X, ±Y, ±Z from 3D Cartesian)
- Interior slice: `[1:-1, 1:-1, 1:-1]`

### Phase 3: Implement Communicator Strategies

**Location**: `src/Poisson/communicator.py`

#### MPIDatatypeCommunicator
- `setup()`: Create and commit MPI datatypes for each face
  - For sliced: 2D plane (contiguous)
  - For cubic: YZ face (contiguous), XZ face (strided), XY face (contiguous)
- `exchange_ghosts()`: Use committed datatypes in Sendrecv calls
- `cleanup()`: Free committed datatypes

#### NumpyContiguousCommunicator
- `setup()`: Pre-allocate receive buffers (optional optimization)
- `exchange_ghosts()`:
  - Create contiguous send buffer with `np.ascontiguousarray(u[slice])`
  - Allocate receive buffer
  - Sendrecv
  - Copy received data back to ghost layer
- `cleanup()`: No-op

### Phase 4: Create Modular MPIJacobi Solver

**Location**: `src/Poisson/mpi.py`

```python
class MPIJacobi(PoissonSolver):
    def __init__(self, decomposition, communicator, **kwargs):
        super().__init__(**kwargs)
        self.decomposition = decomposition
        self.communicator = communicator

        # MPI setup
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Setup topology and decomposition
        self.cart, self.neighbors = self.decomposition.setup_topology(self.comm)

        # Set method name based on strategies
        decomp_name = self.decomposition.__class__.__name__.lower().replace('decomposition', '')
        comm_name = 'datatypes' if 'Datatype' in self.communicator.__class__.__name__ else 'numpy'
        self.config.method = f"mpi_{decomp_name}_{comm_name}"

    def method_solve(self):
        """Solve using MPI Jacobi iteration with pluggable strategies."""

        # Decompose domain
        local_shape, global_indices = self.decomposition.decompose_domain(
            self.config.N, self.cart
        )

        # Setup local arrays
        u1_local, u2_local, f_local = self._setup_local_arrays(
            local_shape, global_indices
        )

        # Initialize communicator
        self.communicator.setup(local_shape, self.neighbors)

        # Main iteration loop
        for i in range(self.config.max_iter):
            uold, u = (u1_local, u2_local) if i % 2 == 0 else (u2_local, u1_local)

            # Exchange ghosts using communicator strategy
            t_halo_start = MPI.Wtime()
            self.communicator.exchange_ghosts(uold, self.cart, self.neighbors)
            t_halo_end = MPI.Wtime()
            self.global_timeseries.halo_exchange_times.append(t_halo_end - t_halo_start)

            # Compute step using base class kernel
            t_comp_start = MPI.Wtime()
            interior = self.decomposition.get_interior_slice()
            local_residual = self._step(uold, u, f_local, self.global_fields.h, self.config.omega)
            t_comp_end = MPI.Wtime()
            self.global_timeseries.compute_times.append(t_comp_end - t_comp_start)

            # Global residual
            t_comm_start = MPI.Wtime()
            global_residual = self.comm.allreduce(local_residual**2, op=MPI.SUM)
            global_residual = np.sqrt(global_residual)
            t_comm_end = MPI.Wtime()
            self.global_timeseries.mpi_comm_times.append(t_comm_end - t_comm_start)
            self.global_timeseries.residual_history.append(float(global_residual))

            # Check convergence
            if global_residual < self.config.tolerance:
                self.global_results.converged = True
                self.global_results.iterations = i + 1
                break
        else:
            self.global_results.iterations = self.config.max_iter

        # Gather solution using decomposition strategy
        u_global = self.decomposition.gather_solution(u, self.config.N, self.comm)

        # Compute error on rank 0
        if self.rank == 0:
            self.global_results.final_error = float(
                np.linalg.norm(u_global - self.global_fields.u_exact)
            )
            self.global_fields.u1 = u_global

        # Cleanup
        self.communicator.cleanup()
```

### Phase 5: Refactor SequentialJacobi

**Current status**: Already follows base pattern ✅

**Minor updates**:
- Ensure `solve()` accepts `max_iter` and `tolerance` as arguments (not just from config)
- Match the interface from user's updated plan

### Phase 6: Update Experiments

Create experiment scripts that use the modular solvers:

```
Experiments/
├── sequential/
│   ├── compute_sequential.py          (uses SequentialJacobi)
│   └── plot_sequential.py
├── mpi_sliced_datatypes/
│   ├── compute_mpi_sliced_dt.py       (uses MPIJacobi + Sliced + Datatypes)
│   └── plot_mpi_sliced_dt.py
├── mpi_sliced_numpy/
│   ├── compute_mpi_sliced_np.py       (uses MPIJacobi + Sliced + Numpy)
│   └── plot_mpi_sliced_np.py
├── mpi_cubic_datatypes/
│   ├── compute_mpi_cubic_dt.py        (uses MPIJacobi + Cubic + Datatypes)
│   └── plot_mpi_cubic_dt.py
├── mpi_cubic_numpy/
│   ├── compute_mpi_cubic_np.py        (uses MPIJacobi + Cubic + Numpy)
│   └── plot_mpi_cubic_np.py
└── benchmarks/
    ├── compare_decompositions.py      (fixes comm, varies decomp)
    ├── compare_communicators.py       (fixes decomp, varies comm)
    ├── strong_scaling.py
    └── weak_scaling.py
```

**Example experiment script**:
```python
from Poisson import MPIJacobi
from Poisson.decomposition import CubicDecomposition
from Poisson.communicator import MPIDatatypeCommunicator

solver = MPIJacobi(
    decomposition=CubicDecomposition(),
    communicator=MPIDatatypeCommunicator(),
    omega=0.75,
    use_numba=True,
    N=100
)

solver.solve(max_iter=1000, tolerance=1e-5)
solver.print_summary()
solver.save_results(data_dir)
```

### Phase 7: Create Benchmark Utilities

**Location**: `Experiments/benchmarks/`

#### compare_decompositions.py
```python
# Fix communicator, vary decomposition
for decomp in [SlicedDecomposition(), CubicDecomposition()]:
    solver = MPIJacobi(decomp, MPIDatatypeCommunicator(), N=N, ...)
    solver.solve(...)
    # Collect timings
```

#### compare_communicators.py
```python
# Fix decomposition, vary communicator
for comm in [MPIDatatypeCommunicator(), NumpyContiguousCommunicator()]:
    solver = MPIJacobi(SlicedDecomposition(), comm, N=N, ...)
    solver.solve(...)
    # Collect timings
```

## Benefits of Modular Design

### 1. Independent Investigation
- **Compare decompositions**: Fix communicator, swap decomposition strategy
- **Compare communicators**: Fix decomposition, swap communicator strategy
- **Compare kernels**: Set `use_numba=True/False` for any combination

### 2. Code Reuse
- Single `MPIJacobi.method_solve()` works for all combinations
- No duplicated iteration logic
- Decomposition and communication concerns separated

### 3. Extensibility
Easy to add new strategies:
- **New decomposition**: 2D slab, pencil, block-cyclic
- **New communicator**: Non-blocking, one-sided RMA, neighborhood collectives
- Just implement the interface, plug it in

### 4. Clean Experiments
```python
# Benchmark all 4 MPI combinations
decompositions = [SlicedDecomposition(), CubicDecomposition()]
communicators = [MPIDatatypeCommunicator(), NumpyContiguousCommunicator()]

for decomp in decompositions:
    for comm in communicators:
        solver = MPIJacobi(decomp, comm, N=100, ...)
        # Run benchmark
```

### 5. Consistent Interface
- All configurations use same `solve()`, `print_summary()`, `save_results()`
- Same datastructures across all experiments
- Fair comparisons guaranteed

## Data Structure Consolidation

### Current Duplication
- Grid creation in both `problems.py` and `datastructures.py`
- Source term functions duplicated

### Resolution
- Move all to `datastructures.py`
- `GlobalFields.__post_init__()` handles initialization
- Delete or deprecate `problems.py`

## Testing Strategy

### Phase 1-3 Testing (Strategy classes)
- Unit test each strategy independently
- Verify decomposition produces correct local shapes
- Verify ghost exchange works correctly
- Test on 1, 2, 4, 8 ranks

### Phase 4 Testing (MPIJacobi)
- Test all 4 combinations (2 decomp × 2 comm)
- Verify convergence on known problem (sinusoidal)
- Compare solutions across all methods (should be identical within tolerance)
- Verify timing data populated correctly

### Integration Testing
- Run sequential and all 4 MPI configs on same problem
- Check iteration counts match (for same omega, tolerance)
- Check final errors are similar
- Verify save/load roundtrip works

## Success Criteria

- [ ] Abstract strategy interfaces defined
- [ ] 2 decomposition strategies implemented
- [ ] 2 communicator strategies implemented
- [ ] `MPIJacobi` uses composition pattern
- [ ] All 4 MPI combinations work correctly
- [ ] All use unified datastructures
- [ ] All produce same 4-file output format
- [ ] All integrate with MLflow
- [ ] Experiments use modular solvers
- [ ] Benchmark utilities created
- [ ] Can independently study decomposition, communication, and kernels

## Implementation Order

1. **Phase 1** - Design strategy interfaces (abstract base classes)
2. **Phase 2** - Implement decomposition strategies (Sliced, Cubic)
3. **Phase 3** - Implement communicator strategies (Datatypes, Numpy)
4. **Phase 4** - Create modular `MPIJacobi` solver
5. **Phase 5** - Minor updates to `SequentialJacobi` (interface consistency)
6. **Phase 6** - Update all experiments to use modular solvers
7. **Phase 7** - Create benchmark utilities for investigation goals

---

**Last Updated**: 2025-11-21
**Status**: Planning Complete - Ready for Implementation
**Architecture**: Modular Composition Pattern (Strategy Pattern)

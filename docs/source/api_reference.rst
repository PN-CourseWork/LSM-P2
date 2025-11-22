.. _api_reference:

=============
API Reference
=============

This page provides comprehensive documentation for the ``Poisson`` package API.

.. currentmodule:: Poisson

Architecture Overview
=====================

Design Philosophy
-----------------

The package implements a unified, **rank-symmetric** architecture for studying parallel performance of 3D Poisson equation solvers. The design prioritizes:

1. **Minimal data duplication**: Each rank stores only its local domain and results
2. **Scalable I/O**: Parallel HDF5 writes eliminate serial bottlenecks
3. **Pluggable strategies**: Duck-typed decomposition and communication strategies
4. **Clean abstractions**: Simple dataclasses without global/local redundancy

Unified Solver Design
----------------------

A single :class:`JacobiPoisson` solver handles both sequential and distributed execution through strategy injection::

   JacobiPoisson (unified solver)
       ├── DecompositionStrategy (pluggable)
       │   ├── NoDecomposition (sequential: entire domain on rank 0)
       │   ├── SlicedDecomposition (1D along Z-axis)
       │   └── CubicDecomposition (3D Cartesian grid)
       └── CommunicatorStrategy (pluggable)
           ├── NumpyCommunicator (array slicing + send/recv)
           └── CustomMPICommunicator (MPI datatypes)

**Key principle:** Sequential execution is just distributed execution with ``decomposition=None``.

.. note::
   We use **duck typing** instead of abstract base classes. Each strategy implements the expected interface documented in docstrings.

Common Interface
----------------

Simple, unified API for all execution modes:

.. code-block:: python

   from Poisson import JacobiPoisson

   # Sequential execution (no decomposition)
   solver = JacobiPoisson(N=100, omega=0.75, use_numba=True)
   solver.solve()
   solver.save_hdf5("results/sequential.h5")

   # Distributed execution with sliced decomposition
   # Run with: mpiexec -n 4 python script.py
   solver = JacobiPoisson(
       decomposition="sliced",
       communicator="numpy",
       N=200,
       omega=0.75
   )
   solver.solve()
   solver.save_hdf5("results/distributed.h5")  # Parallel write!

   # Access results 
   print(f"Converged: {solver.results.converged}")
   print(f"Iterations: {solver.results.iterations}")
   print(f"Error: {solver.results.final_error}")

Data Flow & Rank Symmetry
--------------------------

**Core principle:** Each rank is self-contained with local data only. No redundant global/local splits.

Simplified Datastructures
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Per-Solver (rank 0 only):**
- :class:`Results` - Convergence info (iterations, converged) 

**Per-Rank (all ranks):**

- :class:`Config` - Runtime parameters (N, omega, tolerance, method, MPI size, etc.)
- :class:`LocalFields` - Local domain arrays (u_local with ghosts, f_local)
- :class:`Timeseries` - timing arrays with results pr. iteration (compute_time, mpi_comm_time, halo_exchange_time)

Parallel I/O Strategy
^^^^^^^^^^^^^^^^^^^^^^

**HDF5 collective writes** eliminate the gather-to-rank-0 bottleneck:

.. code-block:: text

   Traditional (serial I/O):
   ┌─────────────┐
   │ Rank 0      │──┐
   │ Rank 1      │──┤  Gather    ┌─────────┐
   │ Rank 2      │──┼───────────▶│ Rank 0  │─────▶ HDF5
   │ Rank 3      │──┘            │ (serial)│
   └─────────────┘               └─────────┘
   Memory bottleneck on rank 0
   Serial write

   New (parallel I/O):
   ┌─────────────┐
   │ Rank 0      │─────┐
   │ Rank 1      │─────┤
   │ Rank 2      │─────┼───▶ HDF5 (parallel)
   │ Rank 3      │─────┘
   └─────────────┘
   Each rank writes its piece to the same file 
   Scales to large problems

**HDF5 File Structure:**

.. code-block:: text

   results.h5
   ├── /config                    # Scalars written by rank 0
   │   ├── N, omega, tolerance, ...
   ├── /fields                  # Collective write (N,N,N)
   │   └── [all ranks write their interior slices]
   ├── /results                   # Convergence info 
   │   ├── iterations, converged 
   └── /timings                   # Per-rank timing data
       ├── rank_0/compute_time, mpi_comm_time, ...
       ├── rank_1/...
       └── rank_N/...

See :ref:`data-structures` for detailed API documentation.

Solver Classes
==============

Unified Solver
--------------

.. autosummary::
   :toctree: generated
   :template: class.rst

   JacobiPoisson

The :class:`JacobiPoisson` solver handles both sequential (no decomposition) and distributed (with decomposition strategy) execution modes through a unified interface.

Decomposition Strategies
-------------------------

.. autosummary::
   :toctree: generated
   :template: class.rst

   NoDecomposition
   SlicedDecomposition
   CubicDecomposition

Communication Strategies
------------------------

.. autosummary::
   :toctree: generated
   :template: class.rst

   NumpyCommunicator
   CustomMPICommunicator

.. _data-structures:

Data Structures
===============

Configuration
-------------

.. autosummary::
   :toctree: generated
   :template: class.rst

   Config

**Stores:** N, omega, tolerance, max_iter, use_numba, method, num_threads, mpi_size

Fields
------

.. autosummary::
   :toctree: generated
   :template: class.rst

   LocalFields

**Stores:** Local domain arrays with ghost zones (u1_local, u2_local, f_local)

Results
-------

.. autosummary::
   :toctree: generated
   :template: class.rst

   Results
   TimingResults

**Results:** Convergence information (iterations, converged)

**TimingResults:** Per-rank timing summary (compute_time, mpi_comm_time, halo_exchange_time)

Time Series (Optional)
----------------------

.. autosummary::
   :toctree: generated
   :template: class.rst

   Timeseries

**Stores:** Detailed profiling data (residual_history, compute_times, mpi_comm_times, halo_exchange_times)

I/O Methods
===========

The solver provides multiple output formats for different use cases:

Parallel HDF5 
--------------------------------------------

.. code-block:: python

   solver.save_hdf5("results/experiment.h5")

**Features:**

- **Parallel writes**: Each rank writes its data concurrently
- **Single file**: All data (config, solution, results, timings) in one place
- **Hierarchical structure**: Organized groups for config/fields/results/timings
- **Compressed arrays**: Automatic compression for large datasets
- **Scalable**: No gather-to-rank-0 bottleneck

MLflow Logging
--------------

.. code-block:: python

   solver.log_to_mlflow("scaling-experiments")

**Features:**

- **Experiment tracking**: Automatic versioning and comparison
- **Web UI**: Browse and visualize results
- **Scalars only**: Cannot store arrays
- **Serial**: Rank 0 only


Computational Kernels
=====================

The package provides two implementations of the Jacobi iteration kernel:

.. autosummary::
   :toctree: generated

   jacobi_step_numpy
   jacobi_step_numba

:func:`jacobi_step_numpy` uses pure NumPy operations for portability and debugging.
:func:`jacobi_step_numba` uses Numba JIT compilation for high performance.

Problem Setup
=============

Utilities for creating test problems with known analytical solutions:

.. autosummary::
   :toctree: generated

   create_grid_3d
   sinusoidal_exact_solution
   sinusoidal_source_term
   setup_sinusoidal_problem

The :func:`setup_sinusoidal_problem` function creates a complete test problem with sinusoidal exact solution
and corresponding source term for verification and benchmarking.

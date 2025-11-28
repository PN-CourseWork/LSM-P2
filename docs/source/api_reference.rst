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
5. **Separation of concerns**: Solver handles computation, I/O, and MLflow logging

Unified Solver Design
----------------------

A single :class:`JacobiPoisson` solver handles both sequential and distributed execution through strategy injection::

   JacobiPoisson (unified solver)
       ├── DecompositionStrategy (pluggable)
       │   ├── NoDecomposition (sequential: entire domain on rank 0)
       │   └── DomainDecomposition (parallel)
       │       ├── strategy="sliced" (1D along Z-axis)
       │       └── strategy="cubic" (3D Cartesian grid)
       └── CommunicatorStrategy (pluggable)
           ├── NumpyHaloExchange (array slicing + send/recv)
           └── CustomHaloExchange (MPI datatypes)

**Key principle:** Sequential execution is just distributed execution with ``decomposition=None``.

.. note::
   We use **duck typing** instead of abstract base classes. Each strategy implements the expected interface documented in docstrings.

Common Workflow
---------------

**Phase 1: Solve + Write (with MPI)**

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
   solver.save_hdf5("results/distributed.h5")  # All ranks write in parallel!

**Phase 2: Run via Subprocess (easier MPI management)**

.. code-block:: python

   from Poisson import run_solver

   # Run solver with MPI via subprocess
   result = run_solver(
       N=100,
       n_ranks=4,
       strategy="sliced",
       communicator="numpy",
       max_iter=10000,
       tol=1e-6,
       validate=True  # Compute L2 error
   )

   print(f"Converged: {result['converged']}")
   print(f"Iterations: {result['iterations']}")
   print(f"L2 Error: {result['final_error']}")

Data Flow & Rank Symmetry
--------------------------

**Core principle:** Each rank is self-contained with local data only. No redundant global/local splits.

Simplified Datastructures
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Global (rank 0 only):**

- :class:`GlobalParams` - Runtime parameters (N, omega, tolerance, MPI size, etc.)
- :class:`GlobalMetrics` - Convergence info (iterations, converged, final_error)

**Per-Rank (all ranks):**

- :class:`LocalParams` - Rank-specific parameters (N_local, coordinates, kernel config)
- :class:`LocalFields` - Local domain arrays with halo zones (u1, u2, f)
- :class:`LocalSeries` - Per-iteration timing arrays (compute_times, halo_exchange_times, residual_history)

Experiment Tracking & I/O
-------------------------

The framework integrates with **MLflow** for experiment tracking and **HDF5** for data storage.

**MLflow Tracking:**

The solver automatically logs:
- **Parameters**: Grid size, tolerance, decomposition strategy, etc.
- **Metrics**: Convergence status, iterations, final error, wall time.
- **Time Series**: Complete history of residuals and performance metrics (compute time, comm time) per step.
- **Artifacts**: The HDF5 result file is uploaded to the MLflow run.

**Parallel HDF5 Strategy:**

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
   DomainDecomposition
   RankInfo

Communication Strategies
------------------------

.. autosummary::
   :toctree: generated
   :template: class.rst

   NumpyHaloExchange
   CustomHaloExchange

Experiment Utilities
====================

.. currentmodule:: utils

.. autosummary::
   :toctree: generated

   mlflow_io

The :mod:`utils.mlflow_io` module provides tools for fetching and managing experiment data from MLflow.

.. _data-structures:

Data Structures
===============

.. currentmodule:: Poisson

Global Configuration & Metrics
------------------------------

.. autosummary::
   :toctree: generated
   :template: class.rst

   GlobalParams
   GlobalMetrics

**GlobalParams:** N, omega, tolerance, max_iter, mpi_size, decomposition, communicator, use_numba

**GlobalMetrics:** iterations, converged, final_error, wall_time, timing breakdown

Local Data Structures
---------------------

.. autosummary::
   :toctree: generated
   :template: class.rst

   LocalParams
   LocalFields
   LocalSeries

**LocalParams:** Rank-specific parameters (N_local, local_start, local_end, kernel config)

**LocalFields:** Local domain arrays with halo zones (u1, u2, f)

**LocalSeries:** Per-iteration timing arrays (compute_times, mpi_comm_times, halo_exchange_times, residual_history)

Kernel Configuration
--------------------

.. autosummary::
   :toctree: generated
   :template: class.rst

   KernelParams
   KernelMetrics
   KernelSeries

**KernelParams:** N, omega, tolerance, max_iter, numba_threads

**KernelMetrics:** converged, iterations, final_residual, total_compute_time

**KernelSeries:** Per-iteration tracking (residuals, compute_times, physical_errors)

Solver I/O
==========

The solver writes complete simulation state to HDF5:

.. code-block:: python

   solver.save_hdf5("results/experiment.h5")

**Features:**

- ✅ **Parallel writes**: Each rank writes its data concurrently
- ✅ **Single file**: All data (config, fields, results, timings) in one place
- ✅ **Hierarchical structure**: Organized groups for config/fields/results/timings
- ✅ **Compressed arrays**: Automatic compression for large datasets
- ✅ **Scalable**: No gather-to-rank-0 bottleneck
- ✅ **Self-contained**: Everything needed to reproduce or analyze the run

Subprocess Runner
=================

.. autosummary::
   :toctree: generated

   run_solver

The :func:`run_solver` function provides a high-level API for running the solver via MPI subprocess:

.. code-block:: python

   from Poisson import run_solver

   # Run with 4 MPI ranks
   result = run_solver(N=64, n_ranks=4, strategy="sliced", communicator="numpy")
   print(f"Converged: {result['converged']}, Iterations: {result['iterations']}")


Computational Kernels
=====================

The package provides two implementations of the Jacobi iteration kernel through the :mod:`Poisson.kernels` module.
NumPy and Numba kernel implementations are available through the ``NumPyKernel`` and ``NumbaKernel`` classes.

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

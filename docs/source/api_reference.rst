.. _api_reference:

=============
API Reference
=============

This page provides an overview of the ``Poisson`` package API.

.. currentmodule:: Poisson

Architecture
============

Modular Composition Pattern
----------------------------

We use a **Factory + Strategy pattern** where users specify strategies via strings, but internally the solver uses pluggable strategy objects::

   SequentialJacobi (baseline)
   MPIJacobi (modular with factory)
       ├── Factory creates strategies from strings
       ├── DecompositionStrategy (duck-typed)
       │   ├── SlicedDecomposition
       │   └── CubicDecomposition
       └── CommunicatorStrategy (duck-typed)
           ├── CustomMPICommunicator
           └── NumpyCommunicator

.. note::
   We use **duck typing** instead of abstract base classes. Each strategy is a regular class that implements the expected interface (documented in docstrings).

Solver Classes
--------------

**SequentialJacobi**
   Single-node baseline solver with no domain decomposition.

   Location: ``src/Poisson/sequential.py``

**MPIJacobi**
   Modular MPI solver that accepts string identifiers for decomposition and communicator strategies.
   Creates strategy objects internally via factory methods.

   Location: ``src/Poisson/mpi.py``

Strategy Classes
----------------

**Decomposition Strategies:**

``SlicedDecomposition``
   * 1D domain decomposition along Z-axis
   * Creates 1D Cartesian topology
   * Exchanges 2 ghost planes (top/bottom)
   * String identifier: ``"sliced"``
   * Location: ``src/Poisson/decomposition.py``

``CubicDecomposition``
   * 3D Cartesian decomposition
   * Creates 3D Cartesian topology
   * Exchanges 6 ghost faces (±X, ±Y, ±Z)
   * String identifier: ``"cubic"``
   * Location: ``src/Poisson/decomposition.py``

**Communicator Strategies:**

``CustomMPICommunicator``
   * Uses custom MPI datatypes (``Create_contiguous``, ``Create_subarray``, etc.)
   * Pre-commits datatypes for efficiency
   * String identifier: ``"custom-mpi"``
   * Location: ``src/Poisson/communicator.py``

``NumpyCommunicator``
   * Uses ``np.ascontiguousarray()`` to create temporary buffers
   * Explicit memory copies before communication
   * String identifier: ``"numpy"``
   * Location: ``src/Poisson/communicator.py``

Unified Interface
-----------------

All solvers extend ``PoissonSolver`` and follow this interface:

.. code-block:: python

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

Solver Configurations
---------------------

The modular design gives us **4 MPI configurations** plus the sequential baseline:

.. list-table:: MPI Solver Combinations
   :header-rows: 1
   :widths: 20 20 30 30

   * - Decomposition
     - Communicator
     - String Args
     - Method Name
   * - Sliced
     - Custom MPI
     - ``"sliced"``, ``"custom-mpi"``
     - ``mpi_sliced_custom``
   * - Sliced
     - Numpy
     - ``"sliced"``, ``"numpy"``
     - ``mpi_sliced_numpy``
   * - Cubic
     - Custom MPI
     - ``"cubic"``, ``"custom-mpi"``
     - ``mpi_cubic_custom``
   * - Cubic
     - Numpy
     - ``"cubic"``, ``"numpy"``
     - ``mpi_cubic_numpy``

**Total: 5 solver configurations** from 6 modular classes (instead of 5 monolithic solver classes).

Unified Data Structures
------------------------

All solvers use consistent data structures for fair comparisons:

``GlobalConfig``
   Runtime configuration (same across all ranks): N, omega, max_iter, tolerance, use_numba, mpi_size, method

``GlobalFields``
   Problem definition: u1, u2, f, h, u_exact

``LocalFields``
   Local domain data (MPI solvers only): local array shapes and indices

``GlobalResults``
   Aggregated results: iterations, converged, final_error, wall_time, compute_time, mpi_comm_time, halo_exchange_time

``LocalResults``
   Per-rank performance data: mpi_rank, hostname, wall_time, compute_time, mpi_comm_time, halo_exchange_time

``TimeSeriesGlobal``
   Time series data: compute_times, mpi_comm_times, residual_history

``TimeSeriesLocal``
   Per-rank time series data: compute_times, mpi_comm_times, halo_exchange_times

Solvers
=======

Main solver classes for solving the 3D Poisson equation.

.. autosummary::
   :toctree: generated
   :nosignatures:

   PoissonSolver
   SequentialJacobi
   MPIJacobiSliced

Data Structures
===============

Data structures for solver configuration and results.

.. autosummary::
   :toctree: generated
   :nosignatures:

   GlobalConfig
   GlobalFields
   LocalFields
   GlobalResults
   LocalResults
   TimeSeriesGlobal
   TimeSeriesLocal

Kernels
=======

Jacobi iteration kernels.

.. autosummary::
   :toctree: generated
   :nosignatures:

   jacobi_step_numpy
   jacobi_step_numba

Problem Setup
=============

Functions for setting up test problems.

.. autosummary::
   :toctree: generated
   :nosignatures:

   create_grid_3d
   sinusoidal_exact_solution
   sinusoidal_source_term
   setup_sinusoidal_problem

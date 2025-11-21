3D Poisson Solver: MPI Performance Study
=========================================

A modular framework for studying parallel performance of 3D Poisson equation solvers using MPI domain decomposition.

**Authors:** Philip Nickel, DTU

This project implements and benchmarks different parallelization strategies for solving the 3D Poisson equation with Dirichlet boundary conditions using iterative Jacobi methods.

Overview
--------

This project unifies multiple Poisson solver implementations under a common base class architecture with a **modular composition pattern**. The goal is to enable fair performance comparisons between different decomposition strategies and MPI communication approaches while maintaining consistent bookkeeping, logging, and data management.

The modular design allows us to independently study:

* **Kernels** (Numba vs Numpy)
* **Decomposition strategies** (Cubic vs Sliced)
* **Communication methods** (Custom MPI datatypes vs Numpy contiguous arrays)

Investigation Goals
-------------------

Benchmark Numba vs Numpy Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Different number of threads
* Performance comparison for same problem size
* Scalability with problem size N

Different Types of Domain Decompositions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cubic** (3D Cartesian decomposition)
   3D domain decomposition that distributes the grid across all three spatial dimensions.

**Sliced** (1D decomposition along Z-axis)
   1D domain decomposition that splits only along the Z-axis, with each rank owning horizontal slices.

**Key Questions:**

* How does communication and computation scale with problem size?
* Plot communication and computation timings as a function of N (fixed number of ranks)
* Which decomposition is more efficient for different problem sizes?

Different Types of Communication Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Numpy-based** (``ascontiguousarray``)
   Uses explicit numpy array copies to create contiguous buffers before MPI communication.

**Custom MPI datatypes**
   Uses MPI's native datatype system (``Create_contiguous``, ``Create_subarray``) for zero-copy communication.

**Key Questions:**

* Can we reduce communication overhead with custom MPI datatypes?
* Is the code cleaner and more readable?
* What's the performance trade-off?

Scaling Analysis
~~~~~~~~~~~~~~~~

**Strong Scaling**
   Fixed problem size with increasing number of ranks. Measures parallel speedup.

**Weak Scaling**
   Problem size grows proportionally with ranks (constant work per rank). Measures parallel efficiency.

**Key Questions:**

* Can we relate decomposition/communication results to scaling behavior?
* Where are the bottlenecks?

Architecture
------------

Modular Composition Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~

**SequentialJacobi**
   Single-node baseline solver with no domain decomposition.

   Location: ``src/Poisson/sequential.py``

**MPIJacobi**
   Modular MPI solver that accepts string identifiers for decomposition and communicator strategies.
   Creates strategy objects internally via factory methods.

   Location: ``src/Poisson/mpi.py``

Strategy Classes
~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~

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

Experiments
-----------

The ``Experiments/`` directory contains organized performance studies:

Sequential Baseline
~~~~~~~~~~~~~~~~~~~

Location: ``Experiments/sequential/``

* ``compute_sequential.py`` - Run sequential solver
* ``plot_sequential.py`` - Visualize results

MPI Sliced Decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~

**With Custom MPI Datatypes**

Location: ``Experiments/mpi_sliced_custom/``

* ``compute_mpi_sliced_custom.py`` - Run with ``MPIJacobi("sliced", "custom-mpi")``
* ``plot_mpi_sliced_custom.py`` - Visualize results

**With Numpy Contiguous Arrays**

Location: ``Experiments/mpi_sliced_numpy/``

* ``compute_mpi_sliced_numpy.py`` - Run with ``MPIJacobi("sliced", "numpy")``
* ``plot_mpi_sliced_numpy.py`` - Visualize results

MPI Cubic Decomposition
~~~~~~~~~~~~~~~~~~~~~~~~

**With Custom MPI Datatypes**

Location: ``Experiments/mpi_cubic_custom/``

* ``compute_mpi_cubic_custom.py`` - Run with ``MPIJacobi("cubic", "custom-mpi")``
* ``plot_mpi_cubic_custom.py`` - Visualize results

**With Numpy Contiguous Arrays**

Location: ``Experiments/mpi_cubic_numpy/``

* ``compute_mpi_cubic_numpy.py`` - Run with ``MPIJacobi("cubic", "numpy")``
* ``plot_mpi_cubic_numpy.py`` - Visualize results

Benchmark Utilities
~~~~~~~~~~~~~~~~~~~

Location: ``Experiments/benchmarks/``

``compare_decompositions.py``
   Compare sliced vs cubic decomposition (fix communicator, vary decomposition)

``compare_communicators.py``
   Compare custom MPI vs numpy communicators (fix decomposition, vary communicator)

``strong_scaling.py``
   Fixed problem size, increasing ranks. Analyze speedup vs number of ranks.

``weak_scaling.py``
   Problem size grows with ranks (constant work per rank). Analyze efficiency vs number of ranks.

Example Usage
~~~~~~~~~~~~~

Run a specific experiment:

.. code-block:: bash

   # Run sliced decomposition with custom MPI datatypes
   mpiexec -n 4 uv run python Experiments/mpi_sliced_custom/compute_mpi_sliced_custom.py

Compare decomposition strategies:

.. code-block:: bash

   # Benchmark both decompositions
   mpiexec -n 8 uv run python Experiments/benchmarks/compare_decompositions.py

Strong scaling study:

.. code-block:: bash

   # Run on 1, 2, 4, 8 ranks
   for np in 1 2 4 8; do
       mpiexec -n $np uv run python Experiments/benchmarks/strong_scaling.py
   done

Contents
--------

:doc:`example_gallery/index`
   Gallery of example scripts demonstrating the use of the package.

:doc:`api_reference`
   Complete API reference for the ``Poisson`` package.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Examples

   example_gallery/index

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Reference

   api_reference

Installation
------------

The package requires Python 3.12+ and uses ``uv`` for dependency management::

   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv sync

MPI Setup
~~~~~~~~~

For MPI functionality, ensure you have an MPI implementation installed::

   # macOS with Homebrew
   brew install open-mpi

   # Ubuntu/Debian
   sudo apt-get install libopenmpi-dev openmpi-bin

Running Examples
~~~~~~~~~~~~~~~~

Run experiments using the main script::

   # Build documentation
   uv run python main.py --build-docs

   # Run all compute scripts
   uv run python main.py --compute

   # Run all plotting scripts
   uv run python main.py --plot

For the full codebase, please visit the `GitHub repository <https://github.com/PhilipNickel-DTU-CourseWork/LSM-P2>`_.

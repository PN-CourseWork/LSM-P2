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

The package implements a unified framework for studying parallel performance of 3D Poisson equation solvers. All solver variants share a common base class architecture with consistent data structures, enabling fair performance comparisons between different parallelization strategies.

**Current Implementation:**

.. code-block:: text

   PoissonSolver (base class)
   ├── SequentialJacobi (single-node baseline)
   └── MPIJacobiSliced (1D domain decomposition)

**Future Modular Design:**

The architecture is designed to support a **Factory + Strategy pattern** for pluggable decomposition and communication strategies::

   MPIJacobi (modular with factory)
       ├── DecompositionStrategy
       │   ├── SlicedDecomposition (1D along Z-axis) ✓ Implemented
       │   └── CubicDecomposition (3D Cartesian)
       └── CommunicatorStrategy
           ├── CustomMPICommunicator (MPI datatypes)
           └── NumpyCommunicator (explicit copies)

.. note::
   We use **duck typing** instead of abstract base classes. Each strategy implements the expected interface documented in docstrings.

Common Interface
----------------

All solvers extend :class:`PoissonSolver` and follow this unified interface:

.. code-block:: python

   from Poisson import SequentialJacobi, MPIJacobiSliced

   # Sequential baseline
   solver = SequentialJacobi(omega=0.75, use_numba=True, N=100)

   # MPI sliced decomposition (currently implemented)
   solver = MPIJacobiSliced(omega=0.75, use_numba=True, N=100)

   # Common workflow for all solvers
   solver.solve(max_iter=1000, tolerance=1e-5)
   solver.print_summary()
   solver.save_results("output/")

Data Flow
---------

All solvers use consistent data structures for configuration, fields, and results:

**Configuration & Problem Setup:**

- :class:`GlobalConfig` - Runtime parameters (N, omega, tolerance, etc.)
- :class:`GlobalFields` - Problem definition (solution arrays, source term, exact solution)
- :class:`LocalFields` - MPI-specific local domain information

**Results & Metrics:**

- :class:`GlobalResults` - Aggregated solver statistics (iterations, error, timings)
- :class:`LocalResults` - Per-rank performance metrics
- :class:`TimeSeriesGlobal` - Global time series data (residuals, compute times)
- :class:`TimeSeriesLocal` - Per-rank time series data

See :ref:`data-structures` for detailed documentation.

Solver Classes
==============

Base Class
----------

.. autosummary::
   :toctree: generated
   :template: class.rst

   PoissonSolver

Concrete Implementations
------------------------

.. autosummary::
   :toctree: generated
   :template: class.rst

   SequentialJacobi
   MPIJacobiSliced

The :class:`SequentialJacobi` solver provides a single-node baseline with no domain decomposition.
The :class:`MPIJacobiSliced` solver implements 1D domain decomposition along the Z-axis with ghost plane exchange.

.. _data-structures:

Data Structures
===============

Configuration
-------------

.. autosummary::
   :toctree: generated
   :template: class.rst

   GlobalConfig

Fields
------

.. autosummary::
   :toctree: generated
   :template: class.rst

   GlobalFields
   LocalFields

Results
-------

.. autosummary::
   :toctree: generated
   :template: class.rst

   GlobalResults
   LocalResults

Time Series
-----------

.. autosummary::
   :toctree: generated
   :template: class.rst

   TimeSeriesGlobal
   TimeSeriesLocal

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

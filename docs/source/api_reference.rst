.. _api_reference:

=============
API Reference
=============

This page provides comprehensive documentation for the ``Poisson`` package API.

.. currentmodule:: Poisson

Overview
========

The package provides a unified interface for solving 3D Poisson problems with MPI domain decomposition and multigrid acceleration. The current API centers on the Jacobi solver, multigrid driver, and the shared dataclasses used for IO and diagnostics.

Core Solvers
============

.. autosummary::
   :toctree: generated
   :template: class.rst

   JacobiPoisson
   MultigridPoisson

Helper Runners
==============

.. autosummary::
   :toctree: generated

   run_solver

MPI Grid Utilities
==================

.. autosummary::
   :toctree: generated
   :template: class.rst

   DistributedGrid
   RankGeometry

Kernel Implementations
======================

.. autosummary::
   :toctree: generated
   :template: class.rst

   NumPyKernel
   NumbaKernel

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

- **Parallel writes**: Each rank writes its data concurrently
- **Single file**: All data (config, fields, results, timings) in one place
- **Hierarchical structure**: Organized groups for config/fields/results/timings
- **Compressed arrays**: Automatic compression for large datasets
- **Scalable**: No gather-to-rank-0 bottleneck
- **Self-contained**: Everything needed to reproduce or analyze the run

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

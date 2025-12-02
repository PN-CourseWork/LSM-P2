.. _api_reference:

=============
API Reference
=============

This page provides comprehensive documentation for the ``Poisson`` package API.

.. currentmodule:: Poisson

Overview
========

The package provides a unified interface for solving 3D Poisson problems with MPI domain decomposition and multigrid acceleration. The API provides both sequential and MPI-parallel solvers using a clean inheritance pattern.

Solver Architecture
===================

The solver hierarchy uses a hook-based design where sequential solvers define the algorithm with no-op hooks, and MPI solvers override only the communication hooks:

.. mermaid::

   classDiagram
       class BaseSolver {
           <<abstract>>
           +N: int
           +omega: float
           +max_iter: int
           +tolerance: float
           +results: GlobalMetrics
           +timeseries: LocalSeries
           +solve()* GlobalMetrics
           +_compute_metrics()
       }

       class JacobiSolver {
           +kernel: NumPyKernel | NumbaKernel
           +u, u_old, f: ndarray
           +solve() GlobalMetrics
           +_get_time() float
           +_sync_halos() float
           +_apply_boundary_conditions()
           +_compute_residual() float
       }

       class JacobiMPISolver {
           +grid: DistributedGrid
           +comm: MPI.Comm
           +_sync_halos() float
           +_apply_boundary_conditions()
           +_compute_residual() float
       }

       class FMGSolver {
           +levels: List~GridLevel~
           +n_levels: int
           +solve() GlobalMetrics
           +fmg_solve() GlobalMetrics
           +_v_cycle() float
           +_get_time() float
           +_sync_halos() float
           +_apply_boundary_conditions()
           +_residual_norm() float
       }

       class FMGMPISolver {
           +comm: MPI.Comm
           +_sync_halos() float
           +_apply_boundary_conditions()
           +_residual_norm() float
       }

       BaseSolver <|-- JacobiSolver
       BaseSolver <|-- FMGSolver
       JacobiSolver <|-- JacobiMPISolver
       FMGSolver <|-- FMGMPISolver

**Key Design Patterns:**

- **Hook Methods**: Sequential solvers define ``_sync_halos()``, ``_apply_boundary_conditions()``, etc. as no-ops
- **MPI Overrides**: MPI solvers only override these hooks to add halo exchange and boundary handling
- **Algorithm Reuse**: ``solve()``, ``fmg_solve()``, and ``_v_cycle()`` are defined once in the sequential solver

Core Solvers
============

Sequential Solvers
------------------

.. autosummary::
   :toctree: generated
   :template: class.rst

   JacobiSolver
   FMGSolver

MPI-Parallel Solvers
--------------------

.. autosummary::
   :toctree: generated
   :template: class.rst

   JacobiMPISolver
   FMGMPISolver

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

Solver Metrics
--------------

.. autosummary::
   :toctree: generated
   :template: class.rst

   GlobalMetrics
   LocalSeries
   GridLevel

**GlobalMetrics:** iterations, converged, final_error, wall_time, timing breakdown

**LocalSeries:** Per-iteration timing arrays (compute_times, halo_exchange_times, residual_history)

**GridLevel:** One level in the multigrid hierarchy (N, h, u, u_temp, f, r, kernel, grid)

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

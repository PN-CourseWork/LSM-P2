.. _api_reference:

=============
API Reference
=============

This page provides an overview of the ``Poisson`` package API.

.. currentmodule:: Poisson

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

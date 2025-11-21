MPI Poisson Solver
==================

A modular framework for studying parallel performance of 3D Poisson equation solvers using MPI domain decomposition.

**Authors:**

* Alexander Elbæk Nielsen (s214724)
* Junriu Li (s242643)
* Philip Korsager Nickel (s214960)

**Institution:** Technical University of Denmark, DTU Compute

This project implements and benchmarks different parallelization strategies for solving the 3D Poisson equation with Dirichlet boundary conditions using iterative Jacobi methods.

Overview
--------

This project unifies multiple Poisson solver implementations under a common base class architecture with a **modular composition pattern**. The goal is to enable fair performance comparisons between different decomposition strategies and MPI communication approaches while maintaining consistent bookkeeping, logging, and data management.

The modular design allows us to independently study:

* **Kernels** (Numba vs Numpy)
* **Decomposition strategies** (Cubic vs Sliced)
* **Communication methods** (Custom MPI datatypes vs Numpy contiguous arrays)

.. include:: investigation_goals.rst

.. include:: architecture.rst

.. include:: experiments.rst

.. include:: installation.rst



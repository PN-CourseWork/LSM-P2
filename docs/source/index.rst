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

This project unifies multiple Poisson solver implementations under a common base class architecture with a **modular composition pattern**.
The modular design allows us to independently study:

* **Kernels** (Numba vs Numpy)
* **Decomposition strategies** (Cubic vs Sliced)
* **Communication methods** (Custom MPI datatypes vs Numpy arrays)

Investigation Goals
-------------------

Benchmark Numba vs Numpy Kernels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Different number of threads
* Performance comparison for same problem size
* Scalability with problem size N

Different Types of Domain Decompositions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Cubic** (3D Cartesian decomposition)
   3D domain decomposition that distributes the grid across all three spatial dimensions.

**Sliced** (1D decomposition along Z-axis)
   1D domain decomposition that splits only along the Z-axis, with each rank owning horizontal slices.

**Key Questions:**

* How does communication and computation scale with problem size?
* Plot communication and computation timings as a function of N (fixed number of ranks)
* Which decomposition is more efficient for different problem sizes?

Different Types of Communication Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Numpy-based** 
   Uses explicit numpy array copies to create contiguous buffers before MPI communication.

**Custom MPI datatypes**
   Uses MPI's native datatype system for zero-copy communication.

**Key Questions:**

* Can we reduce communication overhead with custom MPI datatypes?

Scaling Analysis
^^^^^^^^^^^^^^^^

**Strong Scaling**
   Fixed problem size with increasing number of ranks. Measures parallel speedup.

**Weak Scaling**
   Problem size grows proportionally with ranks (constant work per rank). Measures parallel efficiency.

**Key Questions:**

* Can we relate decomposition/communication results to scaling behavior?
* Where are the bottlenecks?

Installation
------------

The package requires Python 3.12+ and uses ``uv`` for dependency management::

   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv sync

For the full codebase, please visit the `GitHub repository <https://github.com/PhilipNickel-DTU-CourseWork/LSM-P2>`_.


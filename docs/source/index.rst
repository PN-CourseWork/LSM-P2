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
   3D domain decomposition using MPI Cartesian topology (``Create_cart`` + ``Cart_shift``) for optimal neighbor discovery.

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

Methodology
-----------

Hardware Specifications
^^^^^^^^^^^^^^^^^^^^^^^

All experiments were conducted on the DTU HPC cluster with the following specifications:

**Compute Nodes:**

* **Processors:** Intel Xeon E5-2650 v4 (Broadwell), 2.2 GHz, 24 cores/node (2 sockets × 12 cores)
* **Memory:** 128 GB DDR4 RAM per node
* **Interconnect:** InfiniBand FDR (56 Gb/s)

**Software Stack:**

* **MPI:** OpenMPI 4.x with core binding (``--bind-to core --map-by core``)
* **Python:** 3.12 with mpi4py, NumPy, Numba
* **Numba Threading:** Disabled for MPI runs (``NUMBA_NUM_THREADS=1``) to prevent thread oversubscription

**Performance Metrics:**

* **Mlup/s:** Million Lattice Updates per Second = (N³ × iterations) / (wall_time × 10⁶)
* **Strong Scaling:** Speedup S(P) = T(1)/T(P), Efficiency E(P) = S(P)/P
* **Weak Scaling:** Efficiency measured as T(1)/T(P) with constant work per rank

Installation
------------

The package requires Python 3.12+ and uses ``uv`` for dependency management::

   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv sync
   uv run setup_mlflow.py

For the full codebase, please visit the `GitHub repository <https://github.com/PhilipNickel-DTU-CourseWork/LSM-P2>`_.

.. toctree::
   :maxdepth: 1
   :caption: Experiments

   example_gallery/index

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api_reference

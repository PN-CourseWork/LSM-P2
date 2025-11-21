Investigation Goals
===================

Benchmark Numba vs Numpy Kernels
---------------------------------

* Different number of threads
* Performance comparison for same problem size
* Scalability with problem size N

Different Types of Domain Decompositions
-----------------------------------------

**Cubic** (3D Cartesian decomposition)
   3D domain decomposition that distributes the grid across all three spatial dimensions.

**Sliced** (1D decomposition along Z-axis)
   1D domain decomposition that splits only along the Z-axis, with each rank owning horizontal slices.

**Key Questions:**

* How does communication and computation scale with problem size?
* Plot communication and computation timings as a function of N (fixed number of ranks)
* Which decomposition is more efficient for different problem sizes?

Different Types of Communication Methods
-----------------------------------------

**Numpy-based** (``ascontiguousarray``)
   Uses explicit numpy array copies to create contiguous buffers before MPI communication.

**Custom MPI datatypes**
   Uses MPI's native datatype system (``Create_contiguous``, ``Create_subarray``) for zero-copy communication.

**Key Questions:**

* Can we reduce communication overhead with custom MPI datatypes?
* Is the code cleaner and more readable?
* What's the performance trade-off?

Scaling Analysis
----------------

**Strong Scaling**
   Fixed problem size with increasing number of ranks. Measures parallel speedup.

**Weak Scaling**
   Problem size grows proportionally with ranks (constant work per rank). Measures parallel efficiency.

**Key Questions:**

* Can we relate decomposition/communication results to scaling behavior?
* Where are the bottlenecks?

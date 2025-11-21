Experiments
===========

Computational experiments investigating parallel performance of 3D Poisson equation solvers.

These experiments systematically explore:

**Baseline Performance**
   Sequential solver metrics for comparison.

**Kernel Benchmarks**
   Numba JIT compilation vs pure NumPy implementations.

**Domain Decomposition**
   1D sliced vs 3D cubic decomposition strategies.

**Communication Methods**
   Custom MPI datatypes vs NumPy array communication.

**Scaling Analysis**
   Strong and weak scaling behavior across configurations.

Each experiment includes data generation scripts (``compute_*.py``) and visualization scripts (``plot_*.py``).
The plots below are automatically generated from the visualization scripts. 


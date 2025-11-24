04 - Solver Validation
======================

Description
-----------

Comprehensive end-to-end validation of the complete Poisson solver across all implementation permutations. This experiment tests **the fully assembled solver** - combining kernels from experiment 01, decomposition from experiment 02, and communication from experiment 03.

This is the **quality gate** that must pass before proceeding to performance analysis.

Purpose
-------

Establish absolute confidence in solver implementation by validating:

* **Solver correctness** - Verify all solver permutations produce consistent, accurate solutions
* **Analytical comparison** - Test against known exact solution: ``u(x,y,z) = sin(πx)sin(πy)sin(πz)``
* **Spatial convergence** - Demonstrate expected O(h²) convergence order as grid is refined
* **All configurations** - Test combinations of:

  - Kernel: NumPy vs Numba
  - Decomposition: Sliced vs Cubic
  - Communication: MPI datatypes vs NumPy arrays
  - Ranks: 1, 2, 4, 8, ... processors

* **Solution visualization** - Generate publication-quality plots for the report

**Quality Gate:** No experiments proceed to 05-scaling until validation passes.

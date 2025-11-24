04 - Solver Validation
======================

**Experiment Type:** Integration Testing (Quality Gate)

Overview
--------

Comprehensive end-to-end validation of the complete Poisson solver across all implementation permutations.

This experiment tests **the fully assembled solver** - combining kernels from experiment 01, decomposition from experiment 02, and communication from experiment 03. This is the **quality gate** that must pass before proceeding to performance analysis in experiment 05.

**Testing Approach:**
All components working together through complete Jacobi iteration cycles with convergence, comparing against known analytical solutions.

Objectives
----------

This integration test validates:

* **End-to-end solver correctness** - Verify the complete solver produces accurate solutions
* **Analytical solution comparison** - Test against known exact solution: ``u(x,y,z) = sin(πx)sin(πy)sin(πz)``
* **Spatial convergence** - Demonstrate expected O(h²) convergence order as grid is refined
* **Iterative convergence** - Verify residual reduction follows expected behavior
* **All permutations consistent** - Test all combinations of:

  * Kernel: NumPy vs Numba
  * Decomposition: Sliced vs Cubic
  * Communication: MPI datatypes vs NumPy arrays
  * Ranks: 1, 2, 4, 8, ... processors

* **Solution visualization** - Generate publication-quality plots for the report
* **Consistency across ranks** - Ensure parallel implementations agree with sequential results

Goal
----

**Establish absolute confidence** in solver implementation before performance analysis. All configurations must produce consistent, accurate results matching analytical solutions with theoretically expected convergence rates.

**Quality Gate:** No experiments proceed to 05-scaling until validation passes for all tested configurations.

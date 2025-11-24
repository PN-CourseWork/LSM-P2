01 - Kernel Benchmarks
======================

**Experiment Type:** Component Testing

Overview
--------

Compare Numba JIT-compiled kernels vs pure NumPy implementations to identify the optimal kernel configuration for subsequent experiments.

This experiment tests **only the computational kernel** in isolation - the 7-point stencil update function - without MPI, domain decomposition, or parallel communication. This allows us to understand pure computational performance before introducing distributed computing complexity.

Objectives
----------

This component-level benchmark evaluates:

* **Speedup from JIT compilation** - Performance gains from Numba's just-in-time compilation
* **Thread count performance** - Test how different thread configurations affect execution time
* **Problem size scaling** - Analyze kernel behavior as grid dimensions increase
* **Convergence characteristics** - Compare iteration counts and convergence rates
* **Kernel correctness** - Verify both implementations produce identical results

Goal
----

Select the best-performing kernel implementation and thread configuration to use in experiments 02-05.

**Decision Point:** Choose optimal kernel (NumPy or Numba) and thread count for all subsequent experiments. 

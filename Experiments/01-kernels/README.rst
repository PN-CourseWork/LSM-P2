01 - Kernel Benchmarks
======================

Description
-----------

Compare Numba JIT-compiled kernels vs pure NumPy implementations for the 7-point stencil update operation. This experiment tests **only the computational kernel** in isolation - without MPI, domain decomposition, or parallel communication - to understand pure computational performance before introducing distributed computing complexity.

Purpose
-------

Identify the optimal kernel configuration (NumPy vs Numba, thread count) for subsequent experiments by evaluating:

* **Performance with fixed iterations** - Compare execution time for NumPy vs Numba with different thread counts across various problem sizes
* **Performance with fixed tolerance** - Analyze convergence behavior and speedup as a function of problem size
* **Speedup analysis** - Generate plots comparing different Numba thread configurations against NumPy baseline
* **Kernel correctness** - Verify both implementations produce identical results

**Decision Point:** Choose optimal kernel (NumPy or Numba) and thread count for experiments 02-05.

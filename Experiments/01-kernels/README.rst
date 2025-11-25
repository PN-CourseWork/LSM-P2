01 - Choise of Kernel: Numpy vs Numba 
======================

Description
-----------

Here we compare Numba JIT-compiled kernels vs pure NumPy implementations.  This experiment tests **only the computational kernel** in isolation - without MPI, domain decomposition, or parallel communication.

Purpose
-------

Identify impacts of the choice of kernel implementations and parameters like thread-count. 

* **Kernel correctness** - Verify both implementations produce identical results
* **Performance pr. iteration** - Compare execution time for NumPy vs Numba with different thread counts across various problem sizes
* **Speedup analysis** - Comparing different Numba thread configurations against a NumPy baseline. 


* **Compute time scaling** - Measure computation cost with fixed iteration count and also fixed tolerance.

**Decision Point:** Choose optimal kernel (NumPy or Numba) and thread count for subsequent experiments.  

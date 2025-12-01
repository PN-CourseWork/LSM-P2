05 - Scaling Analysis
=====================

Description
-----------

Scaling analysis using the **validated solver configuration** from experiment 04. 
This experiment measures **the performance limits** of the complete solver across different problem sizes and processor counts.

**Strong Scaling:** Fixed problem size with increasing ranks → measures parallel speedup.

**Weak Scaling:** Constant work per rank with proportional growth → measures scalability.

Purpose
-------

Characterize the parallel performance limits of the validated solver by analyzing:

* **Strong scaling efficiency** - Measure speedup curves for fixed problem sizes with increasing processor counts
* **Weak scaling efficiency** - Evaluate performance with constant work per rank as both problem size and processors grow
* **Memory usage scaling** - Analyze per-rank memory footprint and total memory requirements as problem size and rank count vary
* **Parallel I/O considerations** - Demonstrate impact of parallel HDF5 writes vs serial gather-to-rank-0 on scaling behavior


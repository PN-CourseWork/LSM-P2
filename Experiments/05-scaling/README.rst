05 - Scaling Analysis
=====================

**Experiment Type:** Performance Testing

Overview
--------

Comprehensive parallel scaling analysis using the **validated solver configuration** from experiment 04 to evaluate MPI implementation performance characteristics and identify optimal configurations.

This experiment measures **the performance limits** of the complete, validated solver. All components have been tested individually (01-03) and verified end-to-end (04) - now we characterize how performance scales with problem size and processor count.

Strong Scaling
^^^^^^^^^^^^^^

Fixed problem size with increasing ranks to measure parallel speedup.

**Ideal behavior:** Time ∝ 1/ranks (linear speedup)
**Measures:** How well the solver parallelizes a fixed workload

Weak Scaling
^^^^^^^^^^^^

Constant work per rank while growing problem size and rank count proportionally.

**Ideal behavior:** Constant execution time (perfect efficiency)
**Measures:** Whether communication overhead grows as we scale up

Objectives
----------

This performance analysis characterizes:

* **Strong scaling efficiency** - Measure speedup curves for fixed problem sizes
* **Weak scaling efficiency** - Evaluate performance with constant work per rank
* **Parallel efficiency metrics** - Calculate efficiency percentages and identify drop-off points
* **Communication bottlenecks** - Quantify communication overhead as fraction of total time
* **Computation bottlenecks** - Identify when computation dominates vs when communication dominates
* **Time breakdowns** - Separate compute time, MPI time, and halo exchange time
* **Performance metrics** - Measure Mlup/s (million lattice updates per second)
* **Memory bandwidth** - Compare measured bandwidth to hardware specifications
* **Optimal configurations** - Determine best rank counts and problem sizes for efficiency
* **Limiting factors** - Identify what prevents perfect scaling (bandwidth, latency, load imbalance)

Goal
----

Characterize the parallel performance limits of the validated solver, identify scaling bottlenecks (computation vs communication), and provide quantitative guidance for selecting optimal problem sizes and rank counts for production runs.

**Prerequisites:** Experiments 01-04 must be complete with validated solver configuration.

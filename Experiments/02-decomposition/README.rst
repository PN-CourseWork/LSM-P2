02 - Domain Decomposition
==========================

**Experiment Type:** Component Testing

Overview
--------

Compare 1D sliced decomposition vs 3D cubic decomposition strategies for parallel domain partitioning.

This experiment tests **domain decomposition logic** - how the grid is partitioned across MPI ranks, how local indices map to global coordinates, and how ghost zones are structured. This is independent of the specific communication method used for ghost exchange.

**Sliced Decomposition (1D):**
Splits the domain along the Z-axis with each rank owning horizontal slices, exchanging 2 ghost planes.

**Cubic Decomposition (3D):**
Uses 3D Cartesian grid decomposition across all spatial dimensions, exchanging 6 ghost faces.

Objectives
----------

This component-level analysis evaluates:

* **Visual decomposition comparison** - Illustrate how the domain is partitioned for each method
* **Compute time scaling** - Measure computation cost with fixed iteration count
* **Communication time scaling** - Compare communication overhead between decomposition methods
* **Scaling order analysis** - Use log-log plots with reference lines to derive computational complexity
* **Load balance verification** - Ensure work is distributed evenly across ranks
* **Surface-area-to-volume ratios** - Analyze communication/computation trade-offs

Goal
----

Determine which decomposition strategy provides better performance for different problem sizes and rank counts. Understand the fundamental trade-offs between computation and communication costs.

**Decision Point:** Choose decomposition strategy (sliced or cubic) for experiments 04-05.

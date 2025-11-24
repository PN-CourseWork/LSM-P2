02 - Domain Decomposition
==========================

Description
-----------

Compare 1D sliced decomposition vs 3D cubic decomposition strategies for parallel domain partitioning. This experiment tests **domain decomposition logic** - how the grid is partitioned across MPI ranks, how local indices map to global coordinates, and how ghost zones are structured - independent of the specific communication method.

**Sliced (1D):** Splits domain along Z-axis with horizontal slices, exchanging 2 ghost planes.

**Cubic (3D):** Uses 3D Cartesian grid across all dimensions, exchanging 6 ghost faces.

Purpose
-------

Determine which decomposition strategy provides better performance for different problem sizes and rank counts by analyzing:

* **Visual comparison** - Illustrate how the domain is partitioned for each method
* **Compute time scaling** - Measure computation cost with fixed iteration count (optionally: fixed tolerance)
* **Communication time scaling** - Compare communication overhead between decomposition methods
* **Scaling order analysis** - Use log-log plots with reference lines to derive computational complexity
* **Surface-area-to-volume ratios** - Understand communication/computation trade-offs

**Decision Point:** Choose decomposition strategy (sliced or cubic) for experiments 04-05.

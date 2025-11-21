Domain Decomposition
====================

Compare different MPI domain decomposition strategies.

Investigation Goals
-------------------

1. **Sliced vs Cubic**: Compare 1D (sliced) vs 3D (cubic) decomposition
2. **Communication Overhead**: Measure ghost exchange costs
3. **Load Balance**: Verify equal work distribution
4. **Scaling**: How does each decomposition scale with problem size?

Decomposition Strategies
-------------------------

**Sliced Decomposition (1D)**
   Domain split along Z-axis only. Each rank owns horizontal slices.
   Exchanges 2 ghost planes (top/bottom).

**Cubic Decomposition (3D)**
   3D Cartesian grid decomposition across all spatial dimensions.
   Exchanges 6 ghost faces (±X, ±Y, ±Z).

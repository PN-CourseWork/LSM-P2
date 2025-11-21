Scaling Analysis
================

Comprehensive parallel scaling analysis for MPI implementations.

Investigation Goals
-------------------

1. **Strong Scaling**: Fixed problem size, increasing ranks (parallel speedup)
2. **Weak Scaling**: Growing problem size with ranks (parallel efficiency)
3. **Bottleneck Analysis**: Identify communication vs computation limits
4. **Optimal Configuration**: Find best rank count for given problem sizes

Scaling Definitions
-------------------

**Strong Scaling**
   Fixed total problem size N, increase number of MPI ranks.
   Ideal: Time ∝ 1/ranks (perfect speedup).

**Weak Scaling**
   Constant work per rank (N³/ranks = constant).
   Increase both N and ranks proportionally.
   Ideal: Time = constant (perfect efficiency).

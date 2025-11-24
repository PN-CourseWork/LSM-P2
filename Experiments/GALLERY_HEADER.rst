Experiments
===========

This section presents a systematic experimental analysis of the MPI-parallel Poisson solver, organized following software engineering best practices: **component testing → integration testing → performance testing**.

.. mermaid::

   flowchart TD
       A["01: Kernel Benchmarks<br/><small>NumPy vs Numba performance</small>"]
       B["02: Domain Decomposition<br/><small>1D sliced vs 3D cubic</small>"]
       C["03: Communication Methods<br/><small>MPI datatypes vs NumPy arrays</small>"]
       D["04: Solver Validation<br/><small>Quality Gate: End-to-end correctness</small>"]
       E["05: Scaling Analysis<br/><small>Strong & weak scaling studies</small>"]

       A --> B --> C --> D --> E



Experiments
===========

This section presents a systematic experimental analysis of the MPI-parallel Poisson solver, organized following software engineering best practices: **component testing → integration testing → performance testing**.

Experiment Overview
-------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Experiment
     - Description
   * - `01: Kernel Benchmarks <01-kernels/index.html>`_
     - Compare NumPy vs Numba JIT kernels for 7-point stencil operations without MPI.
   * - `02: Domain Decomposition <02-decomposition/index.html>`_
     - Compare 1D sliced vs 3D cubic domain partitioning strategies.
   * - `03: Communication Methods <03-communication/index.html>`_
     - Compare MPI datatypes vs NumPy arrays for ghost exchange.
   * - `04: Solver Validation <04-validation/index.html>`_
     - End-to-end correctness verification against analytical solutions. **Quality Gate.**
   * - `05: Scaling Analysis <05-scaling/index.html>`_
     - Strong and weak scaling analysis of the validated solver.



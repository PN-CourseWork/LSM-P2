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
   * - :doc:`01-kernels/index`
     - Compare NumPy vs Numba JIT kernels for 7-point stencil operations without MPI.
   * - :doc:`02-decomposition/index`
     - Compare 1D sliced vs 3D cubic domain partitioning strategies.
   * - :doc:`03-communication/index`
     - Compare MPI datatypes vs NumPy arrays for ghost exchange.
   * - :doc:`04-validation/index`
     - End-to-end correctness verification against analytical solutions. **Quality Gate.**
   * - :doc:`05-scaling/index`
     - Strong and weak scaling analysis of the validated solver.



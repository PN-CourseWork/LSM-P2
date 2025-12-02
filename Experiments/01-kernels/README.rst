01 - Choice of Kernel: NumPy vs Numba
======================================

Description
-----------

Here we compare Numba JIT-compiled kernels vs pure NumPy implementations.  This experiment tests **only the computational kernel** in isolation - without MPI, domain decomposition, or parallel communication.

Usage
-----

This experiment uses **Hydra** to manage parameters and **MLflow** to track individual runs.
Configuration files are located in ``conf/``.

**Single Run (Local):**
Run a specific configuration (defaults to ``benchmark.yaml``).

.. code-block:: bash

    uv run python Experiments/01-kernels/run_kernel.py N=100 kernel=numba threads=4

**Parameter Sweep (Multirun):**
Run the full benchmark sweep.

.. code-block:: bash

    # Benchmark Sweep (default config)
    uv run python Experiments/01-kernels/run_kernel.py --multirun

    # Convergence Check Sweep
    uv run python Experiments/01-kernels/run_kernel.py --config-name=convergence --multirun

**MLflow Configuration:**
By default, runs are logged to a local SQLite database (``mlflow.db``).
To log to Databricks, override the mode:

.. code-block:: bash

    uv run python Experiments/01-kernels/run_kernel.py --multirun mlflow.mode=databricks

Purpose
-------

Identify impacts of the choice of kernel implementations and parameters like thread-count. 

* **Kernel correctness** - Verify both implementations produce identical results
* **Performance pr. iteration** - Compare execution time for NumPy vs Numba with different thread counts across various problem sizes
* **Speedup analysis** - Comparing different Numba thread configurations against a NumPy baseline. 
* **Compute time scaling** - Measure computation cost with fixed iteration count and also fixed tolerance.

**Decision Point:** Choose optimal kernel (NumPy or Numba) and thread count for subsequent experiments.

Configuration
-------------

.. literalinclude:: ../hydra-conf/01-kernels.yaml
   :language: yaml
   :caption: 01-kernels.yaml

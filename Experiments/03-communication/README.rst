03 - Communication Methods
===========================

Description
-----------

Compare custom MPI datatypes vs NumPy array communication for halo exchange operations. 
This experiment tests **communication implementation details** - how data is transferred between ranks during halo exchanges.

**Custom MPI Datatypes:** Zero-copy communication using ``MPI.Create_contiguous()`` and ``MPI.Create_subarray()``.

**NumPy Arrays:** Explicit buffer copies using ``np.ascontiguousarray()``.

Purpose
-------

Determine whether custom MPI datatypes provide measurable performance improvements over NumPy arrays by evaluating:

* **Communication overhead** - Demonstrate whether custom datatypes reduce overhead compared to NumPy arrays
* **Scaling behavior** - Analyze how each method scales with problem size and rank count
* **Scaling order analysis** - Use log-log plots with reference lines to derive computational complexity

Usage
-----

.. code-block:: bash

    # Run communication benchmark (requires MPI)
    mpiexec -n 4 uv run python Experiments/03-communication/compute_communication.py

    # Plot results from MLflow
    uv run python Experiments/03-communication/plot_communication.py

Configuration
-------------

.. literalinclude:: ../hydra-conf/experiment/03-communication.yaml
   :language: yaml
   :caption: experiment/03-communication.yaml

06 - Scaling Analysis
=====================

Description
-----------

Scaling analysis using the **validated solver configuration** from experiment 04. 
This experiment measures **the performance limits** of the complete solver across different problem sizes and processor counts.

**Strong Scaling:** Fixed problem size with increasing ranks → measures parallel speedup.

**Weak Scaling:** Constant work per rank with proportional growth → measures scalability.

Usage
-----

This experiment uses **Hydra** for configuration and parameter sweeping.

**Local Execution:**
To run a specific experiment configuration locally:

.. code-block:: bash

    # Run the first case of the jacobi_strong_1node experiment
    uv run python Experiments/06-scaling/runner.py experiment=jacobi_strong_1node

**LSF Job Arrays:**
Submit job arrays that map `LSB_JOBINDEX` to parameter combinations defined in `conf/experiment/*.yaml`.

.. code-block:: bash

    # Example LSF script submission
    bsub -J "scaling[1-24]" < jobscript.lsf

    # Inside jobscript.lsf:
    mpiexec -n $LSB_DJOB_NUMPROC uv run python Experiments/06-scaling/runner.py experiment=jacobi_strong_1node

Purpose
-------

Characterize the parallel performance limits of the validated solver by analyzing:

* **Strong scaling efficiency** - Measure speedup curves for fixed problem sizes with increasing processor counts
* **Weak scaling efficiency** - Evaluate performance with constant work per rank as both problem size and processors grow
* **Memory usage scaling** - Analyze per-rank memory footprint and total memory requirements as problem size and rank count vary
* **Parallel I/O considerations** - Demonstrate impact of parallel HDF5 writes vs serial gather-to-rank-0 on scaling behavior

Configuration
-------------

.. literalinclude:: ../hydra-conf/06-scaling.yaml
   :language: yaml
   :caption: 06-scaling.yaml


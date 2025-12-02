02 - Domain Decomposition
==========================

Description
-----------

Compare 1D sliced decomposition and 3D cubic decomposition strategies for parallel domain partitioning. 
This experiment tests **domain decomposition logic** -  partitioning across MPI ranks, local to global indice mapping, and how ghost zones are structured.

**Sliced (1D):** Splits domain along Z-axis with horizontal slices, exchanging 2 ghost planes.

**Cubic (3D):** Uses 3D Cartesian grid across all dimensions, exchanging 6 ghost faces.

Purpose
-------

Determine which decomposition strategy provides better performance for different problem sizes and rank counts by analyzing:
Get an understanding of how the type of domain decomposition impacts the size of the data that needs to be communicated between ranks along
with the 'connectivity' of different ranks. 


* **Visual comparison** - Illustrate how the domain is partitioned for each method

* **Surface-area-to-volume ratios** - investigate how much data needs to be communicated between ranks depending on the decomposition strategy

Usage
-----

.. code-block:: bash

    # Generate domain decomposition visualizations
    uv run python Experiments/02-decomposition/plot_decompositions.py

Configuration
-------------

.. literalinclude:: ../hydra-conf/02-decomposition.yaml
   :language: yaml
   :caption: 02-decomposition.yaml 


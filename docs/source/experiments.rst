Experiments
===========

The ``Experiments/`` directory contains organized performance studies:

Sequential Baseline
-------------------

Location: ``Experiments/sequential/``

* ``compute_sequential.py`` - Run sequential solver
* ``plot_sequential.py`` - Visualize results

MPI Sliced Decomposition
-------------------------

**With Custom MPI Datatypes**

Location: ``Experiments/mpi_sliced_custom/``

* ``compute_mpi_sliced_custom.py`` - Run with ``MPIJacobi("sliced", "custom-mpi")``
* ``plot_mpi_sliced_custom.py`` - Visualize results

**With Numpy Contiguous Arrays**

Location: ``Experiments/mpi_sliced_numpy/``

* ``compute_mpi_sliced_numpy.py`` - Run with ``MPIJacobi("sliced", "numpy")``
* ``plot_mpi_sliced_numpy.py`` - Visualize results

MPI Cubic Decomposition
------------------------

**With Custom MPI Datatypes**

Location: ``Experiments/mpi_cubic_custom/``

* ``compute_mpi_cubic_custom.py`` - Run with ``MPIJacobi("cubic", "custom-mpi")``
* ``plot_mpi_cubic_custom.py`` - Visualize results

**With Numpy Contiguous Arrays**

Location: ``Experiments/mpi_cubic_numpy/``

* ``compute_mpi_cubic_numpy.py`` - Run with ``MPIJacobi("cubic", "numpy")``
* ``plot_mpi_cubic_numpy.py`` - Visualize results

Benchmark Utilities
-------------------

Location: ``Experiments/benchmarks/``

``compare_decompositions.py``
   Compare sliced vs cubic decomposition (fix communicator, vary decomposition)

``compare_communicators.py``
   Compare custom MPI vs numpy communicators (fix decomposition, vary communicator)

``strong_scaling.py``
   Fixed problem size, increasing ranks. Analyze speedup vs number of ranks.

``weak_scaling.py``
   Problem size grows with ranks (constant work per rank). Analyze efficiency vs number of ranks.

Example Usage
-------------

Run a specific experiment:

.. code-block:: bash

   # Run sliced decomposition with custom MPI datatypes
   mpiexec -n 4 uv run python Experiments/mpi_sliced_custom/compute_mpi_sliced_custom.py

Compare decomposition strategies:

.. code-block:: bash

   # Benchmark both decompositions
   mpiexec -n 8 uv run python Experiments/benchmarks/compare_decompositions.py

Strong scaling study:

.. code-block:: bash

   # Run on 1, 2, 4, 8 ranks
   for np in 1 2 4 8; do
       mpiexec -n $np uv run python Experiments/benchmarks/strong_scaling.py
   done

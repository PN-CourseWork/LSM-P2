04 - Solver Validation
======================

Description
-----------

End-to-end validation of the complete Poisson solver across all implementation permutations. 
This experiment tests **the fully assembled solver** - Decomposition from experiment 02, and communication from experiment 03.

The correctness is asserted by comparing the obtained solution with the analytical solution in a grid refinement study
and verifying the theoretical order of spatial accuracy.

.. note::
   We only use a single kernel-configuration here since the kernel correctness has already been established in experiment 01. 


Purpose
-------

Establish correctness of the solver implementation by:

* **Analytical comparison** - Test against known exact solution: ``u(x,y,z) = sin(πx)sin(πy)sin(πz)``
* **Spatial convergence** - Demonstrate expected O(h²) convergence order as grid is refined

Usage
-----

.. code-block:: bash

    # Run validation experiment
    uv run python run_solver.py --config-name=experiment/04-validation

    # Parameter sweep
    uv run python run_solver.py --config-name=experiment/04-validation \
        --multirun N=16,32,48 strategy=sliced,cubic

    # Plot convergence results
    uv run python Experiments/04-validation/plot_validation.py

Configuration
-------------

.. literalinclude:: ../hydra-conf/experiment/04-validation.yaml
   :language: yaml
   :caption: experiment/04-validation.yaml

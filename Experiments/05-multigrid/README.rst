05 - Multigrid Methods
======================

Study Full Multigrid (FMG) solver performance and spatial convergence.
Compare traversal patterns between Jacobi and FMG solvers.

Usage
-----

.. code-block:: bash

    # Run FMG spatial convergence study
    uv run python run_solver.py --config-name=05-multigrid-fmg

    # Parameter sweep
    uv run python run_solver.py --config-name=05-multigrid-fmg \
        --multirun N=65,129,257 strategy=sliced,cubic

    # Run timing comparison (Jacobi vs FMG)
    uv run python run_solver.py --config-name=05-multigrid-fmg \
        solver=jacobi max_iter=200
    uv run python run_solver.py --config-name=05-multigrid-fmg \
        solver=fmg

    # Plot results
    uv run python Experiments/05-multigrid/plot_multigrid_fmg.py
    uv run python Experiments/05-multigrid/plot_timings.py

Configuration
-------------

.. literalinclude:: ../hydra-conf/05-multigrid-fmg.yaml
   :language: yaml
   :caption: 05-multigrid-fmg.yaml

Scripts
-------

- ``plot_multigrid_fmg.py`` - Plot FMG spatial convergence from MLflow
- ``plot_timings.py`` - Plot timing comparison (Jacobi vs FMG traversal patterns)

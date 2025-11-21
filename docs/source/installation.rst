Installation
============

The package requires Python 3.12+ and uses ``uv`` for dependency management::

   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv sync

MPI Setup
---------

For MPI functionality, ensure you have an MPI implementation installed::

   # macOS with Homebrew
   brew install open-mpi

   # Ubuntu/Debian
   sudo apt-get install libopenmpi-dev openmpi-bin

Running Examples
----------------

Run experiments using the main script::

   # Build documentation
   uv run python main.py --docs

   # Run all plotting scripts
   uv run python main.py --plot

   # Copy plots to LaTeX report
   uv run python main.py --copy-plots

   # Clean all generated files
   uv run python main.py --clean

For the full codebase, please visit the `GitHub repository <https://github.com/PhilipNickel-DTU-CourseWork/LSM-P2>`_.

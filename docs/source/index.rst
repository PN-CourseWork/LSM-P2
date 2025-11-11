University Project Template
===========================

A template for numerical computing projects with documentation and examples.

**Authors:** Your Name

This documentation provides both example scripts and API reference for the numerical utilities package.

For the full codebase, please visit the `GitHub repository <https://github.com/yourusername/yourproject>`.

Contents
--------

:doc:`example_gallery/index`
   Gallery of example scripts demonstrating the use of the package.
:doc:`api_reference`
   Complete API reference for the ``numutils`` package.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Examples

   example_gallery/index

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Reference

   api_reference

Installation
------------

The package requires Python 3.12+ and uses ``uv`` for dependency management::

    curl -LsSf https://astral.sh/uv/install.sh | sh
    uv sync


Numerical Integration Example
=============================

This example demonstrates numerical integration using the trapezoidal
and Simpson's rules.

Scripts
-------

1. ``compute.py``: Compute integrals of test functions
2. ``plot_integration.py``: Visualize convergence rates

Results
-------

The example compares convergence rates of two methods:

- Trapezoidal rule: O(h²) convergence
- Simpson's rule: O(h⁴) convergence

Test functions
--------------

1. :math:`\int_0^1 x^2 \, dx = 1/3`
2. :math:`\int_0^\pi \sin(x) \, dx = 2`
3. :math:`\int_0^2 e^{-x} \, dx = 1 - e^{-2}`

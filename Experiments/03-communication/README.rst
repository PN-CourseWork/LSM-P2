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


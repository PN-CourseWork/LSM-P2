03 - Communication Methods
===========================

**Experiment Type:** Component Testing

Overview
--------

Compare custom MPI datatypes vs NumPy array communication for ghost zone exchange operations.

This experiment tests **communication implementation details** - how data is transferred between ranks during halo exchanges. This focuses purely on the data transfer mechanism, independent of the broader decomposition strategy or kernel implementation.

**Custom MPI Datatypes:**
Zero-copy communication using ``MPI.Create_contiguous()`` and ``MPI.Create_subarray()``, offering potentially better performance with more complex implementation.

**NumPy Arrays:**
Explicit buffer copies using ``np.ascontiguousarray()``, providing simpler, more Pythonic code with additional memory overhead.

Objectives
----------

This component-level benchmark evaluates:

* **Datatype creation overhead** - Measure cost of creating MPI datatypes vs array preparation
* **Transfer performance** - Compare actual communication time for ghost exchange
* **Communication scaling** - Analyze how each method scales with problem size and rank count
* **Overhead reduction** - Demonstrate whether custom datatypes reduce communication overhead
* **Data integrity** - Verify both methods transfer data correctly without corruption
* **Implementation complexity** - Document trade-offs between performance and code simplicity

Goal
----

Determine whether the added complexity of custom MPI datatypes provides measurable performance improvements over standard NumPy array communication for this application.

**Decision Point:** Choose communication method (MPI datatypes or NumPy arrays) for experiments 04-05.

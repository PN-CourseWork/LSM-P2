Communication Methods
=====================

Compare different MPI communication strategies for ghost exchange.

Investigation Goals
-------------------

1. **Custom MPI Datatypes vs NumPy Arrays**: Zero-copy vs explicit buffers
2. **Communication Overhead**: Measure datatype creation and transfer costs
3. **Code Clarity**: Evaluate readability and maintainability
4. **Performance Trade-offs**: When does each method excel?

Communication Strategies
------------------------

**Custom MPI Datatypes**
   Uses ``MPI.Create_contiguous()``, ``MPI.Create_subarray()``.
   Zero-copy communication (no temporary buffers).
   More complex code, but potentially faster.

**NumPy Arrays**
   Uses ``np.ascontiguousarray()`` for explicit copies.
   Simpler, more Pythonic code.
   Additional memory overhead.

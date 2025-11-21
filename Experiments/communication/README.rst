Communication Methods
=====================

Compare custom MPI datatypes vs NumPy array communication for ghost exchange. Custom MPI datatypes use zero-copy communication with ``MPI.Create_contiguous()`` and ``MPI.Create_subarray()``, offering potentially better performance with more complex code. NumPy arrays use ``np.ascontiguousarray()`` for explicit buffer copies, providing simpler, more Pythonic code with additional memory overhead. These experiments measure datatype creation costs, transfer overhead, and performance trade-offs.

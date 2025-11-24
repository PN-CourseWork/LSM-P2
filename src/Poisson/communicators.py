"""Communication implementations for halo exchanges.

Provides different strategies for exchanging ghost zones between MPI ranks.
"""
import numpy as np
from mpi4py import MPI


class NumpyCommunicator:
    """Halo exchange using NumPy arrays with explicit copies.

    Uses np.ascontiguousarray() to create contiguous buffers before sending.
    This introduces memory copy overhead but simplifies MPI calls.
    """

    def __init__(self):
        self.name = "NumPy"

    def exchange_halos(self, u, neighbors, comm):
        """Exchange ghost zones with neighboring ranks.

        Parameters
        ----------
        u : np.ndarray
            Local array with ghost zones (modified in-place)
        neighbors : dict
            Neighbor ranks: {'x_lower': rank, 'x_upper': rank, ...} or
                           {'z_lower': rank, 'z_upper': rank}
        comm : MPI.Comm
            MPI communicator
        """
        # Handle different decomposition strategies by checking which neighbors exist
        # Sliced: only z_lower/z_upper, shape is (nz, ny, nx), exchange along axis 0
        # Cubic: x/y/z neighbors, shape is (nx, ny, nz), exchange along all axes

        # Z direction (or first axis for sliced)
        if neighbors.get('z_lower') is not None:
            send_buf = np.ascontiguousarray(u[1, :, :])
            recv_buf = np.empty_like(send_buf)
            comm.Sendrecv(send_buf, dest=neighbors['z_lower'], sendtag=4,
                         recvbuf=recv_buf, source=neighbors['z_lower'], recvtag=5)
            u[0, :, :] = recv_buf

        if neighbors.get('z_upper') is not None:
            send_buf = np.ascontiguousarray(u[-2, :, :])
            recv_buf = np.empty_like(send_buf)
            comm.Sendrecv(send_buf, dest=neighbors['z_upper'], sendtag=5,
                         recvbuf=recv_buf, source=neighbors['z_upper'], recvtag=4)
            u[-1, :, :] = recv_buf

        # X direction (cubic only)
        if neighbors.get('x_lower') is not None:
            send_buf = np.ascontiguousarray(u[1, :, :])
            recv_buf = np.empty_like(send_buf)
            comm.Sendrecv(send_buf, dest=neighbors['x_lower'], sendtag=0,
                         recvbuf=recv_buf, source=neighbors['x_lower'], recvtag=1)
            u[0, :, :] = recv_buf

        if neighbors.get('x_upper') is not None:
            send_buf = np.ascontiguousarray(u[-2, :, :])
            recv_buf = np.empty_like(send_buf)
            comm.Sendrecv(send_buf, dest=neighbors['x_upper'], sendtag=1,
                         recvbuf=recv_buf, source=neighbors['x_upper'], recvtag=0)
            u[-1, :, :] = recv_buf

        # Y direction (cubic only)
        if neighbors.get('y_lower') is not None:
            send_buf = np.ascontiguousarray(u[:, 1, :])
            recv_buf = np.empty_like(send_buf)
            comm.Sendrecv(send_buf, dest=neighbors['y_lower'], sendtag=2,
                         recvbuf=recv_buf, source=neighbors['y_lower'], recvtag=3)
            u[:, 0, :] = recv_buf

        if neighbors.get('y_upper') is not None:
            send_buf = np.ascontiguousarray(u[:, -2, :])
            recv_buf = np.empty_like(send_buf)
            comm.Sendrecv(send_buf, dest=neighbors['y_upper'], sendtag=3,
                         recvbuf=recv_buf, source=neighbors['y_upper'], recvtag=2)
            u[:, -1, :] = recv_buf


class DatatypeCommunicator:
    """Halo exchange using custom MPI datatypes (zero-copy).

    Uses MPI.Create_subarray() to define non-contiguous data regions.
    Avoids memory copies by sending directly from the array.
    """

    def __init__(self):
        self.name = "MPI Datatype"
        self._datatypes = None
        self._shape = None

    def exchange_halos(self, u, neighbors, comm):
        """Exchange ghost zones with neighboring ranks.

        Parameters
        ----------
        u : np.ndarray
            Local array with ghost zones (modified in-place)
        neighbors : dict
            Neighbor ranks: {'x_lower': rank, 'x_upper': rank, ...} or
                           {'z_lower': rank, 'z_upper': rank}
        comm : MPI.Comm
            MPI communicator
        """
        # Create datatypes if needed
        if self._datatypes is None or self._shape != u.shape:
            if self._datatypes is not None:
                self._free_datatypes()
            self._datatypes = self._create_datatypes(u.shape)
            self._shape = u.shape

        n0, n1, n2 = u.shape

        # First axis (Z for sliced, X for cubic) - always contiguous
        if neighbors.get('z_lower') is not None:
            comm.Sendrecv([u[1, :, :], 1, self._datatypes['axis0']],
                         dest=neighbors['z_lower'], sendtag=4,
                         recvbuf=[u[0, :, :], 1, self._datatypes['axis0']],
                         source=neighbors['z_lower'], recvtag=5)

        if neighbors.get('z_upper') is not None:
            comm.Sendrecv([u[n0-2, :, :], 1, self._datatypes['axis0']],
                         dest=neighbors['z_upper'], sendtag=5,
                         recvbuf=[u[n0-1, :, :], 1, self._datatypes['axis0']],
                         source=neighbors['z_upper'], recvtag=4)

        # X direction (cubic only)
        if neighbors.get('x_lower') is not None:
            comm.Sendrecv([u[1, :, :], 1, self._datatypes['axis0']],
                         dest=neighbors['x_lower'], sendtag=0,
                         recvbuf=[u[0, :, :], 1, self._datatypes['axis0']],
                         source=neighbors['x_lower'], recvtag=1)

        if neighbors.get('x_upper') is not None:
            comm.Sendrecv([u[n0-2, :, :], 1, self._datatypes['axis0']],
                         dest=neighbors['x_upper'], sendtag=1,
                         recvbuf=[u[n0-1, :, :], 1, self._datatypes['axis0']],
                         source=neighbors['x_upper'], recvtag=0)

        # Y direction (cubic only) - use subarray datatype for zero-copy
        # Pass full array u, but datatypes describe which planes to send/recv
        if neighbors.get('y_lower') is not None:
            # Create datatypes describing Y-planes at different indices
            send_type = self._create_y_plane_type(u.shape, 1)
            recv_type = self._create_y_plane_type(u.shape, 0)

            comm.Sendrecv([u, 1, send_type], dest=neighbors['y_lower'], sendtag=2,
                         recvbuf=[u, 1, recv_type], source=neighbors['y_lower'], recvtag=3)

            send_type.Free()
            recv_type.Free()

        if neighbors.get('y_upper') is not None:
            send_type = self._create_y_plane_type(u.shape, n1-2)
            recv_type = self._create_y_plane_type(u.shape, n1-1)

            comm.Sendrecv([u, 1, send_type], dest=neighbors['y_upper'], sendtag=3,
                         recvbuf=[u, 1, recv_type], source=neighbors['y_upper'], recvtag=2)

            send_type.Free()
            recv_type.Free()

    def _create_datatypes(self, shape):
        """Create MPI datatypes for contiguous face directions.

        Parameters
        ----------
        shape : tuple
            Array shape (n0, n1, n2) including ghost zones

        Returns
        -------
        dict
            Datatypes for axis 0 (contiguous planes)
        """
        n0, n1, n2 = shape
        datatypes = {}

        # Axis 0 (first dimension) - contiguous in memory
        # u[i, :, :] is contiguous: n1*n2 elements
        plane_0 = MPI.DOUBLE.Create_contiguous(n1 * n2)
        plane_0.Commit()
        datatypes['axis0'] = plane_0

        return datatypes

    def _create_y_plane_type(self, shape, j_index):
        """Create MPI subarray datatype for Y-plane at specific index.

        This describes the non-contiguous memory pattern for u[:, j, :].

        Parameters
        ----------
        shape : tuple
            Array shape (n0, n1, n2)
        j_index : int
            Index along Y-axis to extract

        Returns
        -------
        MPI.Datatype
            Committed datatype (caller must Free() after use)
        """
        n0, n1, n2 = shape

        # Describe subarray: extract plane at [:, j_index, :]
        sizes = [n0, n1, n2]        # Full array dimensions
        subsizes = [n0, 1, n2]      # Extract one Y-plane
        starts = [0, j_index, 0]    # Starting position

        dtype = MPI.DOUBLE.Create_subarray(sizes, subsizes, starts, order=MPI.ORDER_C)
        dtype.Commit()
        return dtype

    def _free_datatypes(self):
        """Free allocated MPI datatypes."""
        if self._datatypes is not None:
            for dt in self._datatypes.values():
                dt.Free()
            self._datatypes = None

    def __del__(self):
        """Cleanup datatypes on destruction."""
        self._free_datatypes()

"""Communication implementations for halo exchanges.

Provides different strategies for exchanging ghost zones between MPI ranks.
"""
import numpy as np
from mpi4py import MPI


class _BaseCommunicator:
    """Base class for halo exchange communicators."""

    def exchange_ghosts(self, u, decomposition, rank, comm):
        """Adapter method for solver interface.

        Parameters
        ----------
        u : np.ndarray
            Local array with ghost zones
        decomposition : DomainDecomposition
            Decomposition strategy (has get_neighbors method)
        rank : int
            Current MPI rank
        comm : MPI.Comm
            MPI communicator
        """
        neighbors = decomposition.get_neighbors(rank)
        self.exchange_halos(u, neighbors, comm)


def _exchange_cubic_numpy(u, neighbors, comm):
    """Exchange ghost zones for cubic decomposition using NumPy arrays.

    Uses MPI PROC_NULL pattern: all ranks participate in all exchanges.
    Sendrecv to/from PROC_NULL is a no-op.
    At physical boundaries, set ghost to 0 (Dirichlet BC).
    """
    x_lo = neighbors.get('x_lower')
    x_hi = neighbors.get('x_upper')
    y_lo = neighbors.get('y_lower')
    y_hi = neighbors.get('y_upper')
    z_lo = neighbors.get('z_lower')
    z_hi = neighbors.get('z_upper')

    # Convert None to PROC_NULL for MPI
    PROC_NULL = MPI.PROC_NULL
    x_lo_rank = x_lo if x_lo is not None else PROC_NULL
    x_hi_rank = x_hi if x_hi is not None else PROC_NULL
    y_lo_rank = y_lo if y_lo is not None else PROC_NULL
    y_hi_rank = y_hi if y_hi is not None else PROC_NULL
    z_lo_rank = z_lo if z_lo is not None else PROC_NULL
    z_hi_rank = z_hi if z_hi is not None else PROC_NULL

    nz, ny, nx = u.shape[0] - 2, u.shape[1] - 2, u.shape[2] - 2

    # X direction: Send right (to x_upper), receive left (from x_lower)
    send_buf = np.ascontiguousarray(u[1:-1, 1:-1, -2])
    recv_buf = np.empty((nz, ny), dtype=u.dtype)
    comm.Sendrecv(send_buf, dest=x_hi_rank, sendtag=100,
                 recvbuf=recv_buf, source=x_lo_rank, recvtag=100)
    if x_lo is not None:
        u[1:-1, 1:-1, 0] = recv_buf
    else:
        u[1:-1, 1:-1, 0] = 0.0

    # X direction: Send left (to x_lower), receive right (from x_upper)
    send_buf = np.ascontiguousarray(u[1:-1, 1:-1, 1])
    recv_buf = np.empty((nz, ny), dtype=u.dtype)
    comm.Sendrecv(send_buf, dest=x_lo_rank, sendtag=101,
                 recvbuf=recv_buf, source=x_hi_rank, recvtag=101)
    if x_hi is not None:
        u[1:-1, 1:-1, -1] = recv_buf
    else:
        u[1:-1, 1:-1, -1] = 0.0

    # Y direction: Send up (to y_upper), receive down (from y_lower)
    send_buf = np.ascontiguousarray(u[1:-1, -2, 1:-1])
    recv_buf = np.empty((nz, nx), dtype=u.dtype)
    comm.Sendrecv(send_buf, dest=y_hi_rank, sendtag=200,
                 recvbuf=recv_buf, source=y_lo_rank, recvtag=200)
    if y_lo is not None:
        u[1:-1, 0, 1:-1] = recv_buf
    else:
        u[1:-1, 0, 1:-1] = 0.0

    # Y direction: Send down (to y_lower), receive up (from y_upper)
    send_buf = np.ascontiguousarray(u[1:-1, 1, 1:-1])
    recv_buf = np.empty((nz, nx), dtype=u.dtype)
    comm.Sendrecv(send_buf, dest=y_lo_rank, sendtag=201,
                 recvbuf=recv_buf, source=y_hi_rank, recvtag=201)
    if y_hi is not None:
        u[1:-1, -1, 1:-1] = recv_buf
    else:
        u[1:-1, -1, 1:-1] = 0.0

    # Z direction: Send up (to z_upper), receive down (from z_lower)
    send_buf = np.ascontiguousarray(u[-2, 1:-1, 1:-1])
    recv_buf = np.empty((ny, nx), dtype=u.dtype)
    comm.Sendrecv(send_buf, dest=z_hi_rank, sendtag=300,
                 recvbuf=recv_buf, source=z_lo_rank, recvtag=300)
    if z_lo is not None:
        u[0, 1:-1, 1:-1] = recv_buf
    else:
        u[0, 1:-1, 1:-1] = 0.0

    # Z direction: Send down (to z_lower), receive up (from z_upper)
    send_buf = np.ascontiguousarray(u[1, 1:-1, 1:-1])
    recv_buf = np.empty((ny, nx), dtype=u.dtype)
    comm.Sendrecv(send_buf, dest=z_lo_rank, sendtag=301,
                 recvbuf=recv_buf, source=z_hi_rank, recvtag=301)
    if z_hi is not None:
        u[-1, 1:-1, 1:-1] = recv_buf
    else:
        u[-1, 1:-1, 1:-1] = 0.0


class NumpyCommunicator(_BaseCommunicator):
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
            Neighbor ranks for sliced (z_lower/z_upper) or cubic (x/y/z)
        comm : MPI.Comm
            MPI communicator
        """
        # Detect decomposition type: cubic has x neighbors, sliced doesn't
        is_cubic = 'x_lower' in neighbors or 'x_upper' in neighbors

        if is_cubic:
            _exchange_cubic_numpy(u, neighbors, comm)
        else:
            # Sliced: shape is (nz, N, N) - z=axis0
            # Use PROC_NULL pattern for safety
            z_lo = neighbors.get('z_lower')
            z_hi = neighbors.get('z_upper')
            z_lo_rank = z_lo if z_lo is not None else MPI.PROC_NULL
            z_hi_rank = z_hi if z_hi is not None else MPI.PROC_NULL

            # Send up, receive down
            send_buf = np.ascontiguousarray(u[-2, :, :])
            recv_buf = np.empty_like(send_buf)
            comm.Sendrecv(send_buf, dest=z_hi_rank, sendtag=300,
                         recvbuf=recv_buf, source=z_lo_rank, recvtag=300)
            if z_lo is not None:
                u[0, :, :] = recv_buf

            # Send down, receive up
            send_buf = np.ascontiguousarray(u[1, :, :])
            recv_buf = np.empty_like(send_buf)
            comm.Sendrecv(send_buf, dest=z_lo_rank, sendtag=301,
                         recvbuf=recv_buf, source=z_hi_rank, recvtag=301)
            if z_hi is not None:
                u[-1, :, :] = recv_buf


class DatatypeCommunicator(_BaseCommunicator):
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
            Neighbor ranks for sliced (z_lower/z_upper) or cubic (x/y/z)
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
        is_cubic = 'x_lower' in neighbors or 'x_upper' in neighbors

        if is_cubic:
            # For cubic, use the numpy-based exchange (safe and correct)
            _exchange_cubic_numpy(u, neighbors, comm)
        else:
            # Sliced: use datatypes for efficient Z-direction exchange
            z_lo = neighbors.get('z_lower')
            z_hi = neighbors.get('z_upper')
            z_lo_rank = z_lo if z_lo is not None else MPI.PROC_NULL
            z_hi_rank = z_hi if z_hi is not None else MPI.PROC_NULL

            # Send up, receive down
            comm.Sendrecv([u[n0-2, :, :], 1, self._datatypes['axis0']],
                         dest=z_hi_rank, sendtag=300,
                         recvbuf=[u[0, :, :], 1, self._datatypes['axis0']],
                         source=z_lo_rank, recvtag=300)

            # Send down, receive up
            comm.Sendrecv([u[1, :, :], 1, self._datatypes['axis0']],
                         dest=z_lo_rank, sendtag=301,
                         recvbuf=[u[n0-1, :, :], 1, self._datatypes['axis0']],
                         source=z_hi_rank, recvtag=301)

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

    def _free_datatypes(self):
        """Free allocated MPI datatypes."""
        if self._datatypes is not None:
            for dt in self._datatypes.values():
                dt.Free()
            self._datatypes = None

    def __del__(self):
        """Cleanup datatypes on destruction."""
        self._free_datatypes()


# Aliases for backward compatibility
NumpyHaloExchange = NumpyCommunicator

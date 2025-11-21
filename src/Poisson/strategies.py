"""Strategy patterns for MPI domain decomposition and communication.

This module implements pluggable strategies for:
- Domain decomposition (sliced 1D, cubic 3D)
- Ghost exchange communication (MPI datatypes, NumPy arrays)

All strategies use duck typing - no abstract base classes.
Each strategy documents its expected interface in the class docstring.
"""

import numpy as np
from mpi4py import MPI


# ==============================================================================
# Decomposition Strategies
# ==============================================================================


class NoDecomposition:
    """No decomposition - entire domain on rank 0 (sequential execution).

    This is a special case where rank 0 owns the entire domain and there
    are no ghost exchanges. Used for sequential execution.
    """

    def initialize_local_arrays_distributed(self, N, rank, comm):
        """Initialize full global arrays on rank 0, empty on others.

        Parameters
        ----------
        N : int
            Global grid size
        rank : int
            MPI rank
        comm : MPI.Comm
            MPI communicator

        Returns
        -------
        u1_local, u2_local, f_local : np.ndarray
            Full global arrays on rank 0, empty on other ranks
        """
        if rank == 0:
            # Rank 0 gets the full global domain
            h = 2.0 / (N - 1)

            # Initialize full arrays
            u1 = np.zeros((N, N, N))
            u2 = np.zeros((N, N, N))
            f = np.zeros((N, N, N))

            # Create global coordinate arrays
            x = np.linspace(-1, 1, N)
            y = np.linspace(-1, 1, N)
            z = np.linspace(-1, 1, N)
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

            # Compute source term: f = 3π²sin(πx)sin(πy)sin(πz)
            f[1:-1, 1:-1, 1:-1] = (3 * np.pi**2 *
                                    np.sin(np.pi * X[1:-1, 1:-1, 1:-1]) *
                                    np.sin(np.pi * Y[1:-1, 1:-1, 1:-1]) *
                                    np.sin(np.pi * Z[1:-1, 1:-1, 1:-1]))

            # Boundary conditions are zero (already initialized)
            return u1, u2, f
        else:
            # Other ranks do nothing in sequential mode
            return None, None, None

    def extract_interior(self, u_local):
        """Extract interior points - for sequential, strip boundary."""
        return u_local[1:-1, 1:-1, 1:-1].copy()

    def get_interior_placement(self, rank_id, N, comm):
        """Entire interior goes to [1:-1, 1:-1, 1:-1]."""
        return (slice(1, N-1), slice(1, N-1), slice(1, N-1))

    def get_ghost_exchange_spec(self, u, rank, comm):
        """No ghost exchanges for sequential execution."""
        return []  # Empty list - no exchanges needed


class SlicedDecomposition:
    """1D domain decomposition along the Z-axis.

    Each rank owns horizontal slices and exchanges 2 ghost planes (top/bottom).

    Expected Interface:
    - decompose(N, rank, size) -> (local_N, z_start, z_end)
    - get_local_shape(N, local_N) -> tuple
    - get_neighbors(rank, size) -> (lower_neighbor, upper_neighbor)
    - setup_local_arrays(u1, u2, f, N, rank, size) -> (u1_local, u2_local, f_local)
    """

    def decompose(self, N, rank, size):
        """Decompose domain along Z-axis.

        Parameters
        ----------
        N : int
            Global grid size (including boundaries)
        rank : int
            MPI rank
        size : int
            MPI size

        Returns
        -------
        local_N : int
            Number of interior points owned by this rank
        z_start : int
            Starting Z index (inclusive)
        z_end : int
            Ending Z index (exclusive)
        """
        interior_N = N - 2
        base_size, remainder = divmod(interior_N, size)
        local_N = base_size + (1 if rank < remainder else 0)

        if rank < remainder:
            z_start = rank * (base_size + 1) + 1
        else:
            z_start = remainder * (base_size + 1) + (rank - remainder) * base_size + 1

        z_end = z_start + local_N
        return local_N, z_start, z_end

    def get_local_shape(self, N, local_N):
        """Get shape of local array including ghost zones.

        Parameters
        ----------
        N : int
            Global grid size
        local_N : int
            Number of interior points for this rank

        Returns
        -------
        shape : tuple
            (local_N + 2, N, N) with ghost zones
        """
        return (local_N + 2, N, N)

    def get_neighbors(self, rank, size):
        """Get neighbor ranks for ghost exchange.

        Parameters
        ----------
        rank : int
            MPI rank
        size : int
            MPI size

        Returns
        -------
        lower : int or None
            Lower neighbor rank (None if at boundary)
        upper : int or None
            Upper neighbor rank (None if at boundary)
        """
        lower = rank - 1 if rank > 0 else None
        upper = rank + 1 if rank < size - 1 else None
        return lower, upper

    def initialize_local_arrays_distributed(self, N, rank, comm):
        """Initialize local arrays directly from global coordinates (PETSc-style).

        This method creates local arrays without requiring global arrays,
        computing the source term and boundary conditions directly from
        global coordinates. This is memory-efficient and scales to large problems.

        Parameters
        ----------
        N : int
            Global grid size
        rank : int
            MPI rank
        comm : MPI.Comm
            MPI communicator

        Returns
        -------
        u1_local, u2_local, f_local : np.ndarray
            Local arrays with ghost zones, properly initialized
        """
        # Get local domain decomposition
        size = comm.Get_size()
        local_N, z_start, z_end = self.decompose(N, rank, size)
        local_shape = self.get_local_shape(N, local_N)

        # Initialize solution arrays as zeros (with ghost zones)
        u1_local = np.zeros(local_shape)
        u2_local = np.zeros(local_shape)
        f_local = np.zeros(local_shape)

        # Create coordinate arrays for interior points
        # Domain is [-1, 1]³ with grid spacing h = 2/(N-1)
        h = 2.0 / (N - 1)

        # Global grid coordinates
        x_global = np.linspace(-1, 1, N)
        y_global = np.linspace(-1, 1, N)
        z_global = np.linspace(-1, 1, N)

        # Extract local z coordinates (sliced in Z direction)
        z_local = z_global[z_start:z_end]

        # Create 3D meshgrid for local Z slices
        # For sliced decomposition: shape is (local_N, N, N) for interior
        # Need to broadcast correctly: Z varies in first dimension, X and Y in others
        Z_local, X_full, Y_full = np.meshgrid(z_local, x_global, y_global, indexing='ij')

        # Compute source term: f = 3π²sin(πx)sin(πy)sin(πz)
        # Interior region is [1:-1, :, :] for sliced decomposition
        f_local[1:-1, :, :] = (3 * np.pi**2 *
                                np.sin(np.pi * X_full) *
                                np.sin(np.pi * Y_full) *
                                np.sin(np.pi * Z_local))

        # Boundary conditions are all zero, so arrays are already correct
        # (initialized to zero). No explicit BC setting needed since u_bc = 0.

        return u1_local, u2_local, f_local

    def extract_interior(self, u_local):
        """Extract interior points from local array (without ghost zones).

        Parameters
        ----------
        u_local : np.ndarray
            Local array with ghost zones

        Returns
        -------
        interior : np.ndarray
            Interior points only
        """
        return u_local[1:-1, :, :].copy()

    def get_interior_placement(self, rank_id, N, comm):
        """Get global array indices where this rank's interior data belongs.

        Parameters
        ----------
        rank_id : int
            Rank ID
        N : int
            Global grid size
        comm : MPI.Comm
            MPI communicator

        Returns
        -------
        slices : tuple of slice objects
            Slices for indexing into global array
        """
        size = comm.Get_size()
        local_N, z_start, z_end = self.decompose(N, rank_id, size)
        return (slice(z_start, z_end), slice(0, N), slice(0, N))

    def get_ghost_exchange_spec(self, u, rank, comm):
        """Get specification for ghost zone exchanges.

        Parameters
        ----------
        u : np.ndarray
            Local array with ghost zones
        rank : int
            MPI rank
        comm : MPI.Comm
            MPI communicator

        Returns
        -------
        exchanges : list of dict
            Each dict specifies one exchange with keys:
            - 'neighbor': neighbor rank (or None)
            - 'send_slice': tuple of slices for data to send
            - 'recv_slice': tuple of slices for where to receive
            - 'tag_offset': integer tag offset for this exchange
        """
        size = comm.Get_size()
        lower, upper = self.get_neighbors(rank, size)

        exchanges = []

        # Lower neighbor exchange (send bottom, receive top ghost)
        if lower is not None:
            # Tag is based on min(rank, neighbor) to ensure both sides use same tag
            tag = min(rank, lower)
            exchanges.append({
                'neighbor': lower,
                'send_slice': (slice(1, 2), slice(None), slice(None)),
                'recv_slice': (slice(0, 1), slice(None), slice(None)),
                'tag': tag
            })

        # Upper neighbor exchange (send top, receive bottom ghost)
        if upper is not None:
            tag = min(rank, upper)
            exchanges.append({
                'neighbor': upper,
                'send_slice': (slice(-2, -1), slice(None), slice(None)),
                'recv_slice': (slice(-1, None), slice(None), slice(None)),
                'tag': tag
            })

        return exchanges


class CubicDecomposition:
    """3D Cartesian domain decomposition.

    Distributes the grid across all three dimensions, exchanging 6 ghost faces.
    Provides better load balance for large rank counts.

    Expected Interface:
    - decompose(N, rank, size) -> (local_Nx, local_Ny, local_Nz, starts, ends)
    - get_local_shape(N, local_dims) -> tuple
    - get_neighbors(rank, cart_comm) -> dict of neighbors
    - setup_local_arrays(u1, u2, f, N, rank, cart_comm) -> (u1_local, u2_local, f_local)
    """

    def __init__(self):
        self.cart_comm = None
        self.cart_coords = None
        self.dims = None

    def decompose(self, N, rank, comm):
        """Decompose domain into 3D Cartesian grid.

        Parameters
        ----------
        N : int
            Global grid size
        rank : int
            MPI rank
        comm : MPI.Comm
            MPI communicator

        Returns
        -------
        local_dims : tuple
            (local_Nx, local_Ny, local_Nz) interior points
        starts : tuple
            (x_start, y_start, z_start) indices
        ends : tuple
            (x_end, y_end, z_end) indices
        """
        size = comm.Get_size()

        # Create 3D Cartesian topology if not exists
        if self.cart_comm is None:
            # Factorize size into 3D grid (as cubic as possible)
            self.dims = MPI.Compute_dims(size, 3)
            self.cart_comm = comm.Create_cart(self.dims, periods=[False, False, False])
            self.cart_coords = self.cart_comm.Get_coords(rank)

        # Decompose each dimension
        interior_N = N - 2
        local_dims = []
        starts = []
        ends = []

        for dim_idx, (dim_size, coord) in enumerate(zip(self.dims, self.cart_coords)):
            base_size, remainder = divmod(interior_N, dim_size)
            local_size = base_size + (1 if coord < remainder else 0)

            if coord < remainder:
                start = coord * (base_size + 1) + 1
            else:
                start = remainder * (base_size + 1) + (coord - remainder) * base_size + 1

            end = start + local_size

            local_dims.append(local_size)
            starts.append(start)
            ends.append(end)

        return tuple(local_dims), tuple(starts), tuple(ends)

    def get_local_shape(self, N, local_dims):
        """Get shape of local array including ghost zones.

        Parameters
        ----------
        N : int
            Global grid size
        local_dims : tuple
            (local_Nx, local_Ny, local_Nz)

        Returns
        -------
        shape : tuple
            Local array shape with ghost zones
        """
        local_Nx, local_Ny, local_Nz = local_dims
        return (local_Nx + 2, local_Ny + 2, local_Nz + 2)

    def get_neighbors(self, rank, comm):
        """Get all 6 face neighbors.

        Parameters
        ----------
        rank : int
            MPI rank
        comm : MPI.Comm
            MPI communicator

        Returns
        -------
        neighbors : dict
            Keys: 'x_lower', 'x_upper', 'y_lower', 'y_upper', 'z_lower', 'z_upper'
            Values: neighbor rank or None
        """
        if self.cart_comm is None:
            raise RuntimeError("Must call decompose() first to create Cartesian topology")

        neighbors = {}
        directions = [('x_lower', 0, -1), ('x_upper', 0, 1),
                     ('y_lower', 1, -1), ('y_upper', 1, 1),
                     ('z_lower', 2, -1), ('z_upper', 2, 1)]

        for name, dim, direction in directions:
            source, dest = self.cart_comm.Shift(dim, direction)
            neighbors[name] = dest if dest != MPI.PROC_NULL else None

        return neighbors

    def get_ghost_exchange_spec(self, u, rank, comm):
        """Get specification for ghost zone exchanges.

        Parameters
        ----------
        u : np.ndarray
            Local array with ghost zones
        rank : int
            MPI rank
        comm : MPI.Comm
            MPI communicator

        Returns
        -------
        exchanges : list of dict
            Each dict specifies one exchange with keys:
            - 'neighbor': neighbor rank (or None)
            - 'send_slice': tuple of slices for data to send
            - 'recv_slice': tuple of slices for where to receive
            - 'tag_offset': integer tag offset for this exchange
        """
        neighbors = self.get_neighbors(rank, comm)

        # Define exchanges for all 6 faces
        face_specs = [
            ('x_lower', (slice(1, 2), slice(1, -1), slice(1, -1)),
                       (slice(0, 1), slice(1, -1), slice(1, -1))),
            ('x_upper', (slice(-2, -1), slice(1, -1), slice(1, -1)),
                       (slice(-1, None), slice(1, -1), slice(1, -1))),
            ('y_lower', (slice(1, -1), slice(1, 2), slice(1, -1)),
                       (slice(1, -1), slice(0, 1), slice(1, -1))),
            ('y_upper', (slice(1, -1), slice(-2, -1), slice(1, -1)),
                       (slice(1, -1), slice(-1, None), slice(1, -1))),
            ('z_lower', (slice(1, -1), slice(1, -1), slice(1, 2)),
                       (slice(1, -1), slice(1, -1), slice(0, 1))),
            ('z_upper', (slice(1, -1), slice(1, -1), slice(-2, -1)),
                       (slice(1, -1), slice(1, -1), slice(-1, None))),
        ]

        exchanges = []
        for face_name, send_slice, recv_slice in face_specs:
            neighbor = neighbors[face_name]
            if neighbor is not None:
                # Use min(rank, neighbor) to ensure matching tags
                tag = min(rank, neighbor)
                exchanges.append({
                    'neighbor': neighbor,
                    'send_slice': send_slice,
                    'recv_slice': recv_slice,
                    'tag': tag
                })

        return exchanges

    def initialize_local_arrays_distributed(self, N, rank, comm):
        """Initialize local arrays directly from global coordinates (PETSc-style).

        This method creates local arrays without requiring global arrays,
        computing the source term and boundary conditions directly from
        global coordinates. This is memory-efficient and scales to large problems.

        Parameters
        ----------
        N : int
            Global grid size
        rank : int
            MPI rank
        comm : MPI.Comm
            MPI communicator

        Returns
        -------
        u1_local, u2_local, f_local : np.ndarray
            Local arrays with ghost zones, properly initialized
        """
        # Get local domain decomposition
        local_dims, starts, ends = self.decompose(N, rank, comm)
        local_shape = self.get_local_shape(N, local_dims)

        x_start, y_start, z_start = starts
        x_end, y_end, z_end = ends

        # Initialize solution arrays as zeros (with ghost zones)
        u1_local = np.zeros(local_shape)
        u2_local = np.zeros(local_shape)
        f_local = np.zeros(local_shape)

        # Create coordinate arrays for interior points
        # Domain is [-1, 1]³ with grid spacing h = 2/(N-1)
        h = 2.0 / (N - 1)

        # Global grid coordinates
        x_global = np.linspace(-1, 1, N)
        y_global = np.linspace(-1, 1, N)
        z_global = np.linspace(-1, 1, N)

        # Extract local coordinates
        x_local = x_global[x_start:x_end]
        y_local = y_global[y_start:y_end]
        z_local = z_global[z_start:z_end]

        # Create 3D meshgrid for local interior
        X, Y, Z = np.meshgrid(x_local, y_local, z_local, indexing='ij')

        # Compute source term: f = 3π²sin(πx)sin(πy)sin(πz)
        f_local[1:-1, 1:-1, 1:-1] = (3 * np.pi**2 *
                                      np.sin(np.pi * X) *
                                      np.sin(np.pi * Y) *
                                      np.sin(np.pi * Z))

        # Set boundary conditions (u = 0 on boundaries)
        # Only set on ghost zones if this rank touches a global boundary
        coords = self.cart_coords
        dims = self.dims

        # Boundary conditions are all zero, so arrays are already correct
        # (initialized to zero). No explicit BC setting needed since u_bc = 0.

        return u1_local, u2_local, f_local

    def extract_interior(self, u_local):
        """Extract interior points from local array (without ghost zones).

        Parameters
        ----------
        u_local : np.ndarray
            Local array with ghost zones

        Returns
        -------
        interior : np.ndarray
            Interior points only
        """
        return u_local[1:-1, 1:-1, 1:-1].copy()

    def get_interior_placement(self, rank_id, N, comm):
        """Get global array indices where this rank's interior data belongs.

        Parameters
        ----------
        rank_id : int
            Rank ID
        N : int
            Global grid size
        comm : MPI.Comm
            MPI communicator (for getting Cartesian coordinates)

        Returns
        -------
        slices : tuple of slice objects
            Slices for indexing into global array
        """
        # Get coordinates for this rank
        rank_coords = self.cart_comm.Get_coords(rank_id)
        interior_N = N - 2

        # Compute global indices for this rank's coordinates
        starts = []
        ends = []
        for dim_idx, (dim_size, coord) in enumerate(zip(self.dims, rank_coords)):
            base_size, remainder = divmod(interior_N, dim_size)
            local_size = base_size + (1 if coord < remainder else 0)

            if coord < remainder:
                start = coord * (base_size + 1) + 1
            else:
                start = remainder * (base_size + 1) + (coord - remainder) * base_size + 1

            end = start + local_size
            starts.append(start)
            ends.append(end)

        x_start, y_start, z_start = starts
        x_end, y_end, z_end = ends

        return (slice(x_start, x_end), slice(y_start, y_end), slice(z_start, z_end))


# ==============================================================================
# Communicator Strategies
# ==============================================================================


class CustomMPICommunicator:
    """Ghost exchange using custom MPI datatypes (zero-copy).

    Uses MPI_Type_create_subarray for efficient communication without
    explicit buffer copies.

    Expected Interface:
    - exchange_ghosts(u, decomposition, rank, comm) -> None (in-place)
    """

    def __init__(self):
        self.datatypes = {}

    def exchange_ghosts(self, u, decomposition, rank, comm):
        """Generic ghost exchange using decomposition's exchange specification.

        Parameters
        ----------
        u : np.ndarray
            Local array with ghost zones (modified in-place)
        decomposition : DecompositionStrategy
            Decomposition strategy providing exchange spec
        rank : int
            MPI rank
        comm : MPI.Comm
            MPI communicator
        """
        # Get exchange specification from decomposition
        exchanges = decomposition.get_ghost_exchange_spec(u, rank, comm)

        # Prepare all send/receive buffers
        send_bufs = []
        recv_bufs = []

        for exchange in exchanges:
            send_slice = exchange['send_slice']
            recv_slice = exchange['recv_slice']

            send_bufs.append(u[send_slice].copy())
            recv_bufs.append(np.empty_like(u[recv_slice]))

        # Post all receives first
        recv_reqs = []
        for i, exchange in enumerate(exchanges):
            req = comm.Irecv(recv_bufs[i], source=exchange['neighbor'], tag=exchange['tag'])
            recv_reqs.append(req)

        # Post all sends
        send_reqs = []
        for i, exchange in enumerate(exchanges):
            req = comm.Isend(send_bufs[i], dest=exchange['neighbor'], tag=exchange['tag'])
            send_reqs.append(req)

        # Wait for all communications to complete
        MPI.Request.Waitall(recv_reqs + send_reqs)

        # Copy received data into ghost zones
        for i, exchange in enumerate(exchanges):
            u[exchange['recv_slice']] = recv_bufs[i]

    def _create_plane_datatype(self, N):
        """Create MPI datatype for 2D plane (sliced decomposition)."""
        key = ('plane', N)
        if key not in self.datatypes:
            dtype = MPI.DOUBLE.Create_contiguous(N * N)
            dtype.Commit()
            self.datatypes[key] = dtype
        return self.datatypes[key]

    def _create_face_datatype(self, shape, face):
        """Create MPI datatype for 3D face (cubic decomposition)."""
        # For cubic decomposition - implement subarray types
        # This is a placeholder for now
        key = ('face', shape, face)
        if key not in self.datatypes:
            # TODO: Implement proper subarray types for each face
            pass
        return self.datatypes.get(key)

    def exchange_ghosts_sliced(self, u, decomposition, rank, size, comm):
        """Exchange ghost planes for sliced decomposition.

        Parameters
        ----------
        u : np.ndarray
            Local array with ghost zones
        decomposition : SlicedDecomposition
            Decomposition strategy
        rank : int
            MPI rank
        size : int
            MPI size
        comm : MPI.Comm
            MPI communicator
        """
        N = u.shape[1]  # Y dimension (same as global for sliced)
        plane_type = self._create_plane_datatype(N)

        lower, upper = decomposition.get_neighbors(rank, size)

        # Exchange with lower neighbor
        if lower is not None:
            comm.Sendrecv(
                [u[1, :, :], 1, plane_type], dest=lower, sendtag=0,
                recvbuf=[u[0, :, :], 1, plane_type], source=lower, recvtag=1
            )

        # Exchange with upper neighbor
        if upper is not None:
            comm.Sendrecv(
                [u[-2, :, :], 1, plane_type], dest=upper, sendtag=1,
                recvbuf=[u[-1, :, :], 1, plane_type], source=upper, recvtag=0
            )

    def exchange_ghosts_cubic(self, u, decomposition, rank, comm):
        """Exchange 6 ghost faces for cubic decomposition.

        Parameters
        ----------
        u : np.ndarray
            Local array with ghost zones
        decomposition : CubicDecomposition
            Decomposition strategy
        rank : int
            MPI rank
        comm : MPI.Comm
            MPI communicator
        """
        neighbors = decomposition.get_neighbors(rank, comm)

        # Exchange faces in each direction
        # X direction
        if neighbors['x_lower'] is not None:
            comm.Sendrecv(
                u[1, :, :].copy(), dest=neighbors['x_lower'], sendtag=0,
                recvbuf=u[0, :, :], source=neighbors['x_lower'], recvtag=1
            )
        if neighbors['x_upper'] is not None:
            comm.Sendrecv(
                u[-2, :, :].copy(), dest=neighbors['x_upper'], sendtag=1,
                recvbuf=u[-1, :, :], source=neighbors['x_upper'], recvtag=0
            )

        # Y direction (requires contiguous buffers)
        if neighbors['y_lower'] is not None:
            send_buf = np.ascontiguousarray(u[:, 1, :])
            recv_buf = np.empty_like(send_buf)
            comm.Sendrecv(send_buf, dest=neighbors['y_lower'], sendtag=2,
                         recvbuf=recv_buf, source=neighbors['y_lower'], recvtag=3)
            u[:, 0, :] = recv_buf
        if neighbors['y_upper'] is not None:
            send_buf = np.ascontiguousarray(u[:, -2, :])
            recv_buf = np.empty_like(send_buf)
            comm.Sendrecv(send_buf, dest=neighbors['y_upper'], sendtag=3,
                         recvbuf=recv_buf, source=neighbors['y_upper'], recvtag=2)
            u[:, -1, :] = recv_buf

        # Z direction
        if neighbors['z_lower'] is not None:
            comm.Sendrecv(
                u[:, :, 1].copy(), dest=neighbors['z_lower'], sendtag=4,
                recvbuf=u[:, :, 0], source=neighbors['z_lower'], recvtag=5
            )
        if neighbors['z_upper'] is not None:
            comm.Sendrecv(
                u[:, :, -2].copy(), dest=neighbors['z_upper'], sendtag=5,
                recvbuf=u[:, :, -1], source=neighbors['z_upper'], recvtag=4
            )


class NumpyCommunicator:
    """Ghost exchange using NumPy arrays with explicit copies.

    Uses ascontiguousarray() to ensure contiguous memory before communication.
    May be more portable but incurs copy overhead.

    Expected Interface:
    - exchange_ghosts(u, decomposition, rank, comm) -> None (in-place)
    """

    def exchange_ghosts(self, u, decomposition, rank, comm):
        """Generic ghost exchange using decomposition's exchange specification.

        Parameters
        ----------
        u : np.ndarray
            Local array with ghost zones (modified in-place)
        decomposition : DecompositionStrategy
            Decomposition strategy providing exchange spec
        rank : int
            MPI rank
        comm : MPI.Comm
            MPI communicator
        """
        # Get exchange specification from decomposition
        exchanges = decomposition.get_ghost_exchange_spec(u, rank, comm)

        # Prepare all send/receive buffers
        send_bufs = []
        recv_bufs = []

        for exchange in exchanges:
            send_slice = exchange['send_slice']
            recv_slice = exchange['recv_slice']

            # Ensure contiguous data for send
            send_bufs.append(np.ascontiguousarray(u[send_slice]))
            recv_bufs.append(np.empty_like(u[recv_slice]))

        # Post all receives first
        recv_reqs = []
        for i, exchange in enumerate(exchanges):
            req = comm.Irecv(recv_bufs[i], source=exchange['neighbor'], tag=exchange['tag'])
            recv_reqs.append(req)

        # Post all sends
        send_reqs = []
        for i, exchange in enumerate(exchanges):
            req = comm.Isend(send_bufs[i], dest=exchange['neighbor'], tag=exchange['tag'])
            send_reqs.append(req)

        # Wait for all communications to complete
        MPI.Request.Waitall(recv_reqs + send_reqs)

        # Copy received data into ghost zones
        for i, exchange in enumerate(exchanges):
            u[exchange['recv_slice']] = recv_bufs[i]

    def exchange_ghosts_sliced(self, u, decomposition, rank, size, comm):
        """Exchange ghost planes using NumPy arrays.

        Parameters
        ----------
        u : np.ndarray
            Local array with ghost zones
        decomposition : SlicedDecomposition
            Decomposition strategy
        rank : int
            MPI rank
        size : int
            MPI size
        comm : MPI.Comm
            MPI communicator
        """
        lower, upper = decomposition.get_neighbors(rank, size)

        # Exchange with lower neighbor
        if lower is not None:
            send_buf = np.ascontiguousarray(u[1, :, :])
            recv_buf = np.empty_like(send_buf)
            comm.Sendrecv(send_buf, dest=lower, sendtag=0,
                         recvbuf=recv_buf, source=lower, recvtag=1)
            u[0, :, :] = recv_buf

        # Exchange with upper neighbor
        if upper is not None:
            send_buf = np.ascontiguousarray(u[-2, :, :])
            recv_buf = np.empty_like(send_buf)
            comm.Sendrecv(send_buf, dest=upper, sendtag=1,
                         recvbuf=recv_buf, source=upper, recvtag=0)
            u[-1, :, :] = recv_buf

    def exchange_ghosts_cubic(self, u, decomposition, rank, comm):
        """Exchange 6 ghost faces using NumPy arrays.

        Parameters
        ----------
        u : np.ndarray
            Local array with ghost zones
        decomposition : CubicDecomposition
            Decomposition strategy
        rank : int
            MPI rank
        comm : MPI.Comm
            MPI communicator
        """
        neighbors = decomposition.get_neighbors(rank, comm)

        # X direction
        if neighbors['x_lower'] is not None:
            send_buf = np.ascontiguousarray(u[1, :, :])
            recv_buf = np.empty_like(send_buf)
            comm.Sendrecv(send_buf, dest=neighbors['x_lower'], sendtag=0,
                         recvbuf=recv_buf, source=neighbors['x_lower'], recvtag=1)
            u[0, :, :] = recv_buf
        if neighbors['x_upper'] is not None:
            send_buf = np.ascontiguousarray(u[-2, :, :])
            recv_buf = np.empty_like(send_buf)
            comm.Sendrecv(send_buf, dest=neighbors['x_upper'], sendtag=1,
                         recvbuf=recv_buf, source=neighbors['x_upper'], recvtag=0)
            u[-1, :, :] = recv_buf

        # Y direction
        if neighbors['y_lower'] is not None:
            send_buf = np.ascontiguousarray(u[:, 1, :])
            recv_buf = np.empty_like(send_buf)
            comm.Sendrecv(send_buf, dest=neighbors['y_lower'], sendtag=2,
                         recvbuf=recv_buf, source=neighbors['y_lower'], recvtag=3)
            u[:, 0, :] = recv_buf
        if neighbors['y_upper'] is not None:
            send_buf = np.ascontiguousarray(u[:, -2, :])
            recv_buf = np.empty_like(send_buf)
            comm.Sendrecv(send_buf, dest=neighbors['y_upper'], sendtag=3,
                         recvbuf=recv_buf, source=neighbors['y_upper'], recvtag=2)
            u[:, -1, :] = recv_buf

        # Z direction
        if neighbors['z_lower'] is not None:
            send_buf = np.ascontiguousarray(u[:, :, 1])
            recv_buf = np.empty_like(send_buf)
            comm.Sendrecv(send_buf, dest=neighbors['z_lower'], sendtag=4,
                         recvbuf=recv_buf, source=neighbors['z_lower'], recvtag=5)
            u[:, :, 0] = recv_buf
        if neighbors['z_upper'] is not None:
            send_buf = np.ascontiguousarray(u[:, :, -2])
            recv_buf = np.empty_like(send_buf)
            comm.Sendrecv(send_buf, dest=neighbors['z_upper'], sendtag=5,
                         recvbuf=recv_buf, source=neighbors['z_upper'], recvtag=4)
            u[:, :, -1] = recv_buf

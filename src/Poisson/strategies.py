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

    def initialize_local_arrays_distributed(self, N, rank, size):
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
        size : int
            MPI size

        Returns
        -------
        u1_local, u2_local, f_local : np.ndarray
            Local arrays with ghost zones, properly initialized
        """
        # Get local domain decomposition
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

    def gather_solution(self, u_local, N, rank, comm):
        """Gather local solutions to global array on rank 0.

        Parameters
        ----------
        u_local : np.ndarray
            Local solution array with ghost zones
        N : int
            Global grid size
        rank : int
            MPI rank
        comm : MPI.Comm
            MPI communicator

        Returns
        -------
        u_global : np.ndarray or None
            Global solution on rank 0, None on other ranks
        """
        u_global = np.zeros((N, N, N)) if rank == 0 else None
        local_interior = u_local[1:-1, :, :].copy()
        all_locals = comm.gather(local_interior, root=0)

        if rank == 0:
            current_z = 1
            for rank_data in all_locals:
                rank_local_N = rank_data.shape[0]
                u_global[current_z : current_z + rank_local_N, :, :] = rank_data
                current_z += rank_local_N

        return u_global


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

    def gather_solution(self, u_local, N, rank, comm):
        """Gather local solutions to global array on rank 0.

        Parameters
        ----------
        u_local : np.ndarray
            Local solution array with ghost zones
        N : int
            Global grid size
        rank : int
            MPI rank
        comm : MPI.Comm
            MPI communicator (not used, cart_comm is stored in self)

        Returns
        -------
        u_global : np.ndarray or None
            Global solution on rank 0, None on other ranks
        """
        # Extract interior points
        local_interior = u_local[1:-1, 1:-1, 1:-1].copy()

        # Gather to rank 0
        all_locals = self.cart_comm.gather(local_interior, root=0)

        if rank == 0:
            u_global = np.zeros((N, N, N))
            interior_N = N - 2

            # Reconstruct global array from all local pieces
            for rank_id, rank_data in enumerate(all_locals):
                # Get coordinates for this rank
                rank_coords = self.cart_comm.Get_coords(rank_id)

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

                u_global[x_start:x_end, y_start:y_end, z_start:z_end] = rank_data

            return u_global
        else:
            return None


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

"""Halo exchange communicators for MPI domain decomposition."""
import numpy as np
from mpi4py import MPI

# Axis config: (index, name) - used to build neighbor keys like 'z_lower'
_AXES = [(0, 'z'), (1, 'y'), (2, 'x')]


def _face_slice(axis, idx, has_halo):
    """Slice for face at idx, excluding halos of other decomposed axes."""
    return tuple(
        idx if ax == axis else slice(1, -1) if has_halo[ax] else slice(None)
        for ax in range(3)
    )


class _BaseHaloExchange:
    """Base class with shared exchange loop logic."""

    def exchange_halos(self, u, decomposition, rank, comm):
        """Exchange halo zones with neighbors."""
        neighbors = decomposition.get_neighbors(rank)
        has_halo = {ax: f'{name}_lower' in neighbors for ax, name in _AXES}

        for axis, name in _AXES:
            if not has_halo[axis]:
                continue
            lo = neighbors.get(f'{name}_lower')
            hi = neighbors.get(f'{name}_upper')
            self._exchange_axis(u, axis, lo, hi, has_halo, comm, tag=axis * 100)


class NumpyHaloExchange(_BaseHaloExchange):
    """Halo exchange using NumPy array copies."""
    name = "NumPy"

    def _exchange_axis(self, u, axis, lo_rank, hi_rank, has_halo, comm, tag):
        lo = lo_rank if lo_rank is not None else MPI.PROC_NULL
        hi = hi_rank if hi_rank is not None else MPI.PROC_NULL

        for send_i, recv_i, dest, src in [(-2, 0, hi, lo), (1, -1, lo, hi)]:
            send = np.ascontiguousarray(u[_face_slice(axis, send_i, has_halo)])
            recv = np.empty_like(send)
            comm.Sendrecv(send, dest, tag, recv, src, tag)
            u[_face_slice(axis, recv_i, has_halo)] = recv if src != MPI.PROC_NULL else 0.0
            tag += 1


class CustomHaloExchange(_BaseHaloExchange):
    """Halo exchange using MPI derived datatypes (zero-copy)."""
    name = "MPI Datatype"

    def __init__(self):
        self._cache = {}  # (shape, has_halo_tuple) -> {axis: datatype}

    def _exchange_axis(self, u, axis, lo_rank, hi_rank, has_halo, comm, tag):
        lo = lo_rank if lo_rank is not None else MPI.PROC_NULL
        hi = hi_rank if hi_rank is not None else MPI.PROC_NULL

        # Get or create datatype for this configuration
        key = (u.shape, tuple(has_halo.items()))
        if key not in self._cache:
            self._cache[key] = self._create_datatypes(u.shape, has_halo)
        dtype = self._cache[key][axis]

        nz, ny, nx = u.shape
        u_flat = u.ravel()

        # Compute offsets into flattened array
        start = [1 if has_halo[ax] else 0 for ax in range(3)]
        flat = lambda z, y, x: z * ny * nx + y * nx + x

        if axis == 0:
            send_hi, recv_lo = flat(nz-2, start[1], start[2]), flat(0, start[1], start[2])
            send_lo, recv_hi = flat(1, start[1], start[2]), flat(nz-1, start[1], start[2])
        elif axis == 1:
            send_hi, recv_lo = flat(start[0], ny-2, start[2]), flat(start[0], 0, start[2])
            send_lo, recv_hi = flat(start[0], 1, start[2]), flat(start[0], ny-1, start[2])
        else:
            send_hi, recv_lo = flat(start[0], start[1], nx-2), flat(start[0], start[1], 0)
            send_lo, recv_hi = flat(start[0], start[1], 1), flat(start[0], start[1], nx-1)

        # Exchange
        comm.Sendrecv([u_flat[send_hi:], 1, dtype], hi, tag,
                      [u_flat[recv_lo:], 1, dtype], lo, tag)
        if lo_rank is None:
            u[_face_slice(axis, 0, has_halo)] = 0.0

        comm.Sendrecv([u_flat[send_lo:], 1, dtype], lo, tag + 1,
                      [u_flat[recv_hi:], 1, dtype], hi, tag + 1)
        if hi_rank is None:
            u[_face_slice(axis, -1, has_halo)] = 0.0

    def _create_datatypes(self, shape, has_halo):
        """Create MPI datatypes for each axis."""
        nz, ny, nx = shape
        nz_int = nz - 2 if has_halo[0] else nz
        ny_int = ny - 2 if has_halo[1] else ny
        nx_int = nx - 2 if has_halo[2] else nx

        datatypes = {}

        # Z-face: contiguous if no y/x halos, else strided
        if not has_halo[1] and not has_halo[2]:
            dt = MPI.DOUBLE.Create_contiguous(ny * nx)
        else:
            dt = MPI.DOUBLE.Create_vector(ny_int, nx_int, nx)
        dt.Commit()
        datatypes[0] = dt

        # Y-face: strided across z-planes
        dt = MPI.DOUBLE.Create_vector(nz_int, nx_int, ny * nx)
        dt.Commit()
        datatypes[1] = dt

        # X-face: 2D strided (most complex)
        row = MPI.DOUBLE.Create_vector(ny_int, 1, nx)
        row.Commit()
        dt = row.Create_hvector(nz_int, 1, ny * nx * MPI.DOUBLE.Get_size())
        dt.Commit()
        row.Free()
        datatypes[2] = dt

        return datatypes

    def __del__(self):
        for datatypes in self._cache.values():
            for dt in datatypes.values():
                dt.Free()

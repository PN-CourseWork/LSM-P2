"""Halo exchange implementations for distributed grids."""

from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np
from mpi4py import MPI


class HaloExchanger(ABC):
    """Abstract base for halo exchange strategies."""

    @abstractmethod
    def setup(self, local_shape: tuple[int, int, int], neighbors: dict):
        """Initialize exchange buffers or datatypes."""
        pass

    @abstractmethod
    def exchange(self, arr: np.ndarray, cart_comm: MPI.Comm, neighbors: dict):
        """Perform halo exchange on array."""
        pass


class NumpyHaloExchanger(HaloExchanger):
    """Halo exchange using numpy buffer copies and Sendrecv.

    Simple implementation that copies face data to contiguous buffers
    before sending. Good baseline performance.
    """

    def setup(self, local_shape: tuple[int, int, int], neighbors: dict):
        """Pre-allocate send/receive buffers."""
        nz, ny, nx = local_shape
        self._local_shape = local_shape

        self._send_bufs = {}
        self._recv_bufs = {}

        face_sizes = {
            "z": ny * nx,
            "y": nz * nx,
            "x": nz * ny,
        }

        for axis in ["z", "y", "x"]:
            for direction in ["lower", "upper"]:
                key = f"{axis}_{direction}"
                if neighbors.get(key) is not None:
                    self._send_bufs[key] = np.empty(face_sizes[axis], dtype=np.float64)
                    self._recv_bufs[key] = np.empty(face_sizes[axis], dtype=np.float64)

    def exchange(self, arr: np.ndarray, cart_comm: MPI.Comm, neighbors: dict):
        """Exchange halos using Sendrecv with buffer copies."""
        nz, ny, nx = self._local_shape

        # Z-direction
        if neighbors["z_lower"] is not None or neighbors["z_upper"] is not None:
            lo = (
                neighbors["z_lower"]
                if neighbors["z_lower"] is not None
                else MPI.PROC_NULL
            )
            hi = (
                neighbors["z_upper"]
                if neighbors["z_upper"] is not None
                else MPI.PROC_NULL
            )

            # Send to upper, receive from lower
            send_buf = np.ascontiguousarray(arr[-2, 1:-1, 1:-1])
            recv_buf = np.empty_like(send_buf)
            cart_comm.Sendrecv(send_buf, hi, 0, recv_buf, lo, 0)
            if neighbors["z_lower"] is not None:
                arr[0, 1:-1, 1:-1] = recv_buf.reshape(ny, nx)

            # Send to lower, receive from upper
            send_buf = np.ascontiguousarray(arr[1, 1:-1, 1:-1])
            recv_buf = np.empty_like(send_buf)
            cart_comm.Sendrecv(send_buf, lo, 1, recv_buf, hi, 1)
            if neighbors["z_upper"] is not None:
                arr[-1, 1:-1, 1:-1] = recv_buf.reshape(ny, nx)

        # Y-direction
        if neighbors["y_lower"] is not None or neighbors["y_upper"] is not None:
            lo = (
                neighbors["y_lower"]
                if neighbors["y_lower"] is not None
                else MPI.PROC_NULL
            )
            hi = (
                neighbors["y_upper"]
                if neighbors["y_upper"] is not None
                else MPI.PROC_NULL
            )

            send_buf = np.ascontiguousarray(arr[1:-1, -2, 1:-1])
            recv_buf = np.empty_like(send_buf)
            cart_comm.Sendrecv(send_buf, hi, 2, recv_buf, lo, 2)
            if neighbors["y_lower"] is not None:
                arr[1:-1, 0, 1:-1] = recv_buf.reshape(nz, nx)

            send_buf = np.ascontiguousarray(arr[1:-1, 1, 1:-1])
            recv_buf = np.empty_like(send_buf)
            cart_comm.Sendrecv(send_buf, lo, 3, recv_buf, hi, 3)
            if neighbors["y_upper"] is not None:
                arr[1:-1, -1, 1:-1] = recv_buf.reshape(nz, nx)

        # X-direction
        if neighbors["x_lower"] is not None or neighbors["x_upper"] is not None:
            lo = (
                neighbors["x_lower"]
                if neighbors["x_lower"] is not None
                else MPI.PROC_NULL
            )
            hi = (
                neighbors["x_upper"]
                if neighbors["x_upper"] is not None
                else MPI.PROC_NULL
            )

            send_buf = np.ascontiguousarray(arr[1:-1, 1:-1, -2])
            recv_buf = np.empty_like(send_buf)
            cart_comm.Sendrecv(send_buf, hi, 4, recv_buf, lo, 4)
            if neighbors["x_lower"] is not None:
                arr[1:-1, 1:-1, 0] = recv_buf.reshape(nz, ny)

            send_buf = np.ascontiguousarray(arr[1:-1, 1:-1, 1])
            recv_buf = np.empty_like(send_buf)
            cart_comm.Sendrecv(send_buf, lo, 5, recv_buf, hi, 5)
            if neighbors["x_upper"] is not None:
                arr[1:-1, 1:-1, -1] = recv_buf.reshape(nz, ny)


class DatatypeHaloExchanger(HaloExchanger):
    """Halo exchange using MPI derived datatypes (zero-copy).

    Creates custom MPI datatypes that describe the memory layout of
    each face, allowing direct send/recv without buffer copies.
    """

    def setup(self, local_shape: tuple[int, int, int], neighbors: dict):
        """Create MPI datatypes for each face."""
        nz, ny, nx = local_shape
        self._halo_shape = (nz + 2, ny + 2, nx + 2)
        self._datatypes = self._create_datatypes()

    def _create_datatypes(self):
        """Create MPI datatypes for zero-copy exchange."""
        nz, ny, nx = self._halo_shape
        datatypes = {}

        ny_int, nx_int = ny - 2, nx - 2
        nz_int = nz - 2

        # Z-face: ny_int x nx_int contiguous block
        dt = MPI.DOUBLE.Create_vector(ny_int, nx_int, nx)
        dt.Commit()
        datatypes["z"] = dt

        # Y-face: nz_int planes, nx_int per plane
        dt = MPI.DOUBLE.Create_vector(nz_int, nx_int, ny * nx)
        dt.Commit()
        datatypes["y"] = dt

        # X-face: 2D strided
        row = MPI.DOUBLE.Create_vector(ny_int, 1, nx)
        row.Commit()
        dt = row.Create_hvector(nz_int, 1, ny * nx * MPI.DOUBLE.Get_size())
        dt.Commit()
        row.Free()
        datatypes["x"] = dt

        return datatypes

    def exchange(self, arr: np.ndarray, cart_comm: MPI.Comm, neighbors: dict):
        """Exchange halos using MPI datatypes."""
        nz, ny, nx = self._halo_shape
        arr_flat = arr.ravel()

        def flat_idx(z, y, x):
            return z * ny * nx + y * nx + x

        # Z-direction
        if neighbors["z_lower"] is not None or neighbors["z_upper"] is not None:
            lo = (
                neighbors["z_lower"]
                if neighbors["z_lower"] is not None
                else MPI.PROC_NULL
            )
            hi = (
                neighbors["z_upper"]
                if neighbors["z_upper"] is not None
                else MPI.PROC_NULL
            )
            dt = self._datatypes["z"]

            send_off = flat_idx(nz - 2, 1, 1)
            recv_off = flat_idx(0, 1, 1)
            cart_comm.Sendrecv(
                [arr_flat[send_off:], 1, dt], hi, 0, [arr_flat[recv_off:], 1, dt], lo, 0
            )
            if neighbors["z_lower"] is None:
                arr[0, 1:-1, 1:-1] = 0.0

            send_off = flat_idx(1, 1, 1)
            recv_off = flat_idx(nz - 1, 1, 1)
            cart_comm.Sendrecv(
                [arr_flat[send_off:], 1, dt], lo, 1, [arr_flat[recv_off:], 1, dt], hi, 1
            )
            if neighbors["z_upper"] is None:
                arr[-1, 1:-1, 1:-1] = 0.0

        # Y-direction
        if neighbors["y_lower"] is not None or neighbors["y_upper"] is not None:
            lo = (
                neighbors["y_lower"]
                if neighbors["y_lower"] is not None
                else MPI.PROC_NULL
            )
            hi = (
                neighbors["y_upper"]
                if neighbors["y_upper"] is not None
                else MPI.PROC_NULL
            )
            dt = self._datatypes["y"]

            send_off = flat_idx(1, ny - 2, 1)
            recv_off = flat_idx(1, 0, 1)
            cart_comm.Sendrecv(
                [arr_flat[send_off:], 1, dt], hi, 2, [arr_flat[recv_off:], 1, dt], lo, 2
            )
            if neighbors["y_lower"] is None:
                arr[1:-1, 0, 1:-1] = 0.0

            send_off = flat_idx(1, 1, 1)
            recv_off = flat_idx(1, ny - 1, 1)
            cart_comm.Sendrecv(
                [arr_flat[send_off:], 1, dt], lo, 3, [arr_flat[recv_off:], 1, dt], hi, 3
            )
            if neighbors["y_upper"] is None:
                arr[1:-1, -1, 1:-1] = 0.0

        # X-direction
        if neighbors["x_lower"] is not None or neighbors["x_upper"] is not None:
            lo = (
                neighbors["x_lower"]
                if neighbors["x_lower"] is not None
                else MPI.PROC_NULL
            )
            hi = (
                neighbors["x_upper"]
                if neighbors["x_upper"] is not None
                else MPI.PROC_NULL
            )
            dt = self._datatypes["x"]

            send_off = flat_idx(1, 1, nx - 2)
            recv_off = flat_idx(1, 1, 0)
            cart_comm.Sendrecv(
                [arr_flat[send_off:], 1, dt], hi, 4, [arr_flat[recv_off:], 1, dt], lo, 4
            )
            if neighbors["x_lower"] is None:
                arr[1:-1, 1:-1, 0] = 0.0

            send_off = flat_idx(1, 1, 1)
            recv_off = flat_idx(1, 1, nx - 1)
            cart_comm.Sendrecv(
                [arr_flat[send_off:], 1, dt], lo, 5, [arr_flat[recv_off:], 1, dt], hi, 5
            )
            if neighbors["x_upper"] is None:
                arr[1:-1, 1:-1, -1] = 0.0

    def __del__(self):
        """Free MPI datatypes."""
        if hasattr(self, "_datatypes"):
            for dt in self._datatypes.values():
                if dt != MPI.DATATYPE_NULL:
                    dt.Free()


def create_halo_exchanger(exchange_type: str) -> HaloExchanger:
    """Factory function to create halo exchanger.

    Parameters
    ----------
    exchange_type : str
        'numpy' for buffer-based exchange,
        'custom' for MPI derived datatypes.

    Returns
    -------
    HaloExchanger
        The appropriate exchanger instance.
    """
    if exchange_type == "numpy":
        return NumpyHaloExchanger()
    elif exchange_type == "custom":
        return DatatypeHaloExchanger()
    else:
        raise ValueError(f"Unknown halo_exchange type: {exchange_type}")

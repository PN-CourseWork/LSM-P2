"""Halo exchange implementations for distributed grids."""

from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np
from mpi4py import MPI


# Axis names for neighbor dict keys
AXIS_NAMES = ["z", "y", "x"]


def _neighbor_ranks(neighbors: dict, axis_idx: int) -> tuple[int, int]:
    """Get neighbor ranks for an axis, defaulting to MPI.PROC_NULL."""
    name = AXIS_NAMES[axis_idx]
    lo = neighbors.get(f"{name}_lower")
    hi = neighbors.get(f"{name}_upper")
    return (lo if lo is not None else MPI.PROC_NULL,
            hi if hi is not None else MPI.PROC_NULL)


def _make_slice(axis_idx: int, axis_val, ndim: int = 3):
    """Build slice tuple with axis_val at axis_idx, slice(1,-1) elsewhere."""
    slices = [slice(1, -1)] * ndim
    slices[axis_idx] = axis_val
    return tuple(slices)


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
    """Halo exchange using numpy buffer copies and Sendrecv."""

    def setup(self, local_shape: tuple[int, int, int], neighbors: dict):
        """Pre-compute slices and shapes for each axis."""
        self._axis_info = []
        for axis_idx in range(3):
            # recv_shape: dimensions excluding this axis
            recv_shape = tuple(local_shape[i] for i in range(3) if i != axis_idx)
            self._axis_info.append({
                "send_hi": _make_slice(axis_idx, -2),
                "send_lo": _make_slice(axis_idx, 1),
                "recv_lo": _make_slice(axis_idx, 0),
                "recv_hi": _make_slice(axis_idx, -1),
                "recv_shape": recv_shape,
            })

    def exchange(self, arr: np.ndarray, cart_comm: MPI.Comm, neighbors: dict):
        """Exchange halos using Sendrecv with buffer copies."""
        for axis_idx, info in enumerate(self._axis_info):
            lo, hi = _neighbor_ranks(neighbors, axis_idx)
            if lo == MPI.PROC_NULL and hi == MPI.PROC_NULL:
                continue

            name = AXIS_NAMES[axis_idx]
            tag = axis_idx * 2

            # Send to upper, receive from lower
            send = np.ascontiguousarray(arr[info["send_hi"]])
            recv = np.empty_like(send)
            cart_comm.Sendrecv(send, hi, tag, recv, lo, tag)
            if neighbors.get(f"{name}_lower") is not None:
                arr[info["recv_lo"]] = recv.reshape(info["recv_shape"])

            # Send to lower, receive from upper
            send = np.ascontiguousarray(arr[info["send_lo"]])
            recv = np.empty_like(send)
            cart_comm.Sendrecv(send, lo, tag + 1, recv, hi, tag + 1)
            if neighbors.get(f"{name}_upper") is not None:
                arr[info["recv_hi"]] = recv.reshape(info["recv_shape"])


class DatatypeHaloExchanger(HaloExchanger):
    """Halo exchange using MPI derived datatypes (zero-copy)."""

    def setup(self, local_shape: tuple[int, int, int], neighbors: dict):
        """Create MPI datatypes and pre-compute offsets."""
        nz, ny, nx = local_shape
        hz, hy, hx = nz + 2, ny + 2, nx + 2  # halo shape
        self._halo_shape = (hz, hy, hx)

        # Strides for flat indexing
        strides = (hy * hx, hx, 1)

        def flat_idx(z, y, x):
            return z * strides[0] + y * strides[1] + x * strides[2]

        # Create datatypes and offsets for each axis
        self._axis_info = []

        # Z-face datatype
        dt_z = MPI.DOUBLE.Create_vector(ny, nx, hx)
        dt_z.Commit()
        self._axis_info.append({
            "dt": dt_z,
            "send_hi": flat_idx(hz - 2, 1, 1),
            "send_lo": flat_idx(1, 1, 1),
            "recv_lo": flat_idx(0, 1, 1),
            "recv_hi": flat_idx(hz - 1, 1, 1),
        })

        # Y-face datatype
        dt_y = MPI.DOUBLE.Create_vector(nz, nx, hy * hx)
        dt_y.Commit()
        self._axis_info.append({
            "dt": dt_y,
            "send_hi": flat_idx(1, hy - 2, 1),
            "send_lo": flat_idx(1, 1, 1),
            "recv_lo": flat_idx(1, 0, 1),
            "recv_hi": flat_idx(1, hy - 1, 1),
        })

        # X-face datatype (2D strided)
        row = MPI.DOUBLE.Create_vector(ny, 1, hx)
        row.Commit()
        dt_x = row.Create_hvector(nz, 1, hy * hx * MPI.DOUBLE.Get_size())
        dt_x.Commit()
        row.Free()
        self._axis_info.append({
            "dt": dt_x,
            "send_hi": flat_idx(1, 1, hx - 2),
            "send_lo": flat_idx(1, 1, 1),
            "recv_lo": flat_idx(1, 1, 0),
            "recv_hi": flat_idx(1, 1, hx - 1),
        })

    def exchange(self, arr: np.ndarray, cart_comm: MPI.Comm, neighbors: dict):
        """Exchange halos using MPI datatypes."""
        flat = arr.ravel()

        for axis_idx, info in enumerate(self._axis_info):
            lo, hi = _neighbor_ranks(neighbors, axis_idx)
            if lo == MPI.PROC_NULL and hi == MPI.PROC_NULL:
                continue

            name = AXIS_NAMES[axis_idx]
            tag = axis_idx * 2
            dt = info["dt"]

            # Send to upper, receive from lower
            cart_comm.Sendrecv(
                [flat[info["send_hi"]:], 1, dt], hi, tag,
                [flat[info["recv_lo"]:], 1, dt], lo, tag,
            )
            if neighbors.get(f"{name}_lower") is None:
                arr[_make_slice(axis_idx, 0)] = 0.0

            # Send to lower, receive from upper
            cart_comm.Sendrecv(
                [flat[info["send_lo"]:], 1, dt], lo, tag + 1,
                [flat[info["recv_hi"]:], 1, dt], hi, tag + 1,
            )
            if neighbors.get(f"{name}_upper") is None:
                arr[_make_slice(axis_idx, -1)] = 0.0

    def __del__(self):
        """Free MPI datatypes."""
        if hasattr(self, "_axis_info"):
            for info in self._axis_info:
                dt = info["dt"]
                if dt != MPI.DATATYPE_NULL:
                    dt.Free()


def create_halo_exchanger(exchange_type: str) -> HaloExchanger:
    """Factory: 'numpy' for buffer-based, 'custom' for MPI datatypes."""
    if exchange_type == "numpy":
        return NumpyHaloExchanger()
    elif exchange_type == "custom":
        return DatatypeHaloExchanger()
    else:
        raise ValueError(f"Unknown halo_exchange type: {exchange_type}")

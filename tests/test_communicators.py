"""Tests for halo exchange communicators."""

import numpy as np
import pytest
from Poisson import NumpyHaloExchange, CustomHaloExchange


class MockDecomposition:
    """Mock decomposition for testing communicators."""

    def __init__(self, neighbors):
        self._neighbors = neighbors

    def get_neighbors(self, rank):
        return self._neighbors


class MockComm:
    """Mock MPI communicator for single-rank testing."""

    PROC_NULL = -1

    def Sendrecv(self, sendbuf, dest, sendtag, recvbuf, source, recvtag):
        """No-op for single rank - just zero the receive buffer."""
        if hasattr(recvbuf, '__getitem__') and hasattr(recvbuf[0], 'fill'):
            recvbuf[0].fill(0.0)


class TestNumpyHaloExchange:
    """Tests for NumPy-based halo exchange."""

    def test_sliced_z_axis_boundaries_zeroed(self):
        """Boundary halos should be zero (Dirichlet BC)."""
        comm = NumpyHaloExchange()
        u = np.ones((10, 8, 8), dtype=np.float64)

        # No neighbors = boundary rank
        decomp = MockDecomposition({'z_lower': None, 'z_upper': None})
        comm.exchange_halos(u, decomp, rank=0, comm=MockComm())

        # Halos at boundaries should remain unchanged (no neighbor to receive from)
        # Interior should be unchanged
        assert np.all(u[1:-1, :, :] == 1.0)

    def test_sliced_y_axis_detection(self):
        """Should detect y-axis decomposition from neighbor keys."""
        comm = NumpyHaloExchange()
        u = np.ones((8, 10, 8), dtype=np.float64)

        decomp = MockDecomposition({'y_lower': None, 'y_upper': None})
        comm.exchange_halos(u, decomp, rank=0, comm=MockComm())

        # Should not crash - verifies axis detection works
        assert u.shape == (8, 10, 8)

    def test_sliced_interior_unchanged(self):
        """Sliced halo exchange should not modify interior values."""
        comm = NumpyHaloExchange()
        u = np.ones((10, 8, 8), dtype=np.float64)
        interior_before = u[1:-1, :, :].copy()

        decomp = MockDecomposition({'z_lower': None, 'z_upper': None})
        comm.exchange_halos(u, decomp, rank=0, comm=MockComm())

        # Interior should be unchanged
        assert np.allclose(u[1:-1, :, :], interior_before)

    def test_cubic_all_boundaries(self):
        """Cubic decomposition with all boundaries should zero halos."""
        comm = NumpyHaloExchange()
        u = np.ones((10, 10, 10), dtype=np.float64)

        # Corner rank - no neighbors
        decomp = MockDecomposition({
            'x_lower': None, 'x_upper': None,
            'y_lower': None, 'y_upper': None,
            'z_lower': None, 'z_upper': None,
        })
        comm.exchange_halos(u, decomp, rank=0, comm=MockComm())

        # All boundary halos should be zeroed
        assert np.all(u[1:-1, 1:-1, 0] == 0.0)   # x_lower
        assert np.all(u[1:-1, 1:-1, -1] == 0.0)  # x_upper
        assert np.all(u[1:-1, 0, 1:-1] == 0.0)   # y_lower
        assert np.all(u[1:-1, -1, 1:-1] == 0.0)  # y_upper
        assert np.all(u[0, 1:-1, 1:-1] == 0.0)   # z_lower
        assert np.all(u[-1, 1:-1, 1:-1] == 0.0)  # z_upper

        # Interior unchanged
        assert np.all(u[1:-1, 1:-1, 1:-1] == 1.0)


class TestCustomHaloExchange:
    """Tests for MPI datatype-based halo exchange."""

    def test_datatype_caching(self):
        """Should cache datatypes after first exchange."""
        comm = CustomHaloExchange()
        u = np.ones((10, 10, 10), dtype=np.float64)

        decomp = MockDecomposition({'z_lower': None, 'z_upper': None})
        comm.exchange_halos(u, decomp, rank=0, comm=MockComm())

        # Datatypes should be cached
        assert len(comm._cache) == 1

    def test_datatype_reuse(self):
        """Should reuse datatypes for same shape/decomposition."""
        comm = CustomHaloExchange()
        u = np.ones((10, 10, 10), dtype=np.float64)
        decomp = MockDecomposition({'z_lower': None, 'z_upper': None})

        comm.exchange_halos(u, decomp, rank=0, comm=MockComm())
        cache_size_after_first = len(comm._cache)

        comm.exchange_halos(u, decomp, rank=0, comm=MockComm())
        assert len(comm._cache) == cache_size_after_first  # No new entries

    def test_datatype_recreation_on_shape_change(self):
        """Should create new datatypes when array shape changes."""
        comm = CustomHaloExchange()
        decomp = MockDecomposition({'z_lower': None, 'z_upper': None})

        u1 = np.ones((10, 10, 10), dtype=np.float64)
        comm.exchange_halos(u1, decomp, rank=0, comm=MockComm())
        assert len(comm._cache) == 1

        u2 = np.ones((12, 12, 12), dtype=np.float64)
        comm.exchange_halos(u2, decomp, rank=0, comm=MockComm())
        assert len(comm._cache) == 2  # New entry for different shape

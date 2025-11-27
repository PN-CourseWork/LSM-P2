"""Tests for domain decomposition logic."""

import pytest
import numpy as np
from Poisson import DomainDecomposition


class TestSlicedDecomposition:
    """Tests for sliced (1D) decomposition."""

    @pytest.mark.parametrize("N,size", [(50, 4), (64, 8), (100, 7)])
    def test_full_coverage_no_overlaps(self, N, size):
        """Each interior point owned by exactly one rank."""
        decomp = DomainDecomposition(N=N, size=size, strategy='sliced')

        total_nz = sum(decomp.get_rank_info(r).local_shape[0] for r in range(size))
        assert total_nz == N - 2  # interior size

    def test_neighbors_correct(self):
        """Interior ranks have 2 neighbors, boundary ranks have 1."""
        decomp = DomainDecomposition(N=50, size=4, strategy='sliced')

        # First rank: no lower neighbor
        assert decomp.get_rank_info(0).neighbors['z_lower'] is None
        assert decomp.get_rank_info(0).neighbors['z_upper'] == 1

        # Interior rank: both neighbors
        assert decomp.get_rank_info(1).neighbors['z_lower'] == 0
        assert decomp.get_rank_info(1).neighbors['z_upper'] == 2

        # Last rank: no upper neighbor
        assert decomp.get_rank_info(3).neighbors['z_lower'] == 2
        assert decomp.get_rank_info(3).neighbors['z_upper'] is None

    def test_halo_shape(self):
        """Halo shape adds 2 in z-direction only."""
        decomp = DomainDecomposition(N=50, size=4, strategy='sliced')
        info = decomp.get_rank_info(1)

        nz, ny, nx = info.local_shape
        hz, hy, hx = info.halo_shape

        assert hz == nz + 2
        assert hy == ny == 50
        assert hx == nx == 50


class TestCubicDecomposition:
    """Tests for cubic (3D) decomposition."""

    def test_full_coverage_no_overlaps(self):
        """Each point owned by exactly one rank."""
        N, size = 64, 8
        decomp = DomainDecomposition(N=N, size=size, strategy='cubic')

        covered = np.zeros((N, N, N), dtype=bool)
        for rank in range(size):
            info = decomp.get_rank_info(rank)
            z0, y0, x0 = info.global_start
            z1, y1, x1 = info.global_end
            assert not np.any(covered[z0:z1, y0:y1, x0:x1])  # no overlap
            covered[z0:z1, y0:y1, x0:x1] = True

        assert np.all(covered)  # full coverage

    def test_neighbor_reciprocity(self):
        """If A neighbors B, then B neighbors A."""
        decomp = DomainDecomposition(N=64, size=8, strategy='cubic')

        opposites = {'x_lower': 'x_upper', 'x_upper': 'x_lower',
                     'y_lower': 'y_upper', 'y_upper': 'y_lower',
                     'z_lower': 'z_upper', 'z_upper': 'z_lower'}

        for rank in range(8):
            info = decomp.get_rank_info(rank)
            for direction, neighbor in info.neighbors.items():
                if neighbor is not None:
                    neighbor_info = decomp.get_rank_info(neighbor)
                    assert neighbor_info.neighbors[opposites[direction]] == rank

    def test_corner_has_3_neighbors(self):
        """Corner rank (rank 0) touches 3 boundaries, has 3 neighbors."""
        decomp = DomainDecomposition(N=64, size=8, strategy='cubic')
        info = decomp.get_rank_info(0)

        assert info.global_start == (0, 0, 0)
        assert info.n_neighbors == 3

    def test_interior_has_6_neighbors(self):
        """Interior rank has all 6 neighbors."""
        decomp = DomainDecomposition(N=81, size=27, strategy='cubic')  # 3x3x3

        # Center rank at (1,1,1) in processor grid
        py, pz = decomp.dims[1], decomp.dims[2]
        center = 1 * (py * pz) + 1 * pz + 1

        info = decomp.get_rank_info(center)
        assert info.n_neighbors == 6
        assert all(n is not None for n in info.neighbors.values())


class TestConfigurableAxis:
    """Tests for configurable slicing axis."""

    def test_z_axis_works(self):
        """Default z-axis decomposition."""
        decomp = DomainDecomposition(N=50, size=4, strategy='sliced')
        info = decomp.get_rank_info(0)

        assert info.local_shape[1] == 50  # full y
        assert info.local_shape[2] == 50  # full x
        assert 'z_lower' in info.neighbors

    def test_y_axis_decomposition(self):
        """Decompose along y-axis."""
        decomp = DomainDecomposition(N=50, size=4, strategy='sliced', axis='y')
        info = decomp.get_rank_info(0)

        assert info.local_shape[0] == 50  # full z
        assert info.local_shape[2] == 50  # full x
        assert 'y_lower' in info.neighbors

    def test_x_axis_decomposition(self):
        """Decompose along x-axis."""
        decomp = DomainDecomposition(N=50, size=4, strategy='sliced', axis='x')
        info = decomp.get_rank_info(0)

        assert info.local_shape[0] == 50  # full z
        assert info.local_shape[1] == 50  # full y
        assert 'x_lower' in info.neighbors


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_single_rank(self):
        """Single rank gets entire interior."""
        decomp = DomainDecomposition(N=50, size=1, strategy='sliced')
        info = decomp.get_rank_info(0)

        assert info.local_shape[0] == 48
        assert info.n_neighbors == 0

    def test_invalid_strategy(self):
        """Unknown strategy raises ValueError."""
        with pytest.raises(ValueError):
            DomainDecomposition(N=50, size=4, strategy='invalid')

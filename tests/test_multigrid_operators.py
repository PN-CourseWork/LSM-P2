"""Tests for multigrid operators (restrict, prolong).

These tests verify the fundamental properties of the grid transfer operators
using vertex-centered alignment: coarse[i] <-> fine[2*i].
"""

import numpy as np
import pytest
from src.Poisson.multigrid_operators import restrict, prolong


class TestRestrict:
    """Tests for the restriction operator (fine -> coarse)."""

    def test_constant_field_preserved(self):
        """Restricting a constant field should preserve the value."""
        # For multigrid: N_coarse = (N_fine - 1) // 2 + 1
        N_fine = 9  # -> N_coarse = 5
        N_coarse = (N_fine - 1) // 2 + 1

        fine = np.ones((N_fine, N_fine, N_fine)) * 7.0
        coarse = np.zeros((N_coarse, N_coarse, N_coarse))

        restrict(fine, coarse)

        # Interior should be preserved
        np.testing.assert_allclose(
            coarse[1:-1, 1:-1, 1:-1],
            7.0,
            rtol=1e-14,
            err_msg="Constant field not preserved by restriction"
        )

    def test_vertex_centered_alignment(self):
        """Verify coarse[i] samples from fine[2*i]."""
        N_fine = 9
        N_coarse = 5

        # Set fine grid with known pattern: fine[i,j,k] = i + j + k
        fine = np.zeros((N_fine, N_fine, N_fine))
        for i in range(N_fine):
            for j in range(N_fine):
                for k in range(N_fine):
                    fine[i, j, k] = float(i + j + k)

        coarse = np.zeros((N_coarse, N_coarse, N_coarse))
        restrict(fine, coarse)

        # For injection (boundary cells), coarse[i] = fine[2*i]
        # For full weighting, coarse[i] ≈ fine[2*i] for smooth fields
        # The linear field i+j+k should be exactly preserved
        for i in range(1, N_coarse - 1):
            for j in range(1, N_coarse - 1):
                for k in range(1, N_coarse - 1):
                    expected = float(2*i + 2*j + 2*k)
                    np.testing.assert_allclose(
                        coarse[i, j, k],
                        expected,
                        rtol=1e-10,
                        err_msg=f"Vertex alignment failed at ({i},{j},{k})"
                    )

    def test_linear_field_exact(self):
        """Full weighting exactly preserves linear fields."""
        N_fine = 17
        N_coarse = 9

        # Linear field: u(x,y,z) = x + 2y + 3z
        fine = np.zeros((N_fine, N_fine, N_fine))
        for i in range(N_fine):
            for j in range(N_fine):
                for k in range(N_fine):
                    fine[i, j, k] = i + 2*j + 3*k

        coarse = np.zeros((N_coarse, N_coarse, N_coarse))
        restrict(fine, coarse)

        # Linear field should be exactly preserved
        for i in range(1, N_coarse - 1):
            for j in range(1, N_coarse - 1):
                for k in range(1, N_coarse - 1):
                    expected = 2*i + 4*j + 6*k  # = 2*(i + 2*j + 3*k)
                    np.testing.assert_allclose(
                        coarse[i, j, k],
                        expected,
                        rtol=1e-10,
                        err_msg=f"Linear field not preserved at ({i},{j},{k})"
                    )


class TestProlong:
    """Tests for the prolongation operator (coarse -> fine)."""

    def test_constant_field_preserved(self):
        """Prolonging a constant field should preserve the value."""
        N_coarse = 5
        N_fine = 9

        coarse = np.ones((N_coarse, N_coarse, N_coarse)) * 3.0
        fine = np.zeros((N_fine, N_fine, N_fine))

        prolong(coarse, fine)

        # Interior should be preserved
        np.testing.assert_allclose(
            fine[1:-1, 1:-1, 1:-1],
            3.0,
            rtol=1e-14,
            err_msg="Constant field not preserved by prolongation"
        )

    def test_vertex_centered_alignment(self):
        """Verify fine[2*i] = coarse[i] at coincident nodes."""
        N_coarse = 5
        N_fine = 9

        # Set coarse grid with known pattern
        coarse = np.zeros((N_coarse, N_coarse, N_coarse))
        for i in range(N_coarse):
            for j in range(N_coarse):
                for k in range(N_coarse):
                    coarse[i, j, k] = float(i * 100 + j * 10 + k)

        fine = np.zeros((N_fine, N_fine, N_fine))
        prolong(coarse, fine)

        # At even indices, fine should equal coarse
        for i in range(N_coarse - 1):
            for j in range(N_coarse - 1):
                for k in range(N_coarse - 1):
                    expected = coarse[i, j, k]
                    np.testing.assert_allclose(
                        fine[2*i, 2*j, 2*k],
                        expected,
                        rtol=1e-14,
                        err_msg=f"Vertex alignment failed: fine[{2*i},{2*j},{2*k}] != coarse[{i},{j},{k}]"
                    )

    def test_midpoint_interpolation(self):
        """Verify midpoints are interpolated correctly."""
        N_coarse = 5
        N_fine = 9

        coarse = np.zeros((N_coarse, N_coarse, N_coarse))
        # Set two adjacent points to test interpolation
        coarse[1, 1, 1] = 0.0
        coarse[2, 1, 1] = 10.0
        coarse[1, 2, 1] = 20.0
        coarse[1, 1, 2] = 30.0

        fine = np.zeros((N_fine, N_fine, N_fine))
        prolong(coarse, fine)

        # Midpoint in z direction: fine[3,2,2] = 0.5*(coarse[1,1,1] + coarse[2,1,1])
        np.testing.assert_allclose(fine[3, 2, 2], 5.0, rtol=1e-14)
        # Midpoint in y direction
        np.testing.assert_allclose(fine[2, 3, 2], 10.0, rtol=1e-14)
        # Midpoint in x direction
        np.testing.assert_allclose(fine[2, 2, 3], 15.0, rtol=1e-14)

    def test_linear_field_exact(self):
        """Trilinear interpolation exactly preserves linear fields."""
        N_coarse = 5
        N_fine = 9

        # Linear field on coarse: u = i + 2*j + 3*k
        coarse = np.zeros((N_coarse, N_coarse, N_coarse))
        for i in range(N_coarse):
            for j in range(N_coarse):
                for k in range(N_coarse):
                    coarse[i, j, k] = i + 2*j + 3*k

        fine = np.zeros((N_fine, N_fine, N_fine))
        prolong(coarse, fine)

        # Fine field should also be linear: u = i/2 + j + 3*k/2
        for i in range(1, N_fine - 1):
            for j in range(1, N_fine - 1):
                for k in range(1, N_fine - 1):
                    expected = i/2 + j + 1.5*k
                    np.testing.assert_allclose(
                        fine[i, j, k],
                        expected,
                        rtol=1e-10,
                        err_msg=f"Linear field not preserved at fine[{i},{j},{k}]"
                    )


class TestRestrictProlongConsistency:
    """Tests verifying restrict and prolong are consistent."""

    def test_restrict_then_prolong_constant(self):
        """Restrict then prolong should preserve a constant field."""
        N_fine = 9
        N_coarse = 5

        original = np.ones((N_fine, N_fine, N_fine)) * 5.0
        coarse = np.zeros((N_coarse, N_coarse, N_coarse))
        restored = np.zeros((N_fine, N_fine, N_fine))

        restrict(original, coarse)
        prolong(coarse, restored)

        # Interior should match (boundaries may differ due to boundary handling)
        np.testing.assert_allclose(
            restored[2:-2, 2:-2, 2:-2],
            original[2:-2, 2:-2, 2:-2],
            rtol=1e-14,
            err_msg="Restrict->prolong changed constant field"
        )

    def test_coarse_correction_additive(self):
        """Verify that prolongation of error correction is additive."""
        N_fine = 9
        N_coarse = 5

        # Simulate a V-cycle: fine solution + prolonged coarse correction
        u_fine = np.random.rand(N_fine, N_fine, N_fine)
        e_coarse = np.random.rand(N_coarse, N_coarse, N_coarse) * 0.1

        correction = np.zeros((N_fine, N_fine, N_fine))
        prolong(e_coarse, correction)

        u_corrected = u_fine.copy()
        u_corrected[1:-1, 1:-1, 1:-1] += correction[1:-1, 1:-1, 1:-1]

        # Verify the correction was added
        assert not np.allclose(u_corrected, u_fine), "Correction should change the solution"


class TestGridSizes:
    """Tests for proper handling of various grid sizes."""

    @pytest.mark.parametrize("N_fine", [5, 9, 17, 33, 65])
    def test_multigrid_compatible_sizes(self, N_fine):
        """Test restriction/prolongation with multigrid-compatible sizes (2^k + 1)."""
        N_coarse = (N_fine - 1) // 2 + 1

        fine = np.random.rand(N_fine, N_fine, N_fine)
        coarse = np.zeros((N_coarse, N_coarse, N_coarse))
        fine_restored = np.zeros((N_fine, N_fine, N_fine))

        restrict(fine, coarse)
        prolong(coarse, fine_restored)

        # Just verify no crashes and shapes are correct
        assert coarse.shape == (N_coarse, N_coarse, N_coarse)
        assert fine_restored.shape == (N_fine, N_fine, N_fine)

    def test_shape_mismatch_graceful(self):
        """Test that mismatched shapes don't crash (operators use bounds checking)."""
        fine = np.ones((9, 9, 9))
        coarse = np.zeros((3, 3, 3))  # Wrong size

        # Should not crash, just do what it can
        restrict(fine, coarse)
        # Values might not be meaningful but shouldn't crash


class TestBoundaryHandling:
    """Tests for proper boundary/halo handling."""

    def test_boundaries_not_modified_by_restrict(self):
        """Restriction should only modify interior of coarse grid."""
        N_fine = 9
        N_coarse = 5

        fine = np.ones((N_fine, N_fine, N_fine))
        coarse = np.zeros((N_coarse, N_coarse, N_coarse))

        # Mark boundaries
        coarse[0, :, :] = -999
        coarse[-1, :, :] = -999
        coarse[:, 0, :] = -999
        coarse[:, -1, :] = -999
        coarse[:, :, 0] = -999
        coarse[:, :, -1] = -999

        restrict(fine, coarse)

        # Boundaries should still be -999 (operators don't touch index 0 or -1)
        assert coarse[0, 2, 2] == -999, "Restriction modified boundary"
        assert coarse[-1, 2, 2] == -999, "Restriction modified boundary"

    def test_zero_boundaries_preserved(self):
        """Zero Dirichlet boundaries should be preserved."""
        N_fine = 9
        N_coarse = 5

        # Fine grid with zero boundaries
        fine = np.ones((N_fine, N_fine, N_fine))
        fine[0, :, :] = 0
        fine[-1, :, :] = 0
        fine[:, 0, :] = 0
        fine[:, -1, :] = 0
        fine[:, :, 0] = 0
        fine[:, :, -1] = 0

        coarse = np.zeros((N_coarse, N_coarse, N_coarse))
        restrict(fine, coarse)

        # Interior should be 1 (from full weighting of interior 1s)
        # At i=1, fi=2, stencil uses fine[1:4] which includes boundary
        # So near-boundary values will be affected
        assert coarse[2, 2, 2] == 1.0, "Interior constant not preserved"

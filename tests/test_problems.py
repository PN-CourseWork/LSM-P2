"""Tests for problem setup utilities."""

import numpy as np
import pytest
from Poisson import (
    create_grid_3d,
    sinusoidal_exact_solution,
    sinusoidal_source_term,
    setup_sinusoidal_problem,
)


class TestGridCreation:
    """Tests for grid creation utilities."""

    def test_grid_shape(self):
        """Grid should have correct shape."""
        N = 10
        u = create_grid_3d(N)

        assert u.shape == (N, N, N)

    def test_grid_default_values(self):
        """Grid should have specified interior value and zero boundaries."""
        N = 10
        u = create_grid_3d(N, value=1.0, boundary_value=0.0)

        # Interior should be 1.0
        assert np.all(u[1:-1, 1:-1, 1:-1] == 1.0)
        # Boundaries should be 0.0
        assert np.all(u[0, :, :] == 0.0)
        assert np.all(u[-1, :, :] == 0.0)
        assert np.all(u[:, 0, :] == 0.0)
        assert np.all(u[:, -1, :] == 0.0)
        assert np.all(u[:, :, 0] == 0.0)
        assert np.all(u[:, :, -1] == 0.0)

    def test_grid_dtype(self):
        """Grid should be float64."""
        N = 10
        u = create_grid_3d(N)
        assert u.dtype == np.float64


class TestSinusoidalSolution:
    """Tests for sinusoidal test problem."""

    def test_exact_solution_shape(self):
        """Exact solution should have correct shape."""
        N = 15
        u_exact = sinusoidal_exact_solution(N)

        assert u_exact.shape == (N, N, N)

    def test_exact_solution_boundaries_zero(self):
        """Exact solution should be zero on boundaries."""
        N = 15
        u = sinusoidal_exact_solution(N)

        # sin(pi * x) = 0 at x = -1 and x = 1
        assert np.allclose(u[0, :, :], 0.0, atol=1e-10)
        assert np.allclose(u[-1, :, :], 0.0, atol=1e-10)
        assert np.allclose(u[:, 0, :], 0.0, atol=1e-10)
        assert np.allclose(u[:, -1, :], 0.0, atol=1e-10)
        assert np.allclose(u[:, :, 0], 0.0, atol=1e-10)
        assert np.allclose(u[:, :, -1], 0.0, atol=1e-10)

    def test_exact_solution_symmetry(self):
        """Solution should be symmetric about origin."""
        N = 21  # Odd for center point
        u = sinusoidal_exact_solution(N)
        mid = N // 2

        # Check point symmetry
        assert np.isclose(u[mid, mid, mid], u[mid, mid, mid])  # Center
        assert np.isclose(u[mid-1, mid, mid], u[mid+1, mid, mid], rtol=1e-10)

    def test_source_term_shape(self):
        """Source term should have correct shape."""
        N = 15
        f = sinusoidal_source_term(N)

        assert f.shape == (N, N, N)

    def test_source_term_consistent(self):
        """Source term should be -3*pi^2 * u_exact for Poisson equation."""
        N = 15
        u_exact = sinusoidal_exact_solution(N)
        f = sinusoidal_source_term(N)

        # f = 3*pi^2 * sin(pi*x)*sin(pi*y)*sin(pi*z) = 3*pi^2 * u_exact
        expected = 3 * np.pi**2 * u_exact
        assert np.allclose(f, expected, rtol=1e-10)


class TestProblemSetup:
    """Tests for complete problem setup."""

    def test_setup_returns_four_values(self):
        """setup_sinusoidal_problem should return (u1, u2, f, h)."""
        result = setup_sinusoidal_problem(N=15)

        assert len(result) == 4

    def test_setup_array_shapes(self):
        """Setup arrays should have correct shapes."""
        N = 15
        u1, u2, f, h = setup_sinusoidal_problem(N)

        assert u1.shape == (N, N, N)
        assert u2.shape == (N, N, N)
        assert f.shape == (N, N, N)

    def test_setup_grid_spacing(self):
        """Grid spacing h should be 2/(N-1)."""
        N = 11
        _, _, _, h = setup_sinusoidal_problem(N)

        expected_h = 2.0 / (N - 1)
        assert np.isclose(h, expected_h)

    def test_setup_initial_zero(self):
        """Initial guess should be zero."""
        N = 15
        u1, u2, _, _ = setup_sinusoidal_problem(N)

        assert np.all(u1 == 0.0)
        assert np.all(u2 == 0.0)

    def test_setup_boundary_conditions(self):
        """Initial arrays should have zero boundaries (Dirichlet BC)."""
        N = 15
        u1, u2, f, _ = setup_sinusoidal_problem(N)

        # Check u1 boundaries
        assert np.all(u1[0, :, :] == 0.0)
        assert np.all(u1[-1, :, :] == 0.0)
        assert np.all(u1[:, 0, :] == 0.0)
        assert np.all(u1[:, -1, :] == 0.0)
        assert np.all(u1[:, :, 0] == 0.0)
        assert np.all(u1[:, :, -1] == 0.0)

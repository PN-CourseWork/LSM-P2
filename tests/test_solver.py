"""Tests for the JacobiPoisson solver."""

import numpy as np
import pytest
from Poisson import JacobiPoisson


@pytest.fixture
def solver():
    """Basic solver for quick tests."""
    s = JacobiPoisson(N=15, omega=1.0, max_iter=100, tolerance=0.0)
    s.solve()
    return s


@pytest.fixture
def converged_solver():
    """Solver run to convergence."""
    s = JacobiPoisson(N=20, omega=1.0, max_iter=10000, tolerance=1e-8)
    s.solve()
    return s


class TestSolverBasics:
    """Core solver functionality."""

    def test_converges(self, converged_solver):
        """Solver should converge and have low error."""
        assert converged_solver.results.converged
        assert converged_solver.compute_l2_error() < 0.1

    def test_reduces_residual(self, solver):
        """Residual should decrease over iterations."""
        r = solver.timeseries.residual_history
        assert r[-1] < r[0]

    def test_timing_recorded(self, solver):
        """Should record timing data."""
        assert len(solver.timeseries.compute_times) == 100
        assert all(t >= 0 for t in solver.timeseries.compute_times)

    def test_boundary_conditions(self, solver):
        """Dirichlet BCs (zero) should be preserved."""
        u = solver.u_global
        for face in [u[0], u[-1], u[:,0], u[:,-1], u[:,:,0], u[:,:,-1]]:
            assert np.allclose(face, 0.0)

    def test_numba_kernel(self):
        """Numba kernel should work."""
        s = JacobiPoisson(N=15, omega=1.0, max_iter=50, tolerance=0.0, use_numba=True)
        s.warmup()
        s.solve()
        assert s.results.iterations == 50

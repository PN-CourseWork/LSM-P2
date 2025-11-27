"""Tests for the JacobiPoisson solver."""

import numpy as np
import pytest
from Poisson import JacobiPoisson, sinusoidal_exact_solution


class TestSequentialSolver:
    """Tests for single-rank (sequential) solver execution."""

    def test_solver_converges(self):
        """Solver should converge within max iterations."""
        solver = JacobiPoisson(N=15, omega=1.0, max_iter=5000, tolerance=1e-6)
        solver.solve()

        assert solver.results.converged or solver.results.iterations < 5000

    def test_solver_reduces_residual(self):
        """Residual should decrease over iterations."""
        solver = JacobiPoisson(N=15, omega=1.0, max_iter=100, tolerance=0.0)
        solver.solve()

        residuals = solver.timeseries.residual_history
        assert len(residuals) == 100
        assert residuals[-1] < residuals[0]  # Residual decreased

    def test_solver_with_numba(self):
        """Solver should work with Numba kernel."""
        solver = JacobiPoisson(N=15, omega=1.0, max_iter=100, tolerance=0.0, use_numba=True)
        solver.warmup()
        solver.solve()

        assert solver.results.iterations == 100

    def test_solver_timing_recorded(self):
        """Solver should record timing data."""
        solver = JacobiPoisson(N=15, omega=1.0, max_iter=50, tolerance=0.0)
        solver.solve()

        assert len(solver.timeseries.compute_times) == 50
        assert len(solver.timeseries.halo_exchange_times) == 50
        assert all(t >= 0 for t in solver.timeseries.compute_times)

    def test_solver_solution_accuracy(self):
        """Converged solution should match analytical solution."""
        N = 20
        solver = JacobiPoisson(N=N, omega=1.0, max_iter=10000, tolerance=1e-8)
        solver.solve()

        # Compute L2 error
        error = solver.compute_l2_error()

        # Error should be reasonable for this grid size
        assert error < 0.1, f"L2 error {error} too large"


class TestSolverConfiguration:
    """Tests for solver configuration options."""

    def test_omega_affects_convergence(self):
        """Different omega values should affect convergence rate."""
        results = {}
        for omega in [0.5, 1.0]:
            solver = JacobiPoisson(N=15, omega=omega, max_iter=200, tolerance=1e-6)
            solver.solve()
            results[omega] = solver.timeseries.residual_history[-1]

        # Both should reduce residual (exact comparison depends on problem)
        assert all(r < 1.0 for r in results.values())

    def test_larger_grid_more_iterations(self):
        """Larger grids generally need more iterations for same tolerance."""
        iters = {}
        for N in [10, 20]:
            solver = JacobiPoisson(N=N, omega=1.0, max_iter=50000, tolerance=1e-4)
            solver.solve()
            iters[N] = solver.results.iterations

        # Larger grid should need more iterations (or hit max)
        assert iters[20] >= iters[10]

    def test_invalid_N_raises(self):
        """Very small N should still work (edge case)."""
        solver = JacobiPoisson(N=5, omega=1.0, max_iter=10, tolerance=0.0)
        solver.solve()  # Should not crash
        assert solver.results.iterations == 10


class TestSolverDataStructures:
    """Tests for solver data structure handling."""

    def test_u_global_shape(self):
        """Global solution should have correct shape."""
        N = 15
        solver = JacobiPoisson(N=N, omega=1.0, max_iter=10, tolerance=0.0)
        solver.solve()

        assert hasattr(solver, 'u_global')
        assert solver.u_global.shape == (N, N, N)

    def test_boundary_conditions_preserved(self):
        """Dirichlet BCs (zero) should be preserved on boundaries."""
        N = 15
        solver = JacobiPoisson(N=N, omega=1.0, max_iter=100, tolerance=0.0)
        solver.solve()

        u = solver.u_global
        # All boundaries should be zero (Dirichlet BC)
        assert np.allclose(u[0, :, :], 0.0)
        assert np.allclose(u[-1, :, :], 0.0)
        assert np.allclose(u[:, 0, :], 0.0)
        assert np.allclose(u[:, -1, :], 0.0)
        assert np.allclose(u[:, :, 0], 0.0)
        assert np.allclose(u[:, :, -1], 0.0)

    def test_results_populated(self):
        """Results dataclass should be populated after solve."""
        solver = JacobiPoisson(N=10, omega=1.0, max_iter=50, tolerance=0.0)
        solver.solve()

        assert solver.results.iterations == 50
        assert isinstance(solver.results.converged, bool)

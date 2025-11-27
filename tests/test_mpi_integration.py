"""MPI integration tests using the subprocess runner.

These tests spawn actual MPI processes to verify end-to-end distributed execution.
Each configuration is run once and results are reused across assertions.
"""

import numpy as np
import pytest
from Poisson import run_solver


# =============================================================================
# Test Configuration
# =============================================================================

RANK_CONFIGS = [
    (1, "sliced"),
    (2, "sliced"),
    (4, "sliced"),
    (8, "cubic"),
]

CONVERGENCE_GRID_SIZES = [15, 25, 45]

COMMUNICATORS = ["numpy", "custom"]

DECOMPOSITIONS = ["sliced", "cubic"]


# =============================================================================
# Fixtures - run each config once, reuse results
# =============================================================================

@pytest.fixture(scope="module")
def rank_results():
    """Run solver for different rank/strategy configs."""
    results = {}
    for n_ranks, strategy in RANK_CONFIGS:
        result = run_solver(N=25, n_ranks=n_ranks, strategy=strategy, max_iter=10000, tol=1e-8, validate=True)
        results[(n_ranks, strategy)] = result
    return results


@pytest.fixture(scope="module")
def convergence_results():
    """Run solver at multiple grid sizes to test O(h²) convergence."""
    results = {}
    for N in CONVERGENCE_GRID_SIZES:
        result = run_solver(N=N, n_ranks=2, strategy="sliced", max_iter=50000, tol=1e-10, validate=True)
        results[N] = result
    return results


@pytest.fixture(scope="module")
def communicator_results():
    """Run solver with different communicators."""
    results = {}
    for comm_type in COMMUNICATORS:
        result = run_solver(N=20, n_ranks=2, strategy="sliced", communicator=comm_type, max_iter=500, tol=0.0, validate=True)
        results[comm_type] = result
    return results


@pytest.fixture(scope="module")
def decomposition_results():
    """Run solver with sliced vs cubic decomposition."""
    return {
        strategy: run_solver(N=24, n_ranks=8, strategy=strategy, max_iter=5000, tol=1e-8, validate=True)
        for strategy in DECOMPOSITIONS
    }


# =============================================================================
# Tests - assertions on cached results
# =============================================================================

class TestMPIExecution:
    """Tests for multi-rank MPI execution."""

    @pytest.mark.parametrize("n_ranks,strategy", RANK_CONFIGS)
    def test_no_errors(self, rank_results, n_ranks, strategy):
        """Solver should complete without errors."""
        result = rank_results[(n_ranks, strategy)]
        assert "error" not in result, f"Solver failed: {result.get('error')}"

    @pytest.mark.parametrize("n_ranks,strategy", RANK_CONFIGS)
    def test_converged(self, rank_results, n_ranks, strategy):
        """Solver should converge."""
        result = rank_results[(n_ranks, strategy)]
        assert result["converged"], f"Did not converge in {result['iterations']} iterations"


class TestMPIAccuracy:
    """Tests for solution accuracy across ranks."""

    @pytest.mark.parametrize("n_ranks,strategy", RANK_CONFIGS)
    def test_accuracy(self, rank_results, n_ranks, strategy):
        """Solution should match analytical solution."""
        result = rank_results[(n_ranks, strategy)]
        assert result["final_error"] < 0.1, f"L2 error {result['final_error']} too large"

    def test_ranks_produce_consistent_error(self, rank_results):
        """Different rank counts should produce similar errors."""
        sliced_configs = [(n, s) for n, s in RANK_CONFIGS if s == "sliced"]
        errors = [rank_results[cfg]["final_error"] for cfg in sliced_configs]
        assert max(errors) / min(errors) < 1.1, f"Errors diverge: {errors}"

    def test_convergence_order(self, convergence_results):
        """MPI solver should exhibit O(h²) convergence."""
        # Check all runs succeeded
        for N in CONVERGENCE_GRID_SIZES:
            assert "error" not in convergence_results[N], f"N={N} failed: {convergence_results[N].get('error')}"

        errors = [convergence_results[N]["final_error"] for N in CONVERGENCE_GRID_SIZES]
        h = [2.0 / (N - 1) for N in CONVERGENCE_GRID_SIZES]

        # Compute convergence order: error ~ h^p => log(e1/e2) / log(h1/h2) = p
        orders = []
        for i in range(len(errors) - 1):
            order = np.log(errors[i] / errors[i + 1]) / np.log(h[i] / h[i + 1])
            orders.append(order)

        avg_order = np.mean(orders)
        assert 1.8 < avg_order < 2.5, f"Expected O(h²), got order {avg_order:.2f} (orders: {orders})"


class TestCommunicators:
    """Tests for different halo exchange implementations."""

    @pytest.mark.parametrize("communicator", COMMUNICATORS)
    def test_no_errors(self, communicator_results, communicator):
        """Communicator should work without errors."""
        result = communicator_results[communicator]
        assert "error" not in result, f"Solver failed: {result.get('error')}"

    def test_produce_same_iterations(self, communicator_results):
        """Both communicators should produce identical iteration counts."""
        iters = [communicator_results[c]["iterations"] for c in COMMUNICATORS]
        assert len(set(iters)) == 1, f"Iteration counts differ: {dict(zip(COMMUNICATORS, iters))}"


class TestDecompositionStrategies:
    """Tests for different decomposition strategies."""

    @pytest.mark.parametrize("strategy", DECOMPOSITIONS)
    def test_no_errors(self, decomposition_results, strategy):
        """Decomposition should work without errors."""
        result = decomposition_results[strategy]
        assert "error" not in result, f"Solver failed: {result.get('error')}"
        assert result["decomposition"] == strategy

    def test_same_accuracy(self, decomposition_results):
        """Decomposition strategies should produce similar accuracy."""
        errors = [decomposition_results[s]["final_error"] for s in DECOMPOSITIONS]
        ratio = max(errors) / min(errors)
        assert ratio < 1.2, f"Errors differ: {dict(zip(DECOMPOSITIONS, errors))}"

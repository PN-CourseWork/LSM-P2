"""MPI integration tests for the Multigrid solver."""

import numpy as np
import pytest
from Poisson import run_solver


MULTIGRID_CASES = [
    (2, "numpy"),
    (4, "numpy"),
]


@pytest.fixture(scope="module")
def multigrid_results():
    """Run Multigrid solver across rank counts and strategies."""
    results = {}
    for ranks, communicator in MULTIGRID_CASES:
        results[(ranks, communicator)] = run_solver(
            N=33,  # 32+1 so three-level V-cycle (33 -> 17 -> 9)
            n_ranks=ranks,
            solver_type="multigrid",
            strategy="sliced",
            communicator=communicator,
            levels=3,
            n_smooth=3,
            max_iter=50,
            tol=1e-6,
            validate=True,
        )
    return results


@pytest.fixture(scope="module")
def multigrid_convergence():
    """Compute multigrid errors for several mesh sizes (fixed communicator/ranks)."""
    Ns = [17, 33, 65]  # 16->32->64 grids (+1) coarsen cleanly
    return {
        N: run_solver(
            N=N,
            n_ranks=4,
            solver_type="multigrid",
            strategy="sliced",
            communicator="numpy",
            levels=3,
            n_smooth=3,
            max_iter=50,
            tol=1e-6,
            validate=True,
        )
        for N in Ns
    }


@pytest.mark.parametrize("ranks,communicator", MULTIGRID_CASES)
def test_multigrid_runs_without_error(multigrid_results, ranks, communicator):
    """Multigrid should execute via mpiexec and produce results."""
    result = multigrid_results[(ranks, communicator)]
    assert "error" not in result, f"Failed: {result.get('error')}"
    assert result["iterations"] > 0
    assert np.isfinite(result.get("final_error", np.inf))


@pytest.mark.parametrize("ranks,communicator", MULTIGRID_CASES)
def test_multigrid_converges_reasonably(multigrid_results, ranks, communicator):
    """Sanity check that multigrid converges to a small error."""
    r = multigrid_results[(ranks, communicator)]
    assert r["converged"]
    assert r["final_error"] < 0.1


def test_multigrid_convergence_order(multigrid_convergence):
    """Multigrid should retain ~second-order accuracy."""
    Ns = [17, 33, 65]
    errors = []
    for N in Ns:
        result = multigrid_convergence[N]
        assert "error" not in result, f"Failed: {result.get('error')}"
        assert result["converged"]
        errors.append(result["final_error"])

    hs = [2.0 / (N - 1) for N in Ns]
    orders = [
        np.log(errors[i] / errors[i + 1]) / np.log(hs[i] / hs[i + 1])
        for i in range(len(Ns) - 1)
    ]
    for p in orders:
        assert 1.8 < p < 2.5, f"Expected ~2nd order, got {p:.2f}"

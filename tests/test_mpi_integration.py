"""MPI integration tests - spawn actual MPI processes via run_solver."""

import numpy as np
import pytest
from Poisson import run_solver


# Run solver once per config, reuse results
@pytest.fixture(scope="module")
def mpi_results():
    """Run all MPI configurations once."""
    return {
        (2, "sliced"): run_solver(
            N=25, n_ranks=2, strategy="sliced", max_iter=10000, tol=1e-8, validate=True
        ),
        (4, "sliced"): run_solver(
            N=25, n_ranks=4, strategy="sliced", max_iter=10000, tol=1e-8, validate=True
        ),
        (8, "cubic"): run_solver(
            N=25, n_ranks=8, strategy="cubic", max_iter=10000, tol=1e-8, validate=True
        ),
        "numpy": run_solver(
            N=20, n_ranks=2, communicator="numpy", max_iter=500, tol=0.0
        ),
        "custom": run_solver(
            N=20, n_ranks=2, communicator="custom", max_iter=500, tol=0.0
        ),
        "convergence": {
            N: run_solver(N=N, n_ranks=2, max_iter=20000, tol=1e-8, validate=True)
            for N in [15, 25]
        },
    }


@pytest.mark.parametrize("config", [(2, "sliced"), (4, "sliced"), (8, "cubic")])
def test_mpi_runs_and_converges(mpi_results, config):
    """MPI solver should run without errors and converge."""
    r = mpi_results[config]
    assert "error" not in r, f"Failed: {r.get('error')}"
    assert r["converged"]
    assert r["final_error"] < 0.1


def test_ranks_produce_consistent_error(mpi_results):
    """Different rank counts should produce similar errors."""
    errors = [mpi_results[(n, "sliced")]["final_error"] for n in [2, 4]]
    assert max(errors) / min(errors) < 1.1


def test_convergence_order(mpi_results):
    """Should exhibit O(h²) convergence."""
    conv = mpi_results["convergence"]
    errors = [conv[N]["final_error"] for N in [15, 25]]
    h = [2.0 / (N - 1) for N in [15, 25]]
    order = np.log(errors[0] / errors[1]) / np.log(h[0] / h[1])
    assert 1.8 < order < 2.5, f"Expected ~2, got {order:.2f}"


@pytest.mark.parametrize("comm", ["numpy", "custom"])
def test_communicators_work(mpi_results, comm):
    """Both communicators should work."""
    assert "error" not in mpi_results[comm]


def test_communicators_same_iterations(mpi_results):
    """Communicators should produce identical iteration counts."""
    assert mpi_results["numpy"]["iterations"] == mpi_results["custom"]["iterations"]

"""FMG solver tests (mostly single-rank for speed)."""

import numpy as np
from Poisson import MultigridPoisson, run_solver


def test_fmg_converges_single_rank():
    """FMG should achieve discretization-level accuracy on a modest grid."""
    solver = MultigridPoisson(
        N=33,
        tolerance=1e-10,  # Low tolerance (FMG achieves ~h^2 accuracy)
        max_iter=20,
        min_coarse_size=3,
        n_smooth=4,
    )
    solver.fmg_solve(cycles=1)
    err = solver.compute_l2_error()
    # FMG achieves discretization accuracy (~h^2) in O(1) cycles
    # For N=33, h=2/32=0.0625, h^2 ≈ 0.004, so expect err < 0.01
    assert err is not None and err < 0.01


def test_fmg_runner_single_rank():
    """Runner helper should dispatch FMG via run_solver."""
    res = run_solver(
        N=33,
        n_ranks=1,
        solver_type="fmg",
        strategy="sliced",
        communicator="numpy",
        max_iter=20,
        n_smooth=4,
        tol=1e-4,
        validate=True,
    )
    assert "error" not in res
    assert res["converged"]
    assert res["final_error"] < 0.05

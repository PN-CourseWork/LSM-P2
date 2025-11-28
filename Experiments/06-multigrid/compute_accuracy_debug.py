"""
Debug script: accuracy and observed order for Multigrid V-cycle and FMG.

Runs on a single rank for larger grids and prints L2 errors and observed orders.
"""

import math
from Poisson import MultigridPoisson


def observed_orders(errors, Ns):
    hs = [2.0 / (N - 1) for N in Ns]
    return [
        math.log(errors[i] / errors[i + 1]) / math.log(hs[i] / hs[i + 1])
        for i in range(len(Ns) - 1)
    ]


def run_vcycle(Ns):
    errors = []
    print("\nV-cycle accuracy (single rank)")
    for N in Ns:
        solver = MultigridPoisson(
            N=N,
            n_smooth=4,
            tolerance=1e-8,
            max_iter=200,
            min_coarse_size=3,
            omega=0.8,
        )
        solver.solve()
        err = solver.compute_l2_error()
        errors.append(err)
        print(f"  N={N}: err={err:.3e}, iters={solver.results.iterations}")
    print("  orders:", observed_orders(errors, Ns))


def run_fmg(Ns):
    errors = []
    print("\nFMG accuracy (single cycle, single rank)")
    for N in Ns:
        solver = MultigridPoisson(
            N=N,
            n_smooth=6,  # more smoothing for single-cycle FMG
            tolerance=1e-8,
            max_iter=20,
            min_coarse_size=3,
            omega=1.0,
        )
        solver.fmg_solve(cycles=1)
        err = solver.compute_l2_error()
        errors.append(err)
        print(f"  N={N}: err={err:.3e}, iters={solver.results.iterations}")
    print("  orders:", observed_orders(errors, Ns))


if __name__ == "__main__":
    grids = [65, 129, 257]
    run_vcycle(grids)
    run_fmg(grids)

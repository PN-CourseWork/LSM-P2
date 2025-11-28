"""
Kernel Experiments
==================

1. Convergence validation with analytical solution
2. Fixed iteration performance benchmark
"""

import numpy as np
import pandas as pd
from dataclasses import asdict
from scipy.ndimage import laplace

from Poisson import problems, get_project_root
from Poisson.kernels import NumPyKernel, NumbaKernel

data_dir = get_project_root() / "data" / "01-kernels"
data_dir.mkdir(parents=True, exist_ok=True)

for old_file in data_dir.glob("*.parquet"):
    old_file.unlink()


def run_kernel(kernel, f, max_iter, track_algebraic=False):
    """Run kernel iterations, return final arrays and optionally track algebraic residual."""
    N = f.shape[0]
    u = np.zeros((N, N, N), dtype=np.float64)
    u_old = np.zeros((N, N, N), dtype=np.float64)

    if track_algebraic:
        kernel.timeseries.physical_errors = []
        h2 = kernel.parameters.h**2

    for _ in range(max_iter):
        kernel.step(u_old, u, f)

        if track_algebraic:
            Au = -laplace(u) / h2
            alg_res = np.max(np.abs(Au[1:-1, 1:-1, 1:-1] - f[1:-1, 1:-1, 1:-1]))
            kernel.timeseries.physical_errors.append(alg_res)

        u, u_old = u_old, u

    return u, u_old


def kernel_to_df(kernel, kernel_name, N, omega, **extra):
    """Convert kernel timeseries to DataFrame."""
    df = pd.DataFrame(asdict(kernel.timeseries))
    df["iteration"] = range(len(df))
    df["kernel"] = kernel_name
    df["N"] = N
    df["omega"] = omega
    for k, v in extra.items():
        df[k] = v
    return df


# Experiment 1: Convergence Validation
print("EXPERIMENT 1: Convergence Validation")

omega = 1.0
max_iter = 5000
all_dfs = []

for N in [25]:
    f = problems.sinusoidal_source_term(N)

    numpy_kernel = NumPyKernel(N=N, omega=omega, tolerance=0.0, max_iter=max_iter)
    numba_kernel = NumbaKernel(
        N=N, omega=omega, tolerance=0.0, max_iter=max_iter, numba_threads=4
    )
    numba_kernel.warmup()

    for name, kernel in [("numpy", numpy_kernel), ("numba", numba_kernel)]:
        run_kernel(kernel, f, max_iter, track_algebraic=True)
        all_dfs.append(kernel_to_df(kernel, name, N, omega, tolerance=0.0))

pd.concat(all_dfs, ignore_index=True).to_parquet(
    data_dir / "kernel_convergence.parquet", index=False
)


# Experiment 2: Fixed Iteration Benchmark
print("EXPERIMENT 2: Fixed Iteration Benchmark")

problem_sizes = [25, 50, 75, 100, 125, 150]
thread_counts = [1, 4, 6, 8]
max_iter = 200
all_dfs = []

# NumPy baseline
for N in problem_sizes:
    kernel = NumPyKernel(N=N, omega=omega, tolerance=0.0, max_iter=max_iter)
    f = np.ones((N, N, N), dtype=np.float64)
    run_kernel(kernel, f, max_iter)
    all_dfs.append(
        kernel_to_df(
            kernel, "numpy", N, omega, max_iter=max_iter, use_numba=False, num_threads=0
        )
    )

# Numba with thread scaling
for num_threads in thread_counts:
    for idx, N in enumerate(problem_sizes):
        kernel = NumbaKernel(
            N=N,
            omega=omega,
            tolerance=0.0,
            max_iter=max_iter,
            numba_threads=num_threads,
        )
        if idx == 0:
            kernel.warmup()
        f = np.ones((N, N, N), dtype=np.float64)
        run_kernel(kernel, f, max_iter)
        all_dfs.append(
            kernel_to_df(
                kernel,
                "numba",
                N,
                omega,
                max_iter=max_iter,
                use_numba=True,
                num_threads=num_threads,
            )
        )

pd.concat(all_dfs, ignore_index=True).to_parquet(
    data_dir / "kernel_benchmark.parquet", index=False
)

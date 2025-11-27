"""Tests for Jacobi iteration kernels."""

import numpy as np
import pytest
from Poisson import NumPyKernel, NumbaKernel, setup_sinusoidal_problem, sinusoidal_exact_solution


def run_iterations(kernel, u1, u2, f, n_iter):
    """Run n_iter Jacobi iterations, return final solution."""
    for i in range(n_iter):
        if i % 2 == 0:
            kernel.step(u1, u2, f)
        else:
            kernel.step(u2, u1, f)
    return u1 if n_iter % 2 == 0 else u2


def test_kernels_produce_identical_results():
    """NumPy and Numba kernels should produce identical results."""
    N = 16
    u1, u2, f, _ = setup_sinusoidal_problem(N)

    numpy_kernel = NumPyKernel(N=N, omega=1.0, tolerance=1e-10, max_iter=1000)
    numba_kernel = NumbaKernel(N=N, omega=1.0, tolerance=1e-10, max_iter=1000, numba_threads=1)
    numba_kernel.warmup()

    u_numpy = run_iterations(numpy_kernel, u1.copy(), u2.copy(), f.copy(), 10)
    u_numba = run_iterations(numba_kernel, u1.copy(), u2.copy(), f.copy(), 10)

    assert np.allclose(u_numpy, u_numba, atol=1e-10)


def test_convergence_order():
    """Jacobi kernel should achieve O(h²) convergence."""
    problem_sizes = [15, 25, 50]
    errors, h_values = [], []

    for N in problem_sizes:
        u1, u2, f, h = setup_sinusoidal_problem(N)
        exact = sinusoidal_exact_solution(N)
        kernel = NumPyKernel(N=N, omega=1.0, tolerance=1e-12, max_iter=50000)

        # Solve until converged
        for i in range(50000):
            if i % 2 == 0:
                res = kernel.step(u1, u2, f)
                u = u2
            else:
                res = kernel.step(u2, u1, f)
                u = u1
            if np.sqrt(res / (N-2)**3) < 1e-12:
                break

        l2_error = np.sqrt(np.mean((u[1:-1,1:-1,1:-1] - exact[1:-1,1:-1,1:-1])**2))
        errors.append(l2_error)
        h_values.append(h)

    # Calculate convergence rate
    rates = [np.log(errors[i]/errors[i+1]) / np.log(h_values[i]/h_values[i+1])
             for i in range(len(errors)-1)]
    avg_rate = np.mean(rates)

    assert abs(avg_rate - 2.0) < 0.2, f"Rate {avg_rate:.2f} not close to 2.0"


def test_boundary_preservation():
    """Kernels should not modify boundary conditions."""
    N = 16
    u1, u2, f, _ = setup_sinusoidal_problem(N)
    boundaries = [u1[0,:,:].copy(), u1[-1,:,:].copy(),
                  u1[:,0,:].copy(), u1[:,-1,:].copy(),
                  u1[:,:,0].copy(), u1[:,:,-1].copy()]

    kernel = NumPyKernel(N=N, omega=1.0, tolerance=1e-10, max_iter=100)
    u = run_iterations(kernel, u1, u2, f, 10)

    assert np.allclose(u[0,:,:], boundaries[0])
    assert np.allclose(u[-1,:,:], boundaries[1])
    assert np.allclose(u[:,0,:], boundaries[2])
    assert np.allclose(u[:,-1,:], boundaries[3])
    assert np.allclose(u[:,:,0], boundaries[4])
    assert np.allclose(u[:,:,-1], boundaries[5])

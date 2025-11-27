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


# Note: O(h²) convergence is tested in test_mpi_integration.py


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

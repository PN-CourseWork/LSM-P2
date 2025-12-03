"""Multigrid operators for 3D Poisson solver.

Simple operators using vertex-centered alignment: coarse[i] <-> fine[2*i].
Halo/boundary handling is the caller's responsibility.
"""

import numpy as np
from numba import njit, prange


@njit(parallel=True)
def restrict(fine: np.ndarray, coarse: np.ndarray):
    """
    Full weighting restriction (fine -> coarse).

    Vertex-centered alignment: coarse[i] corresponds to fine[2*i].
    Uses 27-point full weighting stencil where possible, injection at boundaries.

    Parameters
    ----------
    fine : ndarray
        Fine grid array (includes boundaries/halos)
    coarse : ndarray
        Coarse grid array (includes boundaries/halos), modified in-place
    """
    nz_c, ny_c, nx_c = coarse.shape
    nz_f, ny_f, nx_f = fine.shape

    for i in prange(1, nz_c - 1):
        for j in range(1, ny_c - 1):
            for k in range(1, nx_c - 1):
                fi, fj, fk = 2 * i, 2 * j, 2 * k

                # Check if full stencil fits
                if (
                    fi - 1 >= 0
                    and fi + 1 < nz_f
                    and fj - 1 >= 0
                    and fj + 1 < ny_f
                    and fk - 1 >= 0
                    and fk + 1 < nx_f
                ):
                    # Full weighting 27-point stencil
                    val = 8.0 * fine[fi, fj, fk]

                    val += 4.0 * (
                        fine[fi - 1, fj, fk]
                        + fine[fi + 1, fj, fk]
                        + fine[fi, fj - 1, fk]
                        + fine[fi, fj + 1, fk]
                        + fine[fi, fj, fk - 1]
                        + fine[fi, fj, fk + 1]
                    )

                    val += 2.0 * (
                        fine[fi - 1, fj - 1, fk]
                        + fine[fi - 1, fj + 1, fk]
                        + fine[fi + 1, fj - 1, fk]
                        + fine[fi + 1, fj + 1, fk]
                        + fine[fi - 1, fj, fk - 1]
                        + fine[fi - 1, fj, fk + 1]
                        + fine[fi + 1, fj, fk - 1]
                        + fine[fi + 1, fj, fk + 1]
                        + fine[fi, fj - 1, fk - 1]
                        + fine[fi, fj - 1, fk + 1]
                        + fine[fi, fj + 1, fk - 1]
                        + fine[fi, fj + 1, fk + 1]
                    )

                    val += 1.0 * (
                        fine[fi - 1, fj - 1, fk - 1]
                        + fine[fi - 1, fj - 1, fk + 1]
                        + fine[fi - 1, fj + 1, fk - 1]
                        + fine[fi - 1, fj + 1, fk + 1]
                        + fine[fi + 1, fj - 1, fk - 1]
                        + fine[fi + 1, fj - 1, fk + 1]
                        + fine[fi + 1, fj + 1, fk - 1]
                        + fine[fi + 1, fj + 1, fk + 1]
                    )

                    coarse[i, j, k] = val / 64.0
                else:
                    # Injection for boundary cases
                    if 0 <= fi < nz_f and 0 <= fj < ny_f and 0 <= fk < nx_f:
                        coarse[i, j, k] = fine[fi, fj, fk]


@njit(parallel=True)
def prolong(coarse: np.ndarray, fine: np.ndarray):
    """
    Trilinear interpolation prolongation (coarse -> fine).

    Vertex-centered alignment: fine[2*i] corresponds to coarse[i].
    Fills fine points at both even indices (2i) and odd indices (2i+1).

    Parameters
    ----------
    coarse : ndarray
        Coarse grid array (includes boundaries/halos)
    fine : ndarray
        Fine grid array (includes boundaries/halos), modified in-place
    """
    nz_c, ny_c, nx_c = coarse.shape
    nz_f, ny_f, nx_f = fine.shape

    for i in prange(0, nz_c - 1):
        for j in range(0, ny_c - 1):
            for k in range(0, nx_c - 1):
                fi, fj, fk = 2 * i, 2 * j, 2 * k

                if fi + 1 >= nz_f or fj + 1 >= ny_f or fk + 1 >= nx_f:
                    continue

                # Even-index point (coincides with coarse node)
                fine[fi, fj, fk] = coarse[i, j, k]

                # Odd-index points (midpoints)
                fine[fi + 1, fj, fk] = 0.5 * (coarse[i, j, k] + coarse[i + 1, j, k])
                fine[fi, fj + 1, fk] = 0.5 * (coarse[i, j, k] + coarse[i, j + 1, k])
                fine[fi, fj, fk + 1] = 0.5 * (coarse[i, j, k] + coarse[i, j, k + 1])

                # Face centers
                fine[fi + 1, fj + 1, fk] = 0.25 * (
                    coarse[i, j, k]
                    + coarse[i + 1, j, k]
                    + coarse[i, j + 1, k]
                    + coarse[i + 1, j + 1, k]
                )
                fine[fi + 1, fj, fk + 1] = 0.25 * (
                    coarse[i, j, k]
                    + coarse[i + 1, j, k]
                    + coarse[i, j, k + 1]
                    + coarse[i + 1, j, k + 1]
                )
                fine[fi, fj + 1, fk + 1] = 0.25 * (
                    coarse[i, j, k]
                    + coarse[i, j + 1, k]
                    + coarse[i, j, k + 1]
                    + coarse[i, j + 1, k + 1]
                )

                # Cell center
                fine[fi + 1, fj + 1, fk + 1] = 0.125 * (
                    coarse[i, j, k]
                    + coarse[i + 1, j, k]
                    + coarse[i, j + 1, k]
                    + coarse[i + 1, j + 1, k]
                    + coarse[i, j, k + 1]
                    + coarse[i + 1, j, k + 1]
                    + coarse[i, j + 1, k + 1]
                    + coarse[i + 1, j + 1, k + 1]
                )

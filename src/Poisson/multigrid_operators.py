"""Multigrid operators for 3D Poisson solver."""

import numpy as np
from numba import njit, prange


@njit(parallel=True)
def prolong_halo(coarse: np.ndarray, fine: np.ndarray):
    """
    Trilinear interpolation prolongation for halo-padded arrays.

    For arrays with halo cells at indices 0 and -1 in each dimension:
    - coarse[1] (first interior) physically aligns with fine[1] (first interior)
    - coarse[i] maps to fine[2*(i-1)+1] = fine[2*i-1] for i >= 1

    This differs from the standard prolong which assumes coarse[i] -> fine[2*i].
    """
    nz_c, ny_c, nx_c = coarse.shape
    nz_f, ny_f, nx_f = fine.shape

    # Iterate over coarse interior points (indices 1 to n-2)
    # and fill the corresponding fine points
    for i in prange(1, nz_c - 1):
        for j in range(1, ny_c - 1):
            for k in range(1, nx_c - 1):
                # Fine indices: fi = 2*(i-1)+1 = 2*i-1
                fi = 2 * i - 1
                fj = 2 * j - 1
                fk = 2 * k - 1

                # Check bounds
                if fi + 1 >= nz_f or fj + 1 >= ny_f or fk + 1 >= nx_f:
                    continue

                # 1. Center point (coarse node coincides with fine node)
                fine[fi, fj, fk] = coarse[i, j, k]

                # 2. Edge midpoints (interpolate between 2 coarse nodes)
                if fi + 1 < nz_f:
                    fine[fi + 1, fj, fk] = 0.5 * (coarse[i, j, k] + coarse[i + 1, j, k])
                if fj + 1 < ny_f:
                    fine[fi, fj + 1, fk] = 0.5 * (coarse[i, j, k] + coarse[i, j + 1, k])
                if fk + 1 < nx_f:
                    fine[fi, fj, fk + 1] = 0.5 * (coarse[i, j, k] + coarse[i, j, k + 1])

                # 3. Face centers (interpolate between 4 coarse nodes)
                if fi + 1 < nz_f and fj + 1 < ny_f:
                    fine[fi + 1, fj + 1, fk] = 0.25 * (
                        coarse[i, j, k]
                        + coarse[i + 1, j, k]
                        + coarse[i, j + 1, k]
                        + coarse[i + 1, j + 1, k]
                    )
                if fi + 1 < nz_f and fk + 1 < nx_f:
                    fine[fi + 1, fj, fk + 1] = 0.25 * (
                        coarse[i, j, k]
                        + coarse[i + 1, j, k]
                        + coarse[i, j, k + 1]
                        + coarse[i + 1, j, k + 1]
                    )
                if fj + 1 < ny_f and fk + 1 < nx_f:
                    fine[fi, fj + 1, fk + 1] = 0.25 * (
                        coarse[i, j, k]
                        + coarse[i, j + 1, k]
                        + coarse[i, j, k + 1]
                        + coarse[i, j + 1, k + 1]
                    )

                # 4. Cell center (interpolate between 8 coarse nodes)
                if fi + 1 < nz_f and fj + 1 < ny_f and fk + 1 < nx_f:
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


@njit(parallel=True)
def restrict_halo(fine: np.ndarray, coarse: np.ndarray):
    """
    Full weighting restriction for halo-padded arrays.

    For arrays with halo cells at indices 0 and -1 in each dimension:
    - fine[1] (first interior) physically aligns with coarse[1] (first interior)
    - fine[2*(i-1)+1] = fine[2*i-1] maps to coarse[i] for i >= 1

    This differs from the standard restrict which assumes fine[2*i] -> coarse[i].
    """
    nz_c, ny_c, nx_c = coarse.shape
    nz_f, ny_f, nx_f = fine.shape

    # Iterate over coarse interior points (indices 1 to n-2)
    for i in prange(1, nz_c - 1):
        for j in range(1, ny_c - 1):
            for k in range(1, nx_c - 1):
                # Fine indices: fi = 2*(i-1)+1 = 2*i-1
                fi = 2 * i - 1
                fj = 2 * j - 1
                fk = 2 * k - 1

                # Check bounds for full stencil
                if (
                    fi - 1 < 0
                    or fi + 1 >= nz_f
                    or fj - 1 < 0
                    or fj + 1 >= ny_f
                    or fk - 1 < 0
                    or fk + 1 >= nx_f
                ):
                    # Use injection for boundary cells
                    if 0 <= fi < nz_f and 0 <= fj < ny_f and 0 <= fk < nx_f:
                        coarse[i, j, k] = fine[fi, fj, fk]
                    continue

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


@njit(parallel=True)
def restrict(fine: np.ndarray, coarse: np.ndarray):
    """
    Full weighting restriction (fine -> coarse).

    Assumes vertex-centered alignment:
    Coarse node (i,j,k) corresponds to Fine node (2i, 2j, 2k).

    Handles boundary cases with injection where full stencil can't be applied.
    """
    nz_c, ny_c, nx_c = coarse.shape
    nz_f, ny_f, nx_f = fine.shape

    # Compute safe bounds for full 27-point stencil
    # For index i, we access fine[2*i-1 : 2*i+2], need 2*i+1 <= nz_f-1
    max_i_full = min(nz_c - 1, (nz_f - 2) // 2 + 1)
    max_j_full = min(ny_c - 1, (ny_f - 2) // 2 + 1)
    max_k_full = min(nx_c - 1, (nx_f - 2) // 2 + 1)

    # Bounds for injection (just need 2*i < nz_f)
    max_i_inject = min(nz_c - 1, (nz_f - 1) // 2 + 1)
    max_j_inject = min(ny_c - 1, (ny_f - 1) // 2 + 1)
    max_k_inject = min(nx_c - 1, (nx_f - 1) // 2 + 1)

    # First pass: full weighting where possible
    for i in prange(1, max_i_full):
        for j in range(1, max_j_full):
            for k in range(1, max_k_full):
                fi, fj, fk = 2 * i, 2 * j, 2 * k

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

    # Second pass: injection for boundary cells that couldn't use full stencil
    for i in prange(1, max_i_inject):
        for j in range(1, max_j_inject):
            for k in range(1, max_k_inject):
                # Skip cells already filled with full weighting
                if i < max_i_full and j < max_j_full and k < max_k_full:
                    continue

                fi, fj, fk = 2 * i, 2 * j, 2 * k
                # Use injection (direct copy) for boundary cells
                coarse[i, j, k] = fine[fi, fj, fk]


@njit(parallel=True)
def prolong(coarse: np.ndarray, fine: np.ndarray):
    """
    Trilinear interpolation prolongation (coarse -> fine).

    Assumes vertex-centered alignment:
    Fine node (2i, 2j, 2k) corresponds to Coarse node (i,j,k).

    Handles boundary cases where not all interpolation points are available.
    """
    nz_c, ny_c, nx_c = coarse.shape
    nz_f, ny_f, nx_f = fine.shape

    # Compute safe iteration bounds for full interpolation
    # Need 2*i+1 <= nz_f-1 and i+1 <= nz_c-1
    max_i = min((nz_f - 2) // 2 + 1, nz_c - 1)
    max_j = min((ny_f - 2) // 2 + 1, ny_c - 1)
    max_k = min((nx_f - 2) // 2 + 1, nx_c - 1)

    for i in prange(0, max_i):
        for j in range(0, max_j):
            for k in range(0, max_k):
                fi, fj, fk = 2 * i, 2 * j, 2 * k

                # 1. Center point (coincides with coarse node)
                fine[fi, fj, fk] = coarse[i, j, k]

                # 2. Odd-index points (between coarse nodes)
                fine[fi, fj, fk + 1] = 0.5 * (coarse[i, j, k] + coarse[i, j, k + 1])
                fine[fi, fj + 1, fk] = 0.5 * (coarse[i, j, k] + coarse[i, j + 1, k])
                fine[fi + 1, fj, fk] = 0.5 * (coarse[i, j, k] + coarse[i + 1, j, k])

                # 3. Face centers
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

                # 4. Cell center
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

    # Handle boundary cells that couldn't get full interpolation
    # Fill remaining even indices with direct injection
    max_i_even = min(nz_c, (nz_f + 1) // 2)
    max_j_even = min(ny_c, (ny_f + 1) // 2)
    max_k_even = min(nx_c, (nx_f + 1) // 2)

    for i in prange(0, max_i_even):
        for j in range(0, max_j_even):
            for k in range(0, max_k_even):
                fi, fj, fk = 2 * i, 2 * j, 2 * k
                # Only fill if within fine bounds and not already filled
                if fi < nz_f and fj < ny_f and fk < nx_f:
                    if i >= max_i or j >= max_j or k >= max_k:
                        fine[fi, fj, fk] = coarse[i, j, k]

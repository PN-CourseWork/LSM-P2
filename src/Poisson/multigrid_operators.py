"""Multigrid operators for 3D Poisson solver."""

import numpy as np
import numba
from numba import njit, prange

@njit(parallel=True)
def restrict(fine: np.ndarray, coarse: np.ndarray):
    """
    Full weighting restriction (fine -> coarse).
    
    Assumes vertex-centered alignment:
    Coarse node (i,j,k) corresponds to Fine node (2i, 2j, 2k).
    """
    # Iterate over interior of COARSE grid
    # Coarse interior: 1..Nc
    # Fine corresponding center: 2*i, 2*j, 2*k
    
    nz, ny, nx = coarse.shape
    
    for i in prange(1, nz - 1):
        for j in range(1, ny - 1):
            for k in range(1, nx - 1):
                # Map to fine grid coordinates
                fi = 2 * i
                fj = 2 * j
                fk = 2 * k
                
                # 27-point stencil (tensor product of 1/4 [1,2,1])
                # Weights: Center=8, Face=4, Edge=2, Corner=1
                # All divided by 64
                
                val = 0.0
                
                # Center (1 point) - weight 8
                val += 8.0 * fine[fi, fj, fk]
                
                # Face neighbors (6 points) - weight 4
                val += 4.0 * (fine[fi-1, fj, fk] + fine[fi+1, fj, fk] +
                              fine[fi, fj-1, fk] + fine[fi, fj+1, fk] +
                              fine[fi, fj, fk-1] + fine[fi, fj, fk+1])
                              
                # Edge neighbors (12 points) - weight 2
                val += 2.0 * (fine[fi-1, fj-1, fk] + fine[fi-1, fj+1, fk] +
                              fine[fi+1, fj-1, fk] + fine[fi+1, fj+1, fk] + # z-axis edges
                              fine[fi-1, fj, fk-1] + fine[fi-1, fj, fk+1] +
                              fine[fi+1, fj, fk-1] + fine[fi+1, fj, fk+1] + # y-axis edges
                              fine[fi, fj-1, fk-1] + fine[fi, fj-1, fk+1] +
                              fine[fi, fj+1, fk-1] + fine[fi, fj+1, fk+1])  # x-axis edges
                              
                # Corner neighbors (8 points) - weight 1
                val += 1.0 * (fine[fi-1, fj-1, fk-1] + fine[fi-1, fj-1, fk+1] +
                              fine[fi-1, fj+1, fk-1] + fine[fi-1, fj+1, fk+1] +
                              fine[fi+1, fj-1, fk-1] + fine[fi+1, fj-1, fk+1] +
                              fine[fi+1, fj+1, fk-1] + fine[fi+1, fj+1, fk+1])
                              
                coarse[i, j, k] = val / 64.0


@njit(parallel=True)
def prolong(coarse: np.ndarray, fine: np.ndarray):
    """
    Trilinear interpolation prolongation (coarse -> fine).
    
    Assumes vertex-centered alignment:
    Fine node (2i, 2j, 2k) corresponds to Coarse node (i,j,k).
    """
    # Iterate over interior of COARSE grid to fill FINE grid
    # This covers 1..Nc-1 on coarse
    # Fills 2..2Nc-2 on fine
    
    nz, ny, nx = coarse.shape
    
    for i in prange(0, nz - 1):  # Iterate from 0 to cover boundary conditions
        for j in range(0, ny - 1):
            for k in range(0, nx - 1):
                
                # Indices on fine grid
                fi, fj, fk = 2*i, 2*j, 2*k
                
                # 1. Center point (coincides with coarse node)
                fine[fi, fj, fk] = coarse[i, j, k]
                
                # 2. Odd-index points (between coarse nodes)
                
                # Along X (i,j, k+0.5) -> fine(fi, fj, fk+1)
                fine[fi, fj, fk+1] = 0.5 * (coarse[i, j, k] + coarse[i, j, k+1])
                
                # Along Y (i, j+0.5, k) -> fine(fi, fj+1, fk)
                fine[fi, fj+1, fk] = 0.5 * (coarse[i, j, k] + coarse[i, j+1, k])
                
                # Along Z (i+0.5, j, k) -> fine(fi+1, fj, fk)
                fine[fi+1, fj, fk] = 0.5 * (coarse[i, j, k] + coarse[i+1, j, k])
                
                # 3. Face centers (2 odd indices)
                
                # XY plane (i+0.5, j+0.5, k)
                fine[fi+1, fj+1, fk] = 0.25 * (coarse[i, j, k] + coarse[i+1, j, k] + 
                                               coarse[i, j+1, k] + coarse[i+1, j+1, k])
                                               
                # XZ plane (i+0.5, j, k+0.5)
                fine[fi+1, fj, fk+1] = 0.25 * (coarse[i, j, k] + coarse[i+1, j, k] + 
                                               coarse[i, j, k+1] + coarse[i+1, j, k+1])
                                               
                # YZ plane (i, j+0.5, k+0.5)
                fine[fi, fj+1, fk+1] = 0.25 * (coarse[i, j, k] + coarse[i, j+1, k] + 
                                               coarse[i, j, k+1] + coarse[i, j+1, k+1])
                                               
                # 4. Cell center (3 odd indices)
                # (i+0.5, j+0.5, k+0.5)
                fine[fi+1, fj+1, fk+1] = 0.125 * (coarse[i, j, k] + coarse[i+1, j, k] +
                                                  coarse[i, j+1, k] + coarse[i+1, j+1, k] +
                                                  coarse[i, j, k+1] + coarse[i+1, j, k+1] +
                                                  coarse[i, j+1, k+1] + coarse[i+1, j+1, k+1])
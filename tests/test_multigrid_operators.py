import numpy as np
import pytest
from src.Poisson.multigrid_operators import restrict, prolong

def test_restrict_constant():
    """Test that restricting a constant field preserves the value."""
    N_fine = 8
    N_coarse = N_fine // 2
    
    # Fine grid with ghost layers
    fine_grid = np.ones((N_fine + 2, N_fine + 2, N_fine + 2))
    
    # Coarse grid buffer
    coarse_grid = np.zeros((N_coarse + 2, N_coarse + 2, N_coarse + 2))
    
    restrict(fine_grid, coarse_grid)
    
    expected = np.ones((N_coarse, N_coarse, N_coarse))
    
    # Check interior of coarse grid [1:-1, 1:-1, 1:-1]
    np.testing.assert_allclose(
        coarse_grid[1:-1, 1:-1, 1:-1], 
        expected, 
        rtol=1e-14, 
        err_msg="Restriction of constant field failed"
    )

def test_prolong_constant():
    """Test that prolonging a constant field preserves the value."""
    N_coarse = 4
    N_fine = N_coarse * 2
    
    coarse_grid = np.ones((N_coarse + 2, N_coarse + 2, N_coarse + 2))
    fine_grid = np.zeros((N_fine + 2, N_fine + 2, N_fine + 2))
    
    prolong(coarse_grid, fine_grid)
    
    expected = np.ones((N_fine, N_fine, N_fine))
    
    # Check interior of fine grid [1:-1, 1:-1, 1:-1]
    np.testing.assert_allclose(
        fine_grid[1:-1, 1:-1, 1:-1], 
        expected, 
        rtol=1e-14, 
        err_msg="Prolongation of constant field failed"
    )

def test_prolong_linear_gradient():
    """Test prolongation on a linear gradient in X direction."""
    N_coarse = 4
    N_fine = N_coarse * 2
    
    coarse_grid = np.zeros((N_coarse + 2, N_coarse + 2, N_coarse + 2))
    
    # Set linear gradient on coarse interior
    # coarse interior indices k=1..4
    # value = k-1 (so 0, 1, 2, 3)
    for k in range(N_coarse):
        coarse_grid[1:-1, 1:-1, k+1] = float(k)
        
    fine_grid = np.zeros((N_fine + 2, N_fine + 2, N_fine + 2))
    
    prolong(coarse_grid, fine_grid)
    
    # Expected fine grid
    # Since coarse grid has 0 boundaries in Y and Z, the fine grid will interpolate to 0 near edges.
    # We only check the center line in X (3rd dim) where the 1D gradient should be preserved.
    # Gradient was set in 3rd dimension: coarse[..., k+1] = k
    
    center_z = N_fine // 2 # 1st dim
    center_y = N_fine // 2 # 2nd dim
    
    # Extract line along X (3rd dim) for the interior
    fine_center_line = fine_grid[center_z, center_y, 1:-1]
    
    expected_line = np.zeros(N_fine)
    for k in range(N_fine):
        # fine index k (0..7)
        # Maps to coarse coordinate k/2
        # Coarse values are 0, 1, 2, 3 at indices 1, 2, 3, 4
        # Fine index k (interior) corresponds to global position.
        # The test set coarse[1]=0, coarse[2]=1...
        # fine[1] (k=0) is at x=0 (relative to interior start). 
        # It coincides with coarse[1] if using node-centered?
        # Or is it 0.5*(coarse[0]+coarse[1])?
        
        # Let's use the known correct interpolation logic:
        # even points (0, 2, 4...) map to coarse points.
        # odd points (1, 3, 5...) are averages.
        
        # Wait, previously:
        # i=0 (ghost) -> fi=0 (ghost), fi+1=1 (first interior).
        # fine[1] = 0.5 * (coarse[0] + coarse[1]).
        # coarse[0]=0 (ghost). coarse[1]=0.
        # So fine[1] = 0.
        
        # fine[2] (k=1) -> coarse[1] = 0.
        
        # fine[3] (k=2) -> 0.5*(coarse[1]+coarse[2]) = 0.5*(0+1) = 0.5.
        
        # fine[4] (k=3) -> coarse[2] = 1.
        
        # So the sequence is 0, 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0.
        
        if k == 0: val = 0.0
        elif k == 1: val = 0.0
        else: val = (k - 1) / 2.0
        
        expected_line[k] = val

    np.testing.assert_allclose(
        fine_center_line,
        expected_line,
        rtol=1e-14,
        err_msg="Prolongation of linear gradient failed on center line"
    )

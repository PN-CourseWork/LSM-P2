import numpy as np
import math
import argparse
import os

def compute_true_solution(N: int) -> np.ndarray:
    """Compute the analytical solution."""
    xs, ys, zs = np.ogrid[-1:1:complex(N), -1:1:complex(N), -1:1:complex(N)]
    u_true = np.sin(np.pi * xs) * np.sin(np.pi * ys) * np.sin(np.pi * zs)
    return u_true

def validate_solution(u: np.ndarray, u_true: np.ndarray) -> dict:
    """Validate the computed solution against the true solution."""
    N = u.shape[0]
    
    # Compute error metrics
    diff = u - u_true
    l2_error = np.sqrt(np.sum(diff ** 2)) / (N ** 3)
    max_error = np.abs(diff).max()
    rel_error = l2_error / (np.sqrt(np.sum(u_true ** 2)) / (N ** 3))
    
    return {
        'l2_error': l2_error,
        'max_error': max_error,
        'rel_error': rel_error,
        'mean_u': np.mean(u),
        'mean_u_true': np.mean(u_true),
        'min_u': np.min(u),
        'max_u': np.max(u)
    }

def compare_solutions(file1: str, file2: str):
    """Compare two solution files."""
    u1 = np.load(file1)
    u2 = np.load(file2)
    
    print(f"\n{'='*60}")
    print(f"Comparing: {file1} vs {file2}")
    print(f"{'='*60}")
    
    # Check shapes
    print(f"Shape 1: {u1.shape}")
    print(f"Shape 2: {u2.shape}")
    
    if u1.shape != u2.shape:
        print("ERROR: Shapes don't match!")
        return
    
    # Compute difference
    diff = u1 - u2
    l2_diff = np.sqrt(np.sum(diff ** 2)) / (u1.shape[0] ** 3)
    max_diff = np.abs(diff).max()
    
    print(f"\nDifference metrics:")
    print(f"  L2 norm:     {l2_diff:.6e}")
    print(f"  Max abs:     {max_diff:.6e}")
    print(f"  Mean diff:   {np.mean(diff):.6e}")
    
    if l2_diff < 1e-10:
        print("\n✓ Solutions are IDENTICAL (within machine precision)")
    elif l2_diff < 1e-6:
        print("\n✓ Solutions are very similar (small numerical differences)")
    else:
        print("\n✗ Solutions DIFFER significantly!")

def main():
    parser = argparse.ArgumentParser(description="Verify Poisson solver results")
    parser.add_argument("--file", type=str, default="./Experiments/slice/output/u_final.npy", 
                       help="Solution file to verify")
    parser.add_argument("--compare", type=str, default=None,
                       help="Second file to compare with")
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.file):
        print(f"ERROR: File {args.file} does not exist!")
        return
    
    # Load solution
    print(f"Loading solution from {args.file}...")
    u = np.load(args.file)
    N = u.shape[0]
    
    print(f"\n{'='*60}")
    print(f"Solution Statistics")
    print(f"{'='*60}")
    print(f"Shape:       {u.shape}")
    print(f"Mean:        {np.mean(u):.6e}")
    print(f"Min:         {np.min(u):.6e}")
    print(f"Max:         {np.max(u):.6e}")
    print(f"Std:         {np.std(u):.6e}")
    
    # Check if solution is all zeros (common error)
    if np.allclose(u, 0.0):
        print("\nWARNING: Solution is all zeros! Something went wrong.")
        return
    
    # Compute true solution
    print(f"\nComputing analytical solution...")
    u_true = compute_true_solution(N)
    
    # Validate
    metrics = validate_solution(u, u_true)
    
    print(f"\n{'='*60}")
    print(f"Validation Against True Solution")
    print(f"{'='*60}")
    print(f"L2 error:          {metrics['l2_error']:.6e}")
    print(f"Max error:         {metrics['max_error']:.6e}")
    print(f"Relative error:    {metrics['rel_error']:.6e}")
    print(f"Mean (computed):   {metrics['mean_u']:.6e}")
    print(f"Mean (true):       {metrics['mean_u_true']:.6e}")
    
    # Expected truncation error
    h = 2.0 / (N - 1)
    expected_error = h ** 2
    print(f"\nExpected truncation error: O(h²) ≈ {expected_error:.6e}")
    
    if metrics['l2_error'] < 10 * expected_error:
        print("✓ Solution looks CORRECT (error within expected range)")
    else:
        print("✗ Solution may have issues (error larger than expected)")
    
    # Compare with another file if provided
    if args.compare:
        compare_solutions(args.file, args.compare)

if __name__ == "__main__":
    main()
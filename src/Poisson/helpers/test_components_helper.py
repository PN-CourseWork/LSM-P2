"""Helper script for running MPI component tests."""

import sys
import numpy as np
from mpi4py import MPI
from Poisson import (
    DomainDecomposition,
    NumpyHaloExchange,
    NumPyKernel,
    NoDecomposition,
    GlobalParams
)
from Poisson.problems import sinusoidal_source_term

def test_halo_exchange(comm):
    """Verify halo exchange works correctly."""
    rank = comm.Get_rank()
    size = comm.Get_size()
    N = 16
    
    # Setup decomposition
    decomp = DomainDecomposition(N=N, size=size, strategy='cubic')
    info = decomp.get_rank_info(rank)
    
    # Create array with unique values: rank * 1000 + global_index
    u = np.zeros(info.halo_shape, dtype=np.float64)
    
    # Fill interior
    gs = info.global_start
    nz, ny, nx = info.local_shape
    
    for i in range(nz):
        for j in range(ny):
            for k in range(nx):
                gi, gj, gk = gs[0]+i, gs[1]+j, gs[2]+k
                val = gi * 10000 + gj * 100 + gk
                u[i+1, j+1, k+1] = val
                
    # Exchange
    halo = NumpyHaloExchange()
    halo.exchange_halos(u, decomp, rank, comm)
    
    # Verify Z neighbors
    neighbors = info.neighbors
    success = True
    
    if neighbors['z_lower'] is not None:
        expected_z = gs[0] - 1
        for j in range(ny):
            for k in range(nx):
                gj, gk = gs[1]+j, gs[2]+k
                expected_val = expected_z * 10000 + gj * 100 + gk
                actual_val = u[0, j+1, k+1]
                if not np.isclose(actual_val, expected_val):
                    print(f"Rank {rank}: Z-Lower Halo mismatch. Exp {expected_val}, got {actual_val}")
                    success = False

    if neighbors['z_upper'] is not None:
        expected_z = info.global_end[0]
        for j in range(ny):
            for k in range(nx):
                gj, gk = gs[1]+j, gs[2]+k
                expected_val = expected_z * 10000 + gj * 100 + gk
                actual_val = u[-1, j+1, k+1]
                if not np.isclose(actual_val, expected_val):
                    print(f"Rank {rank}: Z-Upper Halo mismatch. Exp {expected_val}, got {actual_val}")
                    success = False

    if not success:
        print(f"Rank {rank}: Halo Exchange Test FAILED")
        sys.exit(1)

def test_smoothing_correctness(comm):
    """Compare parallel smoothing step against sequential reference."""
    rank = comm.Get_rank()
    size = comm.Get_size()
    N = 65
    omega = 1.0
    
    # 1. Setup Parallel
    decomp = DomainDecomposition(N=N, size=size, strategy='cubic')
    u1, u2, f = decomp.initialize_local_arrays_distributed(N, rank, comm)
    
    kernel = NumPyKernel(N=N, omega=omega)
    halo = NumpyHaloExchange()
    
    # Run one step
    halo.exchange_halos(u1, decomp, rank, comm)
    kernel.step(u1, u2, f)
    decomp.apply_boundary_conditions(u2, rank)
    
    # Gather result
    interior = decomp.extract_interior(u2)
    
    # Gather all interiors to rank 0
    all_interiors = comm.gather(interior, root=0)
    
    if rank == 0:
        # Reconstruct global parallel result
        u_parallel = np.zeros((N, N, N))
        for r, data in enumerate(all_interiors):
            placement = decomp.get_interior_placement(r, N, comm)
            u_parallel[placement] = data
            
        # 2. Run Sequential Reference
        u_seq = np.zeros((N, N, N))
        u_seq_new = np.zeros((N, N, N))
        f_seq = sinusoidal_source_term(N)
        
        # Standard sequential step (no decomposition)
        # Need a sequential kernel
        seq_kernel = NumPyKernel(N=N, omega=omega)
        
        # For sequential, we assume u_seq boundaries are 0 (Dirichlet)
        # The kernel updates interior.
        # Manual BCs: u_seq is already 0.
        
        seq_kernel.step(u_seq, u_seq_new, f_seq)
        
        # Compare
        diff = np.abs(u_parallel - u_seq_new)
        max_diff = np.max(diff)
        print(f"Max difference between Parallel and Sequential: {max_diff}")
        
        if max_diff > 1e-10:
            print("Smoothing Correctness Test FAILED")
            # Print where it failed
            indices = np.where(diff > 1e-10)
            print(f"First failure at: {indices[0][0]}, {indices[1][0]}, {indices[2][0]}")
            print(f"Par: {u_parallel[indices][0]}, Seq: {u_seq_new[indices][0]}")
            sys.exit(1)
        else:
            print("Smoothing Correctness Test PASSED")

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    # test_halo_exchange(comm)
    test_smoothing_correctness(comm)

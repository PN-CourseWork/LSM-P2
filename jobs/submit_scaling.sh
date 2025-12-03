#!/bin/bash
#BSUB -J scaling_experiment
#BSUB -q hpcintro
#BSUB -n 48
#BSUB -W 01:00
#BSUB -R "span[ptile=24]"
#BSUB -o logs/scaling_%J.out
#BSUB -e logs/scaling_%J.err

# Comprehensive scaling experiment submission
# Runs Strong (up to 48 ranks) and Weak scaling (1-32 ranks) on 2 nodes.


# Load MPI module
module load mpi

# Common flags
COMMON="+machine=dtu_hpc hydra/launcher=basic"

echo "=== Starting Strong Scaling: Jacobi ==="
uv run python run_solver.py -m +experiment=strong_scaling_jacobi $COMMON

echo "=== Starting Strong Scaling: FMG ==="
uv run python run_solver.py -m +experiment=strong_scaling_fmg $COMMON

echo "=== Starting Weak Scaling (Jacobi & FMG) ==="
# Weak Scaling: Constant work per rank (approx 17M pts, matching N=257 base)
# We manually pair Ranks with N to maintain constant volume/rank while respecting FMG constraints.

# Ranks: 1,   2,   4,   8,   16,  32
# N:     257, 321, 417, 513, 641, 817

RANKS=(1 2 4 8 16 32)
SIZES=(257 321 417 513 641 817)

# Loop over indices
for i in "${!RANKS[@]}"; do
    R="${RANKS[$i]}"
    N="${SIZES[$i]}"
    
    echo ">>> Weak Scaling: $R Ranks, N=$N <<<"
    
    # Run Jacobi (sweeping strategy)
    uv run python run_solver.py -m \
        +experiment=weak_scaling_jacobi \
        n_ranks=$R \
        N=$N \
        $COMMON

    # Run FMG (sweeping strategy)
    uv run python run_solver.py -m \
        +experiment=weak_scaling_fmg \
        n_ranks=$R \
        N=$N \
        $COMMON
done

echo "All scaling experiments submitted."

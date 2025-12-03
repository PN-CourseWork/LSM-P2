#!/bin/bash
#BSUB -J strong_jacobi
#BSUB -q hpcintro
#BSUB -n 144
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 2:00
#BSUB -o logs/lsf/strong_jacobi_%J.out
#BSUB -e logs/lsf/strong_jacobi_%J.err

# =============================================================================
# Strong Scaling: Jacobi solver only
# 144 cores = 6 nodes × 24 cores
# Fixed problem sizes (N=257, 513), varying rank counts up to 144
# =============================================================================

module load mpi
mkdir -p logs/lsf

export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPI_OPTIONS="--map-by ppr:12:package --bind-to core"

MAX_ITER=300

echo "=== Strong Scaling: Jacobi ==="
echo "MPI_OPTIONS: $MPI_OPTIONS"
echo "MAX_ITER: $MAX_ITER"

# Sweep: N=257,513 × strategy=sliced,cubic × n_ranks=1,2,4,8,16,24,36,48,72,96,144
for N in 257 513; do
    for strategy in sliced cubic; do
        for ranks in 1 2 4 8 16 24 36 48 72 96 144; do
            echo "  N=$N, strategy=$strategy, ranks=$ranks"
            uv run python run_solver.py \
                +experiment=strong_scaling_jacobi \
                N=$N \
                strategy=$strategy \
                n_ranks=$ranks \
                max_iter=$MAX_ITER \
                hydra/launcher=basic \
                mlflow=databricks
        done
    done
done

echo "Strong scaling Jacobi completed"

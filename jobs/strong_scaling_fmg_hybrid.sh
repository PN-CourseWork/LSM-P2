#!/bin/bash
#BSUB -J fmg_hybrid
#BSUB -q hpcintro
#BSUB -n 144
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 1:00
#BSUB -o logs/lsf/fmg_hybrid_%J.out
#BSUB -e logs/lsf/fmg_hybrid_%J.err

# =============================================================================
# Hybrid Strong Scaling: FMG with MPI + Numba threads
# 144 cores = 6 nodes × 24 cores
# Each MPI rank uses 4 Numba threads
# Max ranks = 144 / 4 = 36
# =============================================================================

module load mpi
mkdir -p logs/lsf

export NUMBA_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MPI_OPTIONS="--map-by ppr:6:package --bind-to none"

MAX_ITER=300
NUMBA_THREADS=4

echo "=== Hybrid Strong Scaling: FMG (MPI + 4 Numba threads) ==="
echo "MPI_OPTIONS: $MPI_OPTIONS"
echo "NUMBA_NUM_THREADS: $NUMBA_NUM_THREADS"
echo "MAX_ITER: $MAX_ITER"

# With 4 threads per rank: max 36 ranks on 144 cores
# Rank counts that divide well: 1, 2, 4, 6, 8, 12, 18, 24, 36
for N in 257 513; do
    for strategy in sliced cubic; do
        for ranks in 1 2 4 6 8 12 18 24 36; do
            echo "  N=$N, strategy=$strategy, ranks=$ranks (× 4 threads = $((ranks * 4)) cores)"
            uv run python run_solver.py \
                +experiment=strong_scaling_fmg \
                N=$N \
                strategy=$strategy \
                n_ranks=$ranks \
                specified_numba_threads=$NUMBA_THREADS \
                max_iter=$MAX_ITER \
                hydra/launcher=basic \
                mlflow=databricks
        done
    done
done

echo "Hybrid FMG scaling completed"

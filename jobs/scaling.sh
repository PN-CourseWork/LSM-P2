#!/bin/bash
#BSUB -J scaling
#BSUB -q hpcintro
#BSUB -n 48
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 1:00
#BSUB -o logs/lsf/scaling_%J.out
#BSUB -e logs/lsf/scaling_%J.err

# =============================================================================
# Scaling Experiments: Strong and Weak scaling for Jacobi and FMG
# 96 cores = 4 nodes × 24 cores = 8 packages (2 per node)
# Hydra sweeper handles: N, strategy, n_ranks
# =============================================================================

module load mpi
mkdir -p logs/lsf

export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Spread ranks across all 8 packages (4 nodes × 2 packages)
# ppr:12:package allows up to 96 ranks spread evenly
#export MPI_OPTIONS="--map-by ppr:8:package --bind-to core"

# Iteration count for scaling experiments
MAX_ITER=50

echo "=== Strong Scaling: Jacobi ==="
echo "MPI_OPTIONS: $MPI_OPTIONS"
uv run python run_solver.py \
    +experiment=weak-v2 \
    max_iter=$MAX_ITER \
    hydra/launcher=basic \
    mlflow=databricks \
    experiment_name=weak_scaling_v2-LARGE \
    -m


#!/bin/bash
#BSUB -J weak_jacobi
#BSUB -q hpcintro
#BSUB -n 8
#BSUB -R "span[ptile=4]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 1:00
#BSUB -o logs/lsf/weak_jacobi_%J.out
#BSUB -e logs/lsf/weak_jacobi_%J.err

# =============================================================================
# Weak Scaling: Jacobi solver only
# Debug version with small problems (1 and 2 ranks)
#
# Usage:
#   Local:   bash jobs/weak_scaling_jacobi.sh
#   Cluster: bsub < jobs/weak_scaling_jacobi.sh
# =============================================================================

set -euo pipefail

module load mpi
mkdir -p logs/lsf
export MPI_OPTIONS="--map-by ppr:4:package --bind-to core"
export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1

MAX_ITER=100

echo "=== Weak Scaling: Jacobi (Debug) ==="
echo "MPI_OPTIONS: $MPI_OPTIONS"





# Small debug sizes: N=10 with 1 rank, N=17 with 2 ranks
# (keeping ~8³ points per rank for quick tests)
for pair in "10,1" "17,2"; do
    N=$(echo $pair | cut -d',' -f1)
    ranks=$(echo $pair | cut -d',' -f2)
    echo "  N=$N, ranks=$ranks"
    uv run python run_solver.py \
        +experiment=weak_scaling_jacobi \
        N=$N \
        n_ranks=$ranks \
        max_iter=$MAX_ITER \
        hydra/launcher=basic \
        mlflow=databricks \
        -m
done

echo "Weak scaling Jacobi completed"

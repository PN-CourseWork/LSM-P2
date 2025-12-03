#!/bin/bash
#BSUB -J comm_spread
#BSUB -q hpcintro
#BSUB -n 48
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 1:00
#BSUB -o logs/lsf/comm_spread_%J.out
#BSUB -e logs/lsf/comm_spread_%J.err

# =============================================================================
# Communication Experiment: SPREAD binding
# 24 ranks spread across 2 nodes (6 per package × 2 packages × 2 nodes)
# Maximizes memory bandwidth, tests inter-node communication
# =============================================================================

module load mpi
mkdir -p logs/lsf

export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1

# MPI parameters: 24 ranks spread across 2 nodes
NP=24
NPS=6  # ranks per package (socket)

# Spread: 6 ranks per package, spread across nodes
export MPI_OPTIONS="--map-by ppr:$NPS:package --bind-to core"

echo "=== Communication: Spread binding ==="
echo "NP=$NP, NPS=$NPS (6 ranks/package × 2 packages × 2 nodes)"
echo "MPI_OPTIONS: $MPI_OPTIONS"

uv run python run_solver.py \
    +experiment=communication \
    hydra/launcher=basic \
    experiment_name=comm_spread \
    mlflow=databricks \
    -m

echo "Communication spread completed"

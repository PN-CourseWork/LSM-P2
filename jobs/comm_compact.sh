#!/bin/bash
#BSUB -J comm_compact
#BSUB -q hpcintro
#BSUB -n 48
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 1:00
#BSUB -o logs/lsf/comm_compact_%J.out
#BSUB -e logs/lsf/comm_compact_%J.err

# =============================================================================
# Communication Experiment: COMPACT binding
# 24 ranks packed on 2 nodes (12 per node × 2 nodes)
# Minimizes inter-node communication, tests intra-node bandwidth
# =============================================================================

module load mpi
mkdir -p logs/lsf

export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1

# MPI parameters: 24 ranks compacted across 2 nodes
NP=24
NPN=12  # ranks per node

# Compact: 12 ranks per node, fill nodes before spreading
export MPI_OPTIONS="--map-by ppr:$NPN:node --bind-to core"

echo "=== Communication: Compact binding ==="
echo "NP=$NP, NPN=$NPN (12 ranks/node × 2 nodes)"
echo "MPI_OPTIONS: $MPI_OPTIONS"

uv run python run_solver.py \
    +experiment=communication \
    hydra/launcher=basic \
    experiment_name=comm_compact \
    mlflow=databricks \
    -m

echo "Communication compact completed"

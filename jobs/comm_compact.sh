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
# Ranks packed together on same node first
# =============================================================================

module load mpi
mkdir -p logs/lsf

# Compact: fill one node before moving to next (24 per node)
MPIOPT="--map-by ppr:24:node --bind-to core"

echo "=== Communication: Compact binding, 24 ranks (intra-node) ==="
mpirun $MPIOPT -n 24 uv run python run_solver.py \
    +experiment=communication \
    hydra/launcher=basic \
    n_ranks=24 \
    experiment_name=comm_compact \
    mlflow=databricks \
    -m

echo "=== Communication: Compact binding, 48 ranks (inter-node) ==="
mpirun $MPIOPT -n 48 uv run python run_solver.py \
    +experiment=communication \
    hydra/launcher=basic \
    n_ranks=48 \
    experiment_name=comm_compact \
    mlflow=databricks \
    -m

echo "Communication compact completed"

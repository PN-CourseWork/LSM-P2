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
# Ranks spread across packages for maximum memory bandwidth
# =============================================================================

module load mpi
mkdir -p logs/lsf

# Spread: 12 ranks per package (socket), spread across nodes
MPIOPT="--map-by ppr:12:package --bind-to core"

echo "=== Communication: Spread binding, 24 ranks (intra-node) ==="
mpirun $MPIOPT -n 24 uv run python run_solver.py \
    +experiment=communication \
    hydra/launcher=basic \
    n_ranks=24 \
    experiment_name=comm_spread \
    mlflow=databricks \
    -m

echo "=== Communication: Spread binding, 48 ranks (inter-node) ==="
mpirun $MPIOPT -n 48 uv run python run_solver.py \
    +experiment=communication \
    hydra/launcher=basic \
    n_ranks=48 \
    experiment_name=comm_spread \
    mlflow=databricks \
    -m

echo "Communication spread completed"

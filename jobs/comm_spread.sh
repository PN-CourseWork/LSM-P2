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
# Ranks spread across nodes/sockets for maximum memory bandwidth
# =============================================================================

module load mpi
cd $HOME/LSM-Project_2
mkdir -p logs/lsf

# Spread: fill sockets evenly across all nodes
MPIOPT="--map-by ppr:12:socket --bind-to core"

echo "=== Communication: Spread binding, 24 ranks (intra-node) ==="
mpiexec $MPIOPT -n 24 uv run python run_solver.py \
    +experiment=communication \
    hydra/launcher=basic \
    n_ranks=24 \
    experiment_name=comm_spread

echo "=== Communication: Spread binding, 48 ranks (inter-node) ==="
mpiexec $MPIOPT -n 48 uv run python run_solver.py \
    +experiment=communication \
    hydra/launcher=basic \
    n_ranks=48 \
    experiment_name=comm_spread

echo "Communication spread completed"

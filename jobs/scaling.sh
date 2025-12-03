#!/bin/bash
#BSUB -J scaling
#BSUB -q hpcintro
#BSUB -n 24
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 0:30
#BSUB -o logs/lsf/scaling_%J.out
#BSUB -e logs/lsf/scaling_%J.err

# =============================================================================
# Scaling Experiment
# Sweeps over n_ranks defined in the config
# run_solver.py spawns MPI internally based on config
# =============================================================================

module load mpi
mkdir -p logs/lsf

export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1

echo "=== Running strong scaling (Jacobi) ==="
uv run python run_solver.py \
    +experiment=strong_scaling_jacobi \
    hydra/launcher=basic \
    mlflow=databricks \
    -m

echo "Scaling experiment completed"

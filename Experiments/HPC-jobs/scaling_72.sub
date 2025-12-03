#!/bin/bash
#BSUB -J scaling_72
#BSUB -q hpcintro
#BSUB -n 72
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 1:00
#BSUB -o logs/lsf/scaling_72_%J.out
#BSUB -e logs/lsf/scaling_72_%J.err

# =============================================================================
# Scaling Experiment: 3 nodes (72 cores)
# Override n_ranks for 3-node configuration
# =============================================================================

module load mpi
cd $LS_SUBCWD

uv run python run_solver.py -cn experiment/scaling -m \
    hydra/launcher=basic mpi.bind_to=core n_ranks=72

echo "Scaling 72 completed"

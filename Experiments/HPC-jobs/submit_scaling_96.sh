#!/bin/bash
#BSUB -J scaling_96
#BSUB -q hpcintro
#BSUB -n 96
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 1:00
#BSUB -o logs/lsf/scaling_96_%J.out
#BSUB -e logs/lsf/scaling_96_%J.err

# =============================================================================
# Scaling Experiment: 4 nodes (96 cores)
# Override n_ranks for 4-node configuration
# =============================================================================

module load mpi
cd $LS_SUBCWD

uv run python run_solver.py -cn experiment/scaling -m \
    hydra/launcher=basic mpi.bind_to=core n_ranks=96

echo "Scaling 96 completed"

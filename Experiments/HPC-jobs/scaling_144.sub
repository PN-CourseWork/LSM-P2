#!/bin/bash
#BSUB -J scaling_144
#BSUB -q hpcintro
#BSUB -n 144
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 1:00
#BSUB -o logs/lsf/scaling_144_%J.out
#BSUB -e logs/lsf/scaling_144_%J.err

# =============================================================================
# Scaling Experiment: 6 nodes (144 cores)
# Override n_ranks for 6-node configuration
# =============================================================================

module load mpi
cd $LS_SUBCWD

uv run python run_solver.py -cn experiment/scaling -m \
    hydra/launcher=basic mpi.bind_to=core n_ranks=144

echo "Scaling 144 completed"

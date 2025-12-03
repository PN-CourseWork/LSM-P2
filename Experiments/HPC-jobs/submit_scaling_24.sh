#!/bin/bash
#BSUB -J scaling_24
#BSUB -q hpcintro
#BSUB -n 24
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 2:00
#BSUB -o logs/lsf/scaling_24_%J.out
#BSUB -e logs/lsf/scaling_24_%J.err

# =============================================================================
# Scaling Experiment: 1 node (24 cores)
# Uses native Hydra sweep from experiment/scaling.yaml
# =============================================================================

module load mpi
cd $LS_SUBCWD

uv run python run_solver.py -cn experiment/scaling -m \
    hydra/launcher=basic mpi.bind_to=core

echo "Scaling 24 completed"

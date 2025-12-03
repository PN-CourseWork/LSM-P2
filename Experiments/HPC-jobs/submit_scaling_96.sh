#!/bin/bash
#BSUB -J scaling_96
#BSUB -q hpcintro
#BSUB -n 96
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 1:00
#BSUB -o logs/lsf/scaling_96_%J.out
#BSUB -e logs/lsf/scaling_96_%J.err

# =============================================================================
# Scaling Experiment: 4 nodes (96 cores)
# Strong scaling: n_ranks × strategy × N = 1 × 2 × 2 = 4 runs (Hydra multirun)
# =============================================================================

module load mpi
cd $LS_SUBCWD

uv run python run_solver.py -cn experiment/scaling -m \
    mpi.bind_to=core \
    n_ranks=96 \
    N=257,513 \
    strategy=sliced,cubic

echo "Scaling 96 completed"

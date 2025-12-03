#!/bin/bash
#BSUB -J scaling_24
#BSUB -q hpcintro
#BSUB -n 24
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 2:00
#BSUB -o logs/lsf/scaling_24_%J.out
#BSUB -e logs/lsf/scaling_24_%J.err

# =============================================================================
# Scaling Experiment: 1 node (24 cores)
# Strong scaling: n_ranks × strategy × N = 6 × 2 × 2 = 24 runs (Hydra multirun)
# =============================================================================

module load mpi
cd $LS_SUBCWD

uv run python run_solver.py -cn experiment/scaling -m \
    hydra/launcher=basic \
    mpi.bind_to=core \
    n_ranks=1,2,4,8,12,24 \
    N=257,513 \
    strategy=sliced,cubic

echo "Scaling 24 completed"

#!/bin/bash
#BSUB -J scaling_48
#BSUB -q hpcintro
#BSUB -n 48
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 1:00
#BSUB -o logs/lsf/scaling_48_%J.out
#BSUB -e logs/lsf/scaling_48_%J.err

# =============================================================================
# Scaling Experiment: 2 nodes (48 cores)
# Strong scaling: n_ranks × strategy × N = 2 × 2 × 2 = 8 runs (Hydra multirun)
# =============================================================================

module load mpi
cd $LS_SUBCWD

uv run python run_solver.py -cn experiment/scaling -m \
    hydra/launcher=basic \
    mpi.bind_to=core \
    n_ranks=32,48 \
    N=257,513 \
    strategy=sliced,cubic

echo "Scaling 48 completed"

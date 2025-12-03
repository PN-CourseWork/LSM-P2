#!/bin/bash
#BSUB -J scaling_24[1-24]
#BSUB -q hpcintro
#BSUB -n 24
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 0:15
#BSUB -o logs/lsf/scaling_24_%J_%I.out
#BSUB -e logs/lsf/scaling_24_%J_%I.err

# =============================================================================
# Scaling Experiment: 1 node (24 cores)
# Strong scaling: n_ranks × strategy × N = 6 × 2 × 2 = 24 jobs
# =============================================================================

module load mpi
cd $LS_SUBCWD

uv run python run_solver.py -cn experiment/scaling -m \
    mpi.bind_to=core \
    n_ranks=1,2,4,8,12,24 \
    N=257,513 \
    strategy=sliced,cubic

echo "Job $LSB_JOBINDEX completed"

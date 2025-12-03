#!/bin/bash
#BSUB -J scaling_72[1-8]
#BSUB -q hpcintro
#BSUB -n 72
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 0:15
#BSUB -o logs/lsf/scaling_72_%J_%I.out
#BSUB -e logs/lsf/scaling_72_%J_%I.err

# =============================================================================
# Scaling Experiment: 3 nodes (72 cores)
# Strong scaling: n_ranks × strategy × N = 2 × 2 × 2 = 8 jobs
# =============================================================================

module load mpi
cd $LS_SUBCWD

uv run python run_solver.py -cn experiment/scaling -m \
    mpi.bind_to=core \
    n_ranks=64,72 \
    N=257,513 \
    strategy=sliced,cubic

echo "Job $LSB_JOBINDEX completed"

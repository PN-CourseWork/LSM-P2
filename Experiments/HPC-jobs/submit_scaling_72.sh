#!/bin/bash
#BSUB -J scaling_72[1-4]
#BSUB -q hpc
#BSUB -n 72
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 02:00:00
#BSUB -o logs/lsf/scaling_72_%J_%I.out
#BSUB -e logs/lsf/scaling_72_%J_%I.err

# =============================================================================
# Scaling Experiment: 3 nodes (72 cores)
# Strong scaling: n_ranks × strategy = 2 × 2 = 4 jobs
# =============================================================================

module load mpi
cd $LS_SUBCWD

uv run python run_solver.py -cn experiment/scaling -m \
    mpi.bind_to=core \
    n_ranks=64,72 \
    N=257 \
    strategy=sliced,cubic

echo "Job $LSB_JOBINDEX completed"

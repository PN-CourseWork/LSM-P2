#!/bin/bash
#BSUB -J scaling_48[1-4]
#BSUB -q hpc
#BSUB -n 48
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 2:00
#BSUB -o logs/lsf/scaling_48_%J_%I.out
#BSUB -e logs/lsf/scaling_48_%J_%I.err

# =============================================================================
# Scaling Experiment: 2 nodes (48 cores)
# Strong scaling: n_ranks × strategy = 2 × 2 = 4 jobs
# =============================================================================

module load mpi
cd $LS_SUBCWD

uv run python run_solver.py -cn experiment/scaling -m \
    mpi.bind_to=core \
    n_ranks=32,48 \
    N=257 \
    strategy=sliced,cubic

echo "Job $LSB_JOBINDEX completed"

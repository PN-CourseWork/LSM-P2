#!/bin/bash
#BSUB -J scaling_96[1-2]
#BSUB -q hpc
#BSUB -n 96
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 02:00:00
#BSUB -o logs/lsf/scaling_96_%J_%I.out
#BSUB -e logs/lsf/scaling_96_%J_%I.err

# =============================================================================
# Scaling Experiment: 4 nodes (96 cores)
# Strong scaling: n_ranks × strategy = 1 × 2 = 2 jobs
# =============================================================================

module load mpi/4.1.6-gcc-14.2.0-binutils-2.42 || module load openmpi || true
cd $LS_SUBCWD

uv run python run_solver.py -cn experiment/scaling -m \
    mpi.bind_to=core \
    n_ranks=96 \
    N=257 \
    strategy=sliced,cubic

echo "Job $LSB_JOBINDEX completed"

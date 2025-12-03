#!/bin/bash
#BSUB -J fmg_strong[1-24]
#BSUB -q hpc
#BSUB -n 96
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 2:00
#BSUB -o logs/lsf/fmg_strong_%J_%I.out
#BSUB -e logs/lsf/fmg_strong_%J_%I.err

# =============================================================================
# FMG Strong Scaling: Pure MPI (1 thread) vs Hybrid (4 threads)
# 1 thread:  n_ranks=1,4,8,16,32,64 (6 values)
# 4 threads: n_ranks=1,2,4,6,12,24  (6 values, max 24×4=96 cores)
# Each × strategy(2) = 6 × 2 = 12 jobs per config = 24 total
# =============================================================================

module load mpi
cd $LS_SUBCWD

# Pure MPI: 1 Numba thread, scale up ranks
uv run python run_solver.py -cn experiment/fmg_scaling -m \
    mpi.bind_to=core \
    n_ranks=1,4,8,16,32,64 \
    N=257 \
    strategy=sliced,cubic \
    numba_threads=1

# Hybrid: 4 Numba threads, fewer ranks to avoid oversubscription
uv run python run_solver.py -cn experiment/fmg_scaling -m \
    mpi.bind_to=core \
    n_ranks=1,2,4,6,12,24 \
    N=257 \
    strategy=sliced,cubic \
    numba_threads=4

echo "Job $LSB_JOBINDEX completed"

#!/bin/bash
#BSUB -J fmg_weak[1-16]
#BSUB -q hpc
#BSUB -n 96
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 02:00:00
#BSUB -o logs/lsf/fmg_weak_%J_%I.out
#BSUB -e logs/lsf/fmg_weak_%J_%I.err

# =============================================================================
# FMG Weak Scaling: Hybrid MPI + Numba
# Paired: (n_ranks, N) = (1,257), (8,513), (27,769), (64,1025)
# 4 pairs × strategy(2) × numba_threads(2) = 16 jobs
# =============================================================================

module load mpi
cd $LS_SUBCWD

# Weak scaling requires paired (n_ranks, N)
uv run python run_solver.py -cn experiment/fmg_scaling -m \
    mpi.bind_to=core n_ranks=1 N=257 strategy=sliced,cubic numba_threads=1,8

uv run python run_solver.py -cn experiment/fmg_scaling -m \
    mpi.bind_to=core n_ranks=8 N=513 strategy=sliced,cubic numba_threads=1,8

uv run python run_solver.py -cn experiment/fmg_scaling -m \
    mpi.bind_to=core n_ranks=27 N=769 strategy=sliced,cubic numba_threads=1,8

uv run python run_solver.py -cn experiment/fmg_scaling -m \
    mpi.bind_to=core n_ranks=64 N=1025 strategy=sliced,cubic numba_threads=1,8

echo "Job $LSB_JOBINDEX completed"

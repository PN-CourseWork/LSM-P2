#!/bin/bash
#BSUB -J fmg_weak
#BSUB -q hpcintro
#BSUB -n 96
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 2:00
#BSUB -o logs/lsf/fmg_weak_%J.out
#BSUB -e logs/lsf/fmg_weak_%J.err

# =============================================================================
# FMG Weak Scaling: Hybrid MPI + Numba
# Paired runs required - strategy sweep defined in config
# =============================================================================

module load mpi
cd $LS_SUBCWD

# Weak scaling requires paired (n_ranks, N)
uv run python run_solver.py -cn experiment/fmg_scaling -m hydra/launcher=basic mpi.bind_to=core n_ranks=1 N=257 numba_threads=1,8
uv run python run_solver.py -cn experiment/fmg_scaling -m hydra/launcher=basic mpi.bind_to=core n_ranks=8 N=513 numba_threads=1,8
uv run python run_solver.py -cn experiment/fmg_scaling -m hydra/launcher=basic mpi.bind_to=core n_ranks=27 N=769 numba_threads=1,8
uv run python run_solver.py -cn experiment/fmg_scaling -m hydra/launcher=basic mpi.bind_to=core n_ranks=64 N=1025 numba_threads=1,8

echo "FMG weak scaling completed"

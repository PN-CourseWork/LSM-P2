#!/bin/bash
#BSUB -J fmg_strong
#BSUB -q hpcintro
#BSUB -n 96
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 2:00
#BSUB -o logs/lsf/fmg_strong_%J.out
#BSUB -e logs/lsf/fmg_strong_%J.err

# =============================================================================
# FMG Strong Scaling: Pure MPI (1 thread) vs Hybrid (4 threads)
# N and strategy sweeps defined in config
# =============================================================================

module load mpi
cd $LS_SUBCWD

# Pure MPI: 1 Numba thread (uses config defaults)
uv run python run_solver.py -cn experiment/fmg_scaling -m hydra/launcher=basic mpi.bind_to=core

# Hybrid: 4 Numba threads, fewer ranks to avoid oversubscription
uv run python run_solver.py -cn experiment/fmg_scaling -m hydra/launcher=basic mpi.bind_to=core numba_threads=4 n_ranks=1,2,4,8,16,24

echo "FMG strong scaling completed"

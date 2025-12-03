#!/bin/bash
#BSUB -J weak_scaling
#BSUB -q hpcintro
#BSUB -n 96
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 2:00
#BSUB -o logs/lsf/weak_scaling_%J.out
#BSUB -e logs/lsf/weak_scaling_%J.err

# =============================================================================
# Weak Scaling Experiment: constant local problem size (~257³ per rank)
# Paired runs required - strategy sweep defined in config
# =============================================================================

module load mpi
cd $LS_SUBCWD

# Weak scaling requires paired (n_ranks, N) - run each pair separately
uv run python run_solver.py -cn experiment/weak_scaling -m hydra/launcher=basic mpi.bind_to=core n_ranks=1 N=257
uv run python run_solver.py -cn experiment/weak_scaling -m hydra/launcher=basic mpi.bind_to=core n_ranks=8 N=513
uv run python run_solver.py -cn experiment/weak_scaling -m hydra/launcher=basic mpi.bind_to=core n_ranks=27 N=769
uv run python run_solver.py -cn experiment/weak_scaling -m hydra/launcher=basic mpi.bind_to=core n_ranks=64 N=1025

echo "Weak scaling completed"

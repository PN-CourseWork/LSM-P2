#!/bin/bash
#BSUB -J weak_scaling[1-8]
#BSUB -q hpc
#BSUB -n 96
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 02:00:00
#BSUB -o logs/lsf/weak_scaling_%J_%I.out
#BSUB -e logs/lsf/weak_scaling_%J_%I.err

# =============================================================================
# Weak Scaling Experiment: constant local problem size (~257³ per rank)
# Paired runs: (n_ranks, N) = (1,257), (8,513), (27,769), (64,1025)
# Each pair × strategy(2) = 4 pairs × 2 = 8 jobs
# =============================================================================

module load mpi/5.0.5-gcc-14.2.0
cd $LS_SUBCWD

# Weak scaling requires paired (n_ranks, N) - run each pair separately
# Pair 1: 1 rank, N=257
uv run python run_solver.py -cn experiment/weak_scaling -m \
    mpi.bind_to=core n_ranks=1 N=257 strategy=sliced,cubic

# Pair 2: 8 ranks, N=513
uv run python run_solver.py -cn experiment/weak_scaling -m \
    mpi.bind_to=core n_ranks=8 N=513 strategy=sliced,cubic

# Pair 3: 27 ranks, N=769
uv run python run_solver.py -cn experiment/weak_scaling -m \
    mpi.bind_to=core n_ranks=27 N=769 strategy=sliced,cubic

# Pair 4: 64 ranks, N=1025
uv run python run_solver.py -cn experiment/weak_scaling -m \
    mpi.bind_to=core n_ranks=64 N=1025 strategy=sliced,cubic

echo "Job $LSB_JOBINDEX completed"

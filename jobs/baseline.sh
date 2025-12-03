#!/bin/bash
#BSUB -J baseline
#BSUB -q hpcintro
#BSUB -n 24
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 1:00
#BSUB -o logs/lsf/baseline_%J.out
#BSUB -e logs/lsf/baseline_%J.err

# =============================================================================
# Sequential Baselines: Jacobi and FMG for N=257,513
# Single rank (sequential solver) for strong scaling reference
# =============================================================================

module load mpi
mkdir -p logs/lsf

export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1

MAX_ITER=300

echo "=== Sequential Baseline: Jacobi N=257 ==="
uv run python run_solver.py \
    +experiment=strong_scaling_jacobi \
    N=257 n_ranks=1 \
    max_iter=$MAX_ITER \
    hydra/launcher=basic \
    mlflow=databricks

echo "=== Sequential Baseline: Jacobi N=513 ==="
uv run python run_solver.py \
    +experiment=strong_scaling_jacobi \
    N=513 n_ranks=1 \
    max_iter=$MAX_ITER \
    hydra/launcher=basic \
    mlflow=databricks

echo "=== Sequential Baseline: FMG N=257 ==="
uv run python run_solver.py \
    +experiment=strong_scaling_fmg \
    N=257 n_ranks=1 \
    max_iter=$MAX_ITER \
    hydra/launcher=basic \
    mlflow=databricks

echo "=== Sequential Baseline: FMG N=513 ==="
uv run python run_solver.py \
    +experiment=strong_scaling_fmg \
    N=513 n_ranks=1 \
    max_iter=$MAX_ITER \
    hydra/launcher=basic \
    mlflow=databricks

echo "All sequential baselines completed"

#!/bin/bash
#BSUB -J scaling
#BSUB -q hpcintro
#BSUB -n 24
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 0:30
#BSUB -o logs/lsf/scaling_%J.out
#BSUB -e logs/lsf/scaling_%J.err

# =============================================================================
# Scaling Experiments: Strong and Weak scaling for Jacobi and FMG
# run_solver.py spawns MPI internally based on config
# =============================================================================

module load mpi
mkdir -p logs/lsf

export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1

# -----------------------------------------------------------------------------
# Strong Scaling: Fixed problem size, varying ranks
# -----------------------------------------------------------------------------

# TEST MODE: max_iter=10 for Databricks validation
# TODO: Increase max_iter for real experiments (250 for Jacobi, 50 for FMG)

echo "=== Strong Scaling: Jacobi ==="
uv run python run_solver.py \
    +experiment=strong_scaling_jacobi \
    max_iter=10 \
    hydra/launcher=basic \
    mlflow=databricks \
    -m

echo "=== Strong Scaling: FMG ==="
uv run python run_solver.py \
    +experiment=strong_scaling_fmg \
    max_iter=10 \
    hydra/launcher=basic \
    mlflow=databricks \
    -m

# -----------------------------------------------------------------------------
# Weak Scaling: Constant work per rank (~270K points/rank)
# N = 2^k + 1 for multigrid compatibility
# 65³/1 = 274K, 129³/8 = 268K, 257³/64 = 265K (perfect 8x scaling)
# With 48 cores, we can do: 65@1, 129@8 (true weak scaling points)
# Also test 193@27 (7.2M/27=267K) for cubic decomp (3³=27)
# -----------------------------------------------------------------------------

echo "=== Weak Scaling: Jacobi ==="
# (N, n_ranks) pairs maintaining ~constant work per rank
# 65@1=274K, 129@8=268K, 193@27=267K (193 works for Jacobi, not multigrid)
# Limited to 24 ranks for single node testing
for pair in "65,1" "129,8"; do
    N=$(echo $pair | cut -d',' -f1)
    ranks=$(echo $pair | cut -d',' -f2)
    echo "  N=$N, ranks=$ranks"
    uv run python run_solver.py \
        +experiment=weak_scaling_jacobi \
        N=$N n_ranks=$ranks \
        max_iter=10 \
        hydra/launcher=basic \
        mlflow=databricks \
        -m
done

echo "=== Weak Scaling: FMG ==="
# FMG needs N = 2^k + 1 for multigrid levels
# Only 65@1 and 129@8 are valid multigrid-compatible weak scaling points
for pair in "65,1" "129,8"; do
    N=$(echo $pair | cut -d',' -f1)
    ranks=$(echo $pair | cut -d',' -f2)
    echo "  N=$N, ranks=$ranks"
    uv run python run_solver.py \
        +experiment=weak_scaling_fmg \
        N=$N n_ranks=$ranks \
        max_iter=10 \
        hydra/launcher=basic \
        mlflow=databricks \
        -m
done

echo "All scaling experiments completed"

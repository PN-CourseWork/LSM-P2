#!/bin/bash
#BSUB -J scaling
#BSUB -q hpcintro
#BSUB -n 96
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 1:00
#BSUB -o logs/lsf/scaling_%J.out
#BSUB -e logs/lsf/scaling_%J.err

# =============================================================================
# Scaling Experiments: Strong and Weak scaling for Jacobi and FMG
# 96 cores = 4 nodes × 24 cores = 8 packages (2 per node)
# Hydra sweeper handles: N, strategy, n_ranks
# =============================================================================

module load mpi
mkdir -p logs/lsf

export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Spread ranks across all 8 packages (4 nodes × 2 packages)
# ppr:12:package allows up to 96 ranks spread evenly
export MPI_OPTIONS="--map-by ppr:12:package --bind-to core"

# Iteration count for scaling experiments
MAX_ITER=200

echo "=== Strong Scaling: Jacobi ==="
echo "MPI_OPTIONS: $MPI_OPTIONS"
uv run python run_solver.py \
    +experiment=strong_scaling_jacobi \
    max_iter=$MAX_ITER \
    hydra/launcher=basic \
    mlflow=databricks \
    -m

echo "=== Strong Scaling: FMG ==="
uv run python run_solver.py \
    +experiment=strong_scaling_fmg \
    max_iter=$MAX_ITER \
    hydra/launcher=basic \
    mlflow=databricks \
    -m

# =============================================================================
# Weak Scaling: ~127³ points per rank (local subdomain size)
# N = local_size × ranks_per_dim + 2 (for boundaries)
# 129@1 (127+2), 257@8 (2³ decomp), 513@64 (4³ decomp)
# =============================================================================

echo "=== Weak Scaling: Jacobi ==="
for pair in "129,1" "257,8" "513,64"; do
    N=$(echo $pair | cut -d',' -f1)
    ranks=$(echo $pair | cut -d',' -f2)
    echo "  N=$N, ranks=$ranks (~127³ per rank)"
    uv run python run_solver.py \
        +experiment=weak_scaling_jacobi \
        N=$N n_ranks=$ranks \
        max_iter=$MAX_ITER \
        hydra/launcher=basic \
        mlflow=databricks \
        -m
done

echo "=== Weak Scaling: FMG ==="
for pair in "129,1" "257,8" "513,64"; do
    N=$(echo $pair | cut -d',' -f1)
    ranks=$(echo $pair | cut -d',' -f2)
    echo "  N=$N, ranks=$ranks (~127³ per rank)"
    uv run python run_solver.py \
        +experiment=weak_scaling_fmg \
        N=$N n_ranks=$ranks \
        max_iter=$MAX_ITER \
        hydra/launcher=basic \
        mlflow=databricks \
        -m
done

echo "All scaling experiments completed"

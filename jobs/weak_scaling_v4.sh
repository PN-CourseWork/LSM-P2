#!/bin/bash
#BSUB -J weak_scaling
#BSUB -q hpcintro
#BSUB -n 72
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 6:00
#BSUB -o logs/lsf/weak_scaling_%J.out
#BSUB -e logs/lsf/weak_scaling_%J.err

# =============================================================================
# Weak Scaling: Constant work per rank
# Copy of scaling.sh pattern (which works)
#
# Series 1 (~129³/rank): 129@1, 257@8, 385@27, 513@64
# Series 2 (~257³/rank): 257@1, 513@8, 769@27, 1025@64
# =============================================================================

module purge
module load mpi
mkdir -p logs/lsf

export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Spread ranks across packages
export MPI_OPTIONS="--map-by ppr:12:package --bind-to core"

MAX_ITER=100

echo "=== Weak Scaling ==="
echo "MPI_OPTIONS: $MPI_OPTIONS"
echo "MAX_ITER: $MAX_ITER"

# Series 1: ~129³/rank
# Series 2: ~257³/rank
WEAK_PAIRS="129,1 257,8 385,27 513,64 257,1 513,8 769,27 1025,64"

for rep in 1 2 3 4; do
    echo ""
    echo "=== Repetition $rep/4 ==="

    echo "--- Weak Scaling: Jacobi ---"
    for pair in $WEAK_PAIRS; do
        N=$(echo $pair | cut -d',' -f1)
        ranks=$(echo $pair | cut -d',' -f2)
        echo "  N=$N, ranks=$ranks"
        uv run python run_solver.py \
            +experiment=weak_scaling_jacobi \
            N=$N n_ranks=$ranks \
            max_iter=$MAX_ITER \
            hydra/launcher=basic \
            mlflow=databricks \
            -m
    done

    echo "--- Weak Scaling: FMG ---"
    for pair in $WEAK_PAIRS; do
        N=$(echo $pair | cut -d',' -f1)
        ranks=$(echo $pair | cut -d',' -f2)
        echo "  N=$N, ranks=$ranks"
        uv run python run_solver.py \
            +experiment=weak_scaling_fmg \
            N=$N n_ranks=$ranks \
            max_iter=$MAX_ITER \
            hydra/launcher=basic \
            mlflow=databricks \
            -m
    done
done

echo ""
echo "Weak scaling completed"

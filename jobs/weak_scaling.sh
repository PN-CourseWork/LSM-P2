#!/bin/bash
#BSUB -J weak_scaling
#BSUB -q hpcintro
#BSUB -n 64
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 2:00
#BSUB -o logs/lsf/weak_scaling_%J.out
#BSUB -e logs/lsf/weak_scaling_%J.err

# =============================================================================
# Weak Scaling: Constant work per rank (~127³ points/rank)
# (N, ranks): 129@1, 257@8, 513@64
# =============================================================================

module load mpi
mkdir -p logs/lsf

export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPI_OPTIONS="--map-by ppr:12:package --bind-to core"

MAX_ITER=300

echo "=== Weak Scaling: Jacobi ==="
for pair in "129,1" "257,8" "513,64"; do
    N=$(echo $pair | cut -d',' -f1)
    ranks=$(echo $pair | cut -d',' -f2)
    for strategy in sliced cubic; do
        echo "  N=$N, ranks=$ranks, strategy=$strategy (~127³ per rank)"
        uv run python run_solver.py \
            +experiment=weak_scaling_jacobi \
            N=$N \
            n_ranks=$ranks \
            strategy=$strategy \
            max_iter=$MAX_ITER \
            hydra/launcher=basic \
            mlflow=databricks
    done
done

echo "=== Weak Scaling: FMG ==="
for pair in "129,1" "257,8" "513,64"; do
    N=$(echo $pair | cut -d',' -f1)
    ranks=$(echo $pair | cut -d',' -f2)
    for strategy in sliced cubic; do
        echo "  N=$N, ranks=$ranks, strategy=$strategy (~127³ per rank)"
        uv run python run_solver.py \
            +experiment=weak_scaling_fmg \
            N=$N \
            n_ranks=$ranks \
            strategy=$strategy \
            max_iter=$MAX_ITER \
            hydra/launcher=basic \
            mlflow=databricks
    done
done

echo "Weak scaling experiments completed"

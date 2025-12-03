#!/bin/bash
#BSUB -J weak_scaling
#BSUB -q hpcintro
#BSUB -n 72
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 2:00
#BSUB -o logs/lsf/weak_scaling_%J.out
#BSUB -e logs/lsf/weak_scaling_%J.err

# =============================================================================
# Weak Scaling: Constant work per rank (4 points per series)
#
# Series 1 (~129³/rank): 129@1, 257@8, 385@27, 513@64
# =============================================================================

module load mpi
mkdir -p logs/lsf

export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPI_OPTIONS="--map-by ppr:12:package --bind-to core"
export PMIX_MCA_gds=hash

MAX_ITER=50

echo "=== Weak Scaling ==="
echo "MPI_OPTIONS: $MPI_OPTIONS"
echo "MAX_ITER: $MAX_ITER"

# Weak scaling pairs: N,ranks (constant ~129³ points per rank)
# 129@1, 257@8, 385@27, 513@64
WEAK_PAIRS="129,1 257,8 385,27 513,64"

for solver in jacobi fmg; do
    for strategy in sliced cubic; do
        for pair in $WEAK_PAIRS; do
            N=$(echo $pair | cut -d',' -f1)
            ranks=$(echo $pair | cut -d',' -f2)
            echo "  solver=$solver, N=$N, ranks=$ranks, strategy=$strategy"
            uv run python run_solver.py \
                +experiment=weak_scaling_${solver} \
                solver=$solver \
                N=$N \
                n_ranks=$ranks \
                strategy=$strategy \
                communicator=custom \
                max_iter=$MAX_ITER \
                hydra/launcher=basic \
                mlflow=databricks
        done
    done
done

echo "Weak scaling completed"

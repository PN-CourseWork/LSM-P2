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
# Weak Scaling: Constant work per rank (3 series × 4 points each)
#
# Series 1 (~129³/rank): 129@1, 257@8, 385@27, 513@64
# Series 2 (~257³/rank): 257@1, 513@8, 769@27, 1025@64
# Series 3 (~513³/rank): 513@1, 1025@8, 1537@27, 2049@64
# =============================================================================

module load mpi
uv sync
mkdir -p logs/lsf

export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPI_OPTIONS="--map-by ppr:12:package --bind-to core"
export PMIX_MCA_gds=hash

MAX_ITER=100

echo "=== Weak Scaling ==="
echo "MPI_OPTIONS: $MPI_OPTIONS"
echo "MAX_ITER: $MAX_ITER"

# Series 1: ~129³/rank
SERIES_1="129,1 257,8 385,27 513,64"
# Series 2: ~257³/rank
SERIES_2="257,1 513,8 769,27 1025,64"
# Series 3: ~513³/rank
SERIES_3="513,1 1025,8 1537,27 2049,64"

for rep in 1 2 3 4; do
    echo "=== Repetition $rep/4 ==="
    for solver in jacobi fmg; do
        for strategy in sliced cubic; do
            for series in "$SERIES_1" "$SERIES_2" "$SERIES_3"; do
                for pair in $series; do
                    N=$(echo $pair | cut -d',' -f1)
                    ranks=$(echo $pair | cut -d',' -f2)
                    echo "  [$rep/4] solver=$solver, N=$N, ranks=$ranks, strategy=$strategy"
                    uv run python run_solver.py \
                        +experiment=weak_scaling_${solver} \
                        solver=$solver \
                        N=$N \
                        n_ranks=$ranks \
                        strategy=$strategy \
                        communicator=custom \
                        max_iter=$MAX_ITER \
                        hydra/launcher=basic \
                        mlflow=databricks 2>&1
                    echo "  Exit code: $?"
                done
            done
        done
    done
done

echo "Weak scaling completed"

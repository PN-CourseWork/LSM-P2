#!/bin/bash
#BSUB -J weak_debug
#BSUB -q hpcintro
#BSUB -n 24
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 0:15
#BSUB -o logs/lsf/weak_debug_%J.out
#BSUB -e logs/lsf/weak_debug_%J.err

# Quick debug: just 2 runs to verify everything works

module load mpi
uv sync
mkdir -p logs/lsf

export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPI_OPTIONS="--map-by ppr:12:package --bind-to core"
export PMIX_MCA_gds=hash

echo "=== DEBUG: Testing weak scaling setup ==="
echo "PWD: $(pwd)"
echo "Python: $(which python)"
echo "venv Python: $(uv run which python)"
echo "MPI_OPTIONS: $MPI_OPTIONS"

# Test using EXACT same config as working strong scaling
# Just override N and n_ranks for weak scaling pair
echo ""
echo "=== Test: Using strong_scaling_jacobi config (KNOWN WORKING) ==="
echo "Overriding: N=129, n_ranks=1"
uv run python run_solver.py \
    +experiment=strong_scaling_jacobi \
    N=129 n_ranks=1 strategy=cubic \
    max_iter=5 \
    hydra/launcher=basic \
    mlflow=databricks \
    -m 2>&1
echo "Exit code: $?"

echo ""
echo "=== DEBUG COMPLETE ==="

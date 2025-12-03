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
mkdir -p logs/lsf

export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPI_OPTIONS="--map-by ppr:12:package --bind-to core"
export PMIX_MCA_gds=hash

echo "=== DEBUG: Testing weak scaling setup ==="
echo "PWD: $(pwd)"
echo "Python: $(which python)"
echo "MPI_OPTIONS: $MPI_OPTIONS"

# Test 1: Single rank jacobi
echo ""
echo "=== Test 1: jacobi N=33 ranks=1 ==="
uv run python run_solver.py \
    +experiment=weak_scaling_jacobi \
    solver=jacobi N=33 n_ranks=1 strategy=cubic \
    communicator=custom max_iter=5 \
    hydra/launcher=basic mlflow=databricks 2>&1
echo "Exit code: $?"

# Test 2: Multi rank jacobi
echo ""
echo "=== Test 2: jacobi N=65 ranks=8 ==="
uv run python run_solver.py \
    +experiment=weak_scaling_jacobi \
    solver=jacobi N=65 n_ranks=8 strategy=cubic \
    communicator=custom max_iter=5 \
    hydra/launcher=basic mlflow=databricks 2>&1
echo "Exit code: $?"

echo ""
echo "=== DEBUG COMPLETE ==="

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

# Test 0: Simple MPI test
echo ""
echo "=== Test 0: Simple MPI hello world ==="
PYTHON_PATH=$(uv run which python)
mpiexec -n 2 $PYTHON_PATH -c "from mpi4py import MPI; print(f'Hello from rank {MPI.COMM_WORLD.Get_rank()}')" 2>&1
echo "Exit code: $?"

# Test 1: Using exact pattern from working scaling.sh (with -m flag)
echo ""
echo "=== Test 1: jacobi N=129 ranks=1 (with -m) ==="
uv run python run_solver.py \
    +experiment=weak_scaling_jacobi \
    N=129 n_ranks=1 \
    max_iter=5 \
    hydra/launcher=basic \
    mlflow=databricks \
    -m 2>&1
echo "Exit code: $?"

# Test 2: Without -m flag for comparison
echo ""
echo "=== Test 2: jacobi N=129 ranks=1 (without -m) ==="
uv run python run_solver.py \
    +experiment=weak_scaling_jacobi \
    N=129 n_ranks=1 \
    max_iter=5 \
    hydra/launcher=basic \
    mlflow=databricks 2>&1
echo "Exit code: $?"

echo ""
echo "=== DEBUG COMPLETE ==="

#!/bin/bash
#BSUB -J poisson_weak
#BSUB -q hpcintro
#BSUB -W 4:00
#BSUB -M 8GB
#BSUB -n 64
#BSUB -R "span[ptile=24]"
#BSUB -N
#BSUB -o logs/weak_scaling_%J.out
#BSUB -e logs/weak_scaling_%J.err

# Weak scaling: constant work per rank (~257^3 points per rank)
# Pairs: (ranks, N) = (1,257), (8,513), (27,769), (64,1025)

cd $HOME/LSM-Project_2
source .venv/bin/activate

# Load MPI module
module load mpi/5.0.8-gcc-13.4.0-binutils-2.44 >& /dev/null

export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1

mkdir -p logs

# MPI mapping: 12 ranks per socket
MOPTS="--map-by ppr:12:package --bind-to core"

# Run each (ranks, N) pair separately to avoid Cartesian product
# Constant ~129³ points per rank

echo "=== Weak scaling: 1 rank, N=129 ==="
mpirun --map-by ppr:1:package --bind-to core -np 1 uv run python run_solver.py \
    +experiment=weak_scaling \
    n_ranks=1 N=129 \
    mlflow=databricks \
    hydra/launcher=basic \
    -m

echo "=== Weak scaling: 8 ranks, N=257 ==="
mpirun --map-by ppr:4:package --bind-to core -np 8 uv run python run_solver.py \
    +experiment=weak_scaling \
    n_ranks=8 N=257 \
    mlflow=databricks \
    hydra/launcher=basic \
    -m

echo "=== Weak scaling: 27 ranks, N=385 ==="
mpirun $MOPTS -np 27 uv run python run_solver.py \
    +experiment=weak_scaling \
    n_ranks=27 N=385 \
    mlflow=databricks \
    hydra/launcher=basic \
    -m

echo "=== Weak scaling: 64 ranks, N=513 ==="
mpirun $MOPTS -np 64 uv run python run_solver.py \
    +experiment=weak_scaling \
    n_ranks=64 N=513 \
    mlflow=databricks \
    hydra/launcher=basic \
    -m

echo "Weak scaling complete"

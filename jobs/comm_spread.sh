#!/bin/bash
#BSUB -J poisson_comm_spread
#BSUB -q hpcintro
#BSUB -W 1:00
#BSUB -M 8GB
#BSUB -n 24
#BSUB -R "span[ptile=12]"
#BSUB -N
#BSUB -o logs/comm_spread_%J.out
#BSUB -e logs/comm_spread_%J.err

# Communication experiment: SPREAD placement
# 24 ranks across 2 nodes (12 per node) - inter-node communication

cd $HOME/LSM-Project_2
source .venv/bin/activate

# Load MPI module
module load mpi/5.0.8-gcc-13.4.0-binutils-2.44 >& /dev/null

export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1

mkdir -p logs

NP=24

# Spread: 6 ranks per socket, 12 per node, across 2 nodes
MOPTS="--map-by ppr:6:package --bind-to core"

echo "=== Communication experiment: SPREAD (24 ranks, 2 nodes) ==="
mpirun $MOPTS -np $NP uv run python run_solver.py \
    +experiment=communication \
    n_ranks=$NP \
    mlflow=databricks \
    hydra/launcher=basic \
    -m

echo "Communication (spread) complete"

#!/bin/bash
#BSUB -J poisson_comm_compact
#BSUB -q hpcintro
#BSUB -W 1:00
#BSUB -M 8GB
#BSUB -n 24
#BSUB -R "span[hosts=1]"
#BSUB -N
#BSUB -o logs/comm_compact_%J.out
#BSUB -e logs/comm_compact_%J.err

# Communication experiment: COMPACT placement
# All 24 ranks on a single node (intra-node communication only)

cd $HOME/LSM-Project_2
source .venv/bin/activate

# Load MPI module
module load mpi/5.0.8-gcc-13.4.0-binutils-2.44 >& /dev/null

export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1

mkdir -p logs

NP=24

# Compact: 12 ranks per socket on single node
MOPTS="--map-by ppr:12:package --bind-to core"

echo "=== Communication experiment: COMPACT (24 ranks, 1 node) ==="
mpirun $MOPTS -np $NP uv run python run_solver.py \
    +experiment=communication \
    n_ranks=$NP \
    mlflow=databricks \
    hydra/launcher=basic \
    -m

echo "Communication (compact) complete"

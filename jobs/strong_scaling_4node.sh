#!/bin/bash
#BSUB -J poisson_strong_4n
#BSUB -q hpcintro
#BSUB -W 2:00
#BSUB -M 8GB
#BSUB -n 96
#BSUB -R "span[ptile=24]"
#BSUB -N
#BSUB -o logs/strong_4node_%J.out
#BSUB -e logs/strong_4node_%J.err

# Strong scaling: 4 nodes (72-96 ranks)
# Sweeps: N=257,513 x strategy=sliced,cubic

cd $HOME/LSM-Project_2
source .venv/bin/activate

# Load MPI module
module load mpi/5.0.8-gcc-13.4.0-binutils-2.44 >& /dev/null

export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1

mkdir -p logs

# MPI mapping: 12 ranks per socket (24 per node)
MOPTS="--map-by ppr:12:package --bind-to core"

for NP in 72 96; do
    echo "=== Running with $NP ranks ==="
    mpirun $MOPTS -np $NP uv run python run_solver.py \
        +experiment=scaling \
        n_ranks=$NP \
        mlflow=databricks \
        hydra/launcher=basic \
        -m
done

echo "Strong scaling (4 nodes) complete"

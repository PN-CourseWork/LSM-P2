#!/bin/bash
#BSUB -J poisson_strong_1n
#BSUB -q hpcintro
#BSUB -W 2:00
#BSUB -M 8GB
#BSUB -n 24
#BSUB -R "span[hosts=1]"
#BSUB -N
#BSUB -o logs/strong_1node_%J.out
#BSUB -e logs/strong_1node_%J.err

# Strong scaling: 1 node (1-24 ranks)
# Sweeps: N=257,513 x strategy=sliced,cubic

cd $HOME/LSM-Project_2
source .venv/bin/activate

# Load MPI module
module load mpi/5.0.8-gcc-13.4.0-binutils-2.44 >& /dev/null

export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1

mkdir -p logs

# MPI mapping options (2 sockets, 12 cores each)
NPS=12  # ranks per socket

# Run scaling experiment for each rank count
for NP in 1 2 4 8 12 24; do
    echo "=== Running with $NP ranks ==="

    # Calculate ranks per socket for binding
    if [ $NP -le 12 ]; then
        MOPTS="--map-by ppr:$NP:package --bind-to core"
    else
        MOPTS="--map-by ppr:12:package --bind-to core"
    fi

    mpirun $MOPTS -np $NP uv run python run_solver.py \
        +experiment=scaling \
        n_ranks=$NP \
        mlflow=databricks \
        hydra/launcher=basic \
        -m
done

echo "Strong scaling (1 node) complete"

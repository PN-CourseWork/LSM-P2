#!/bin/bash
#BSUB -J poisson_fmg_1n
#BSUB -q hpcintro
#BSUB -W 2:00
#BSUB -M 8GB
#BSUB -n 24
#BSUB -R "span[hosts=1]"
#BSUB -N
#BSUB -o logs/fmg_1node_%J.out
#BSUB -e logs/fmg_1node_%J.err

# FMG scaling: 1 node (1-24 ranks)
# Sweeps: N=257,513 x strategy=sliced,cubic

cd $HOME/LSM-Project_2
source .venv/bin/activate

# Load MPI module
module load mpi/5.0.8-gcc-13.4.0-binutils-2.44 >& /dev/null

export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1

mkdir -p logs

for NP in 1 4 8 16 24; do
    echo "=== Running FMG with $NP ranks ==="

    # Calculate ranks per socket for binding
    if [ $NP -le 12 ]; then
        MOPTS="--map-by ppr:$NP:package --bind-to core"
    else
        MOPTS="--map-by ppr:12:package --bind-to core"
    fi

    mpirun $MOPTS -np $NP uv run python run_solver.py \
        +experiment=fmg_scaling \
        n_ranks=$NP \
        mlflow=databricks \
        hydra/launcher=basic \
        -m
done

echo "FMG scaling (1 node) complete"

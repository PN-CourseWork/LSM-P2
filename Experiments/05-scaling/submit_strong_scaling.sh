#!/bin/bash
#BSUB -J strong_scaling[1-48]
#BSUB -q hpc
#BSUB -W 01:00
#BSUB -n 32
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -o logs/strong_%J_%I.out
#BSUB -e logs/strong_%J_%I.err

# ============================================================================
# Strong Scaling Experiment - Full Parameter Sweep
# ============================================================================
# Fixed problem size N, sweep over:
#   - Ranks: 1, 2, 4, 8, 16, 32 (6 values)
#   - Communicators: numpy, custom (2 values)
#   - Strategies: sliced, cubic (2 values)
#   - Kernels: numpy, numba (2 values)
#
# Total: 6 * 2 * 2 * 2 = 48 jobs
#
# Usage:
#   bsub < submit_strong_scaling.sh
#   N=128 bsub < submit_strong_scaling.sh
# ============================================================================

# Configuration
N=${N:-64}
TOL=${TOL:-1e-6}
MAX_ITER=${MAX_ITER:-50000}

# Parameter arrays
RANKS_ARRAY=(1 2 4 8 16 32)
COMM_ARRAY=(numpy custom)
STRATEGY_ARRAY=(sliced cubic)
KERNEL_ARRAY=(numpy numba)

# Decode job array index (1-48) into parameter indices
# Index = r + 6*c + 12*s + 24*k where r=0-5, c=0-1, s=0-1, k=0-1
IDX=$((LSB_JOBINDEX - 1))
R_IDX=$((IDX % 6))
C_IDX=$(((IDX / 6) % 2))
S_IDX=$(((IDX / 12) % 2))
K_IDX=$((IDX / 24))

NRANKS=${RANKS_ARRAY[$R_IDX]}
COMMUNICATOR=${COMM_ARRAY[$C_IDX]}
STRATEGY=${STRATEGY_ARRAY[$S_IDX]}
KERNEL=${KERNEL_ARRAY[$K_IDX]}

# Build numba flag
NUMBA_FLAG=""
if [ "$KERNEL" = "numba" ]; then
    NUMBA_FLAG="--numba"
fi

# Create log directory
mkdir -p logs

# Load modules (adjust for your HPC system)
module load python3/3.11.4
module load mpi/4.1.5-gcc-12.3.0-binutils-2.40

echo "============================================================"
echo "Strong Scaling: N=${N}³, ranks=${NRANKS}"
echo "Strategy: ${STRATEGY}, Communicator: ${COMMUNICATOR}, Kernel: ${KERNEL}"
echo "Job ID: ${LSB_JOBID}, Array Index: ${LSB_JOBINDEX}"
echo "============================================================"

# Run solver
mpiexec -n $NRANKS uv run python Experiments/05-scaling/compute_scaling.py \
    --N $N \
    --strategy $STRATEGY \
    --communicator $COMMUNICATOR \
    --tol $TOL \
    --max-iter $MAX_ITER \
    $NUMBA_FLAG

echo "============================================================"
echo "Job completed at $(date)"
echo "============================================================"

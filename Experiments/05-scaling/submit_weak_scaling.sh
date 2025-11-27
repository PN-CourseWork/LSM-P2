#!/bin/bash
#BSUB -J weak_scaling[1-48]
#BSUB -q hpc
#BSUB -W 02:00
#BSUB -n 32
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -o logs/weak_%J_%I.out
#BSUB -e logs/weak_%J_%I.err

# ============================================================================
# Weak Scaling Experiment - Full Parameter Sweep
# ============================================================================
# Problem size N scales with ranks, sweep over:
#   - Ranks: 1, 2, 4, 8, 16, 32 (6 values)
#   - Communicators: numpy, custom (2 values)
#   - Strategies: sliced, cubic (2 values)
#   - Kernels: numpy, numba (2 values)
#
# Total: 6 * 2 * 2 * 2 = 48 jobs
#
# For 3D: N ∝ ranks^(1/3) to keep N³/ranks constant.
#
# Usage:
#   bsub < submit_weak_scaling.sh
#   N_BASE=48 bsub < submit_weak_scaling.sh
# ============================================================================

# Configuration
N_BASE=${N_BASE:-32}
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

# Compute N for weak scaling: N = N_BASE * NRANKS^(1/3)
N=$(echo "scale=0; $N_BASE * e(l($NRANKS)/3)" | bc -l)
N=$(( (N + 1) / 2 * 2 ))  # Round to nearest even

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
echo "Weak Scaling: N=${N}³ (base=${N_BASE}), ranks=${NRANKS}"
echo "Points per rank: $((N*N*N / NRANKS))"
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

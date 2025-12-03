#!/bin/bash
#BSUB -J weak_scaling
#BSUB -q hpcintro
#BSUB -n 72
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 2:00
#BSUB -o logs/lsf/weak_scaling_%J.out
#BSUB -e logs/lsf/weak_scaling_%J.err

# =============================================================================
# Weak Scaling: Constant work per rank (4 points each)
#
# Series 1 (~129³/rank): 129@1, 257@8, 385@27, 513@64
# Series 2 (~257³/rank): 257@1, 513@8, 769@27, 1025@64
# Series 3 (~513³/rank): 513@1, 1025@8, 1537@27, 2049@64
# =============================================================================

module load mpi
mkdir -p logs/lsf

export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPI_OPTIONS="--map-by ppr:12:package --bind-to core"

MAX_ITER=50

# =============================================================================
# WEAK SCALING SERIES 1: ~129³ points/rank
# Pairs: 129@1, 257@8, 385@27, 513@64
# =============================================================================

# =============================================================================
# N=129, ranks=1
# =============================================================================
echo "=== Weak Scaling (129³/rank): N=129, ranks=1 ==="

uv run python run_solver.py \
    +experiment=weak_scaling_jacobi \
    N=129 n_ranks=1 strategy=sliced communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

uv run python run_solver.py \
    +experiment=weak_scaling_jacobi \
    N=129 n_ranks=1 strategy=cubic communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

uv run python run_solver.py \
    +experiment=weak_scaling_fmg \
    N=129 n_ranks=1 strategy=sliced communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

uv run python run_solver.py \
    +experiment=weak_scaling_fmg \
    N=129 n_ranks=1 strategy=cubic communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

# =============================================================================
# N=257, ranks=8
# =============================================================================
echo "=== Weak Scaling (129³/rank): N=257, ranks=8 ==="

uv run python run_solver.py \
    +experiment=weak_scaling_jacobi \
    N=257 n_ranks=8 strategy=sliced communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

uv run python run_solver.py \
    +experiment=weak_scaling_jacobi \
    N=257 n_ranks=8 strategy=cubic communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

uv run python run_solver.py \
    +experiment=weak_scaling_fmg \
    N=257 n_ranks=8 strategy=sliced communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

uv run python run_solver.py \
    +experiment=weak_scaling_fmg \
    N=257 n_ranks=8 strategy=cubic communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

# =============================================================================
# N=385, ranks=27
# =============================================================================
echo "=== Weak Scaling (129³/rank): N=385, ranks=27 ==="

uv run python run_solver.py \
    +experiment=weak_scaling_jacobi \
    N=385 n_ranks=27 strategy=sliced communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

uv run python run_solver.py \
    +experiment=weak_scaling_jacobi \
    N=385 n_ranks=27 strategy=cubic communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

uv run python run_solver.py \
    +experiment=weak_scaling_fmg \
    N=385 n_ranks=27 strategy=sliced communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

uv run python run_solver.py \
    +experiment=weak_scaling_fmg \
    N=385 n_ranks=27 strategy=cubic communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

# =============================================================================
# N=513, ranks=64
# =============================================================================
echo "=== Weak Scaling (129³/rank): N=513, ranks=64 ==="

uv run python run_solver.py \
    +experiment=weak_scaling_jacobi \
    N=513 n_ranks=64 strategy=sliced communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

uv run python run_solver.py \
    +experiment=weak_scaling_jacobi \
    N=513 n_ranks=64 strategy=cubic communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

uv run python run_solver.py \
    +experiment=weak_scaling_fmg \
    N=513 n_ranks=64 strategy=sliced communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

uv run python run_solver.py \
    +experiment=weak_scaling_fmg \
    N=513 n_ranks=64 strategy=cubic communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

# =============================================================================
# WEAK SCALING SERIES 2: ~257³ points/rank
# Pairs: 257@1, 513@8, 769@27, 1025@64
# =============================================================================

# =============================================================================
# N=257, ranks=1
# =============================================================================
echo "=== Weak Scaling (257³/rank): N=257, ranks=1 ==="

uv run python run_solver.py \
    +experiment=weak_scaling_jacobi \
    N=257 n_ranks=1 strategy=sliced communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

uv run python run_solver.py \
    +experiment=weak_scaling_jacobi \
    N=257 n_ranks=1 strategy=cubic communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

uv run python run_solver.py \
    +experiment=weak_scaling_fmg \
    N=257 n_ranks=1 strategy=sliced communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

uv run python run_solver.py \
    +experiment=weak_scaling_fmg \
    N=257 n_ranks=1 strategy=cubic communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

# =============================================================================
# N=513, ranks=8
# =============================================================================
echo "=== Weak Scaling (257³/rank): N=513, ranks=8 ==="

uv run python run_solver.py \
    +experiment=weak_scaling_jacobi \
    N=513 n_ranks=8 strategy=sliced communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

uv run python run_solver.py \
    +experiment=weak_scaling_jacobi \
    N=513 n_ranks=8 strategy=cubic communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

uv run python run_solver.py \
    +experiment=weak_scaling_fmg \
    N=513 n_ranks=8 strategy=sliced communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

uv run python run_solver.py \
    +experiment=weak_scaling_fmg \
    N=513 n_ranks=8 strategy=cubic communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

# =============================================================================
# N=769, ranks=27
# =============================================================================
echo "=== Weak Scaling (257³/rank): N=769, ranks=27 ==="

uv run python run_solver.py \
    +experiment=weak_scaling_jacobi \
    N=769 n_ranks=27 strategy=sliced communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

uv run python run_solver.py \
    +experiment=weak_scaling_jacobi \
    N=769 n_ranks=27 strategy=cubic communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

uv run python run_solver.py \
    +experiment=weak_scaling_fmg \
    N=769 n_ranks=27 strategy=sliced communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

uv run python run_solver.py \
    +experiment=weak_scaling_fmg \
    N=769 n_ranks=27 strategy=cubic communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

# =============================================================================
# N=1025, ranks=64
# =============================================================================
echo "=== Weak Scaling (257³/rank): N=1025, ranks=64 ==="

uv run python run_solver.py \
    +experiment=weak_scaling_jacobi \
    N=1025 n_ranks=64 strategy=sliced communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

uv run python run_solver.py \
    +experiment=weak_scaling_jacobi \
    N=1025 n_ranks=64 strategy=cubic communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

uv run python run_solver.py \
    +experiment=weak_scaling_fmg \
    N=1025 n_ranks=64 strategy=sliced communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

uv run python run_solver.py \
    +experiment=weak_scaling_fmg \
    N=1025 n_ranks=64 strategy=cubic communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

# =============================================================================
# WEAK SCALING SERIES 3: ~513³ points/rank
# Pairs: 513@1, 1025@8, 1537@27, 2049@64
# =============================================================================

# =============================================================================
# N=513, ranks=1
# =============================================================================
echo "=== Weak Scaling (513³/rank): N=513, ranks=1 ==="

uv run python run_solver.py \
    +experiment=weak_scaling_jacobi \
    N=513 n_ranks=1 strategy=sliced communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

uv run python run_solver.py \
    +experiment=weak_scaling_jacobi \
    N=513 n_ranks=1 strategy=cubic communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

uv run python run_solver.py \
    +experiment=weak_scaling_fmg \
    N=513 n_ranks=1 strategy=sliced communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

uv run python run_solver.py \
    +experiment=weak_scaling_fmg \
    N=513 n_ranks=1 strategy=cubic communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

# =============================================================================
# N=1025, ranks=8
# =============================================================================
echo "=== Weak Scaling (513³/rank): N=1025, ranks=8 ==="

uv run python run_solver.py \
    +experiment=weak_scaling_jacobi \
    N=1025 n_ranks=8 strategy=sliced communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

uv run python run_solver.py \
    +experiment=weak_scaling_jacobi \
    N=1025 n_ranks=8 strategy=cubic communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

uv run python run_solver.py \
    +experiment=weak_scaling_fmg \
    N=1025 n_ranks=8 strategy=sliced communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

uv run python run_solver.py \
    +experiment=weak_scaling_fmg \
    N=1025 n_ranks=8 strategy=cubic communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

# =============================================================================
# N=1537, ranks=27
# =============================================================================
echo "=== Weak Scaling (513³/rank): N=1537, ranks=27 ==="

uv run python run_solver.py \
    +experiment=weak_scaling_jacobi \
    N=1537 n_ranks=27 strategy=sliced communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

uv run python run_solver.py \
    +experiment=weak_scaling_jacobi \
    N=1537 n_ranks=27 strategy=cubic communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

uv run python run_solver.py \
    +experiment=weak_scaling_fmg \
    N=1537 n_ranks=27 strategy=sliced communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

uv run python run_solver.py \
    +experiment=weak_scaling_fmg \
    N=1537 n_ranks=27 strategy=cubic communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

# =============================================================================
# N=2049, ranks=64
# =============================================================================
echo "=== Weak Scaling (513³/rank): N=2049, ranks=64 ==="

uv run python run_solver.py \
    +experiment=weak_scaling_jacobi \
    N=2049 n_ranks=64 strategy=sliced communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

uv run python run_solver.py \
    +experiment=weak_scaling_jacobi \
    N=2049 n_ranks=64 strategy=cubic communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

uv run python run_solver.py \
    +experiment=weak_scaling_fmg \
    N=2049 n_ranks=64 strategy=sliced communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

uv run python run_solver.py \
    +experiment=weak_scaling_fmg \
    N=2049 n_ranks=64 strategy=cubic communicator=custom \
    max_iter=$MAX_ITER hydra/launcher=basic mlflow=databricks

echo "Weak scaling experiments completed"

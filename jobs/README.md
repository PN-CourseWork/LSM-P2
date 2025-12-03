# HPC Job Scripts

LSF job scripts for DTU HPC cluster.

## Prerequisites

1. Clone repo to `$HOME/LSM-Project_2`
2. Create virtual environment: `uv venv && uv sync`
3. Configure Databricks credentials in `~/.databrickscfg`

## Job Scripts

### Strong Scaling (Jacobi)
- `strong_scaling_1node.sh` - 1-24 ranks on 1 node
- `strong_scaling_2node.sh` - 36-48 ranks on 2 nodes
- `strong_scaling_4node.sh` - 72-96 ranks on 4 nodes

### Weak Scaling (Jacobi)
- `weak_scaling.sh` - Constant work per rank: (1,257), (8,513), (27,769), (64,1025)

### FMG Scaling
- `fmg_scaling_1node.sh` - FMG solver on 1 node

## Submission

```bash
cd $HOME/LSM-Project_2
mkdir -p logs
bsub < jobs/strong_scaling_1node.sh
bsub < jobs/strong_scaling_2node.sh
bsub < jobs/strong_scaling_4node.sh
bsub < jobs/weak_scaling.sh
bsub < jobs/fmg_scaling_1node.sh
```

## Monitor Jobs

```bash
bstat              # Job status
bjobs -l JOB_ID    # Detailed info
bpeek JOB_ID       # View output
```

## MPI Configuration

All jobs use:
- `--map-by ppr:N:package` - N ranks per socket
- `--bind-to core` - Pin each rank to a core
- NUMBA_NUM_THREADS=1 - Single-threaded Numba kernel

## Experiment Configuration

- Numba JIT kernel (1 thread per rank for pure MPI scaling)
- Custom MPI datatypes for zero-copy halo exchange
- Databricks MLflow for tracking
- Hydra multirun for parameter sweeps (N, strategy)

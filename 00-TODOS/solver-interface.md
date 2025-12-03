# Solver Interface Architecture

Seamless integration between **Hydra** (config), **Solver** (compute), and **MLflow** (tracking).

---

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CONFIGURATION                                  │
│  conf/config.yaml ──→ Hydra validates ──→ GlobalParams dataclass            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                 SOLVER                                      │
│  solver = JacobiSolver(params: GlobalParams)                                │
│  solver.global_params   ← stored directly                                   │
│  solver.local_params    ← populated from MPI/grid                           │
│  solver.global_metrics  ← populated during solve                            │
│  solver.local_metrics   ← populated during solve                            │
│  log.info(...)          ← just use logging (Hydra captures it)              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRACKING (automatic)                              │
│                                                                             │
│  Hydra Callback (on_job_end):                                               │
│    mlflow.log_artifacts(hydra_output_dir)  ← config.yaml, logs, overrides   │
│                                                                             │
│  Solver Results:                                                            │
│    mlflow.log_params(asdict(solver.global_params))                          │
│    mlflow.log_metrics(asdict(solver.global_metrics))                        │
│    mlflow.log_table(all_local, "rank_topology.json")                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Dataclass Architecture

### 2×2 Matrix: Params vs Metrics × Global vs Local

```
                 Params (input/config)         Metrics (output/results)
                 ─────────────────────         ────────────────────────
Global           GlobalParams                  GlobalMetrics
(same across     N, solver, omega,             wall_time, mlups,
ranks / agg)     n_ranks, strategy...          converged, iterations...

Local            LocalParams                   LocalMetrics
(per-rank)       rank, hostname,               compute_times[],
                 neighbors, local_shape...     halo_times[]...
```

---

## Dataclass Definitions

### `GlobalParams` - Run Configuration

Immutable configuration set before the run. Identical across all MPI ranks.
Uses `MISSING` for required fields (Hydra will error if not provided in YAML).

```python
from dataclasses import dataclass
from omegaconf import MISSING

@dataclass
class GlobalParams:
    """Run configuration - validated by Hydra, logged to MLflow as params."""

    # Required (must be in YAML)
    N: int = MISSING

    # Solver defaults
    solver: str = "jacobi"  # "jacobi" | "fmg"
    omega: float = 0.8
    tolerance: float = 1e-6
    max_iter: int = 1000

    # Parallelization defaults
    n_ranks: int = 1
    strategy: str | None = None  # "sliced" | "cubic"
    communicator: str | None = None  # "numpy" | "custom"

    # Numba defaults
    use_numba: bool = False
    numba_threads: int = 1

    # Auto-detected at runtime
    environment: str = "local"  # "local" | "hpc"
```

### `LocalParams` - Rank-Specific Geometry

Per-rank topology information. Gathered for topology artifact.

```python
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

@dataclass
class LocalParams:
    """Per-rank geometry - gathered to rank 0, logged as artifact."""

    rank: int
    hostname: str
    cart_coords: Optional[Tuple[int, int, int]] = None
    neighbors: Dict[str, Optional[int]] = field(default_factory=dict)
    local_shape: Optional[Tuple[int, int, int]] = None
    global_start: Optional[Tuple[int, int, int]] = None
    global_end: Optional[Tuple[int, int, int]] = None
```

### `GlobalMetrics` - Aggregated Results

Final results computed/aggregated on rank 0. Logged to MLflow as metrics.

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class GlobalMetrics:
    """Aggregated results - logged to MLflow as metrics."""

    converged: bool = False
    iterations: int = 0
    final_error: Optional[float] = None
    wall_time: Optional[float] = None
    total_compute_time: Optional[float] = None
    total_halo_time: Optional[float] = None
    mlups: Optional[float] = None
    bandwidth_gb_s: Optional[float] = None
```

### `LocalMetrics` - Per-Rank Timeseries

Per-rank timing data. Different data types require different logging strategies.

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class LocalMetrics:
    """Per-rank timeseries - gathered for topology artifact."""

    # Per-rank (gathered to artifact for load balancing analysis)
    compute_times: List[float] = field(default_factory=list)
    halo_times: List[float] = field(default_factory=list)

    # Global (rank 0 only - logged as step metrics for convergence charts)
    residual_history: List[float] = field(default_factory=list)
```

#### Timeseries Logging Strategy

**Critical constraint:** No MLflow calls during solve - all logging happens post-solve.

```
DURING SOLVE (hot path)              AFTER SOLVE (cold path)
─────────────────────────            ────────────────────────
self.local_metrics.                  log_solver_results(solver, comm)
    compute_times.append(t)              ↓
    halo_times.append(t)             mlflow.log_params(...)
    residual_history.append(r)       mlflow.log_metrics(...)
                                     mlflow.log_table(...)
Just list.append() = ~0 overhead     All I/O happens here
```

| Data | Nature | Storage Location | Logging Method (post-solve) |
|------|--------|------------------|----------------|
| `residual_history` | Global (rank 0 only) | Rank 0 | Step-based metrics → MLflow charts |
| `compute_times` | Per-rank | Each rank | Gather **summary** (total, avg) → `log_table()` |
| `halo_times` | Per-rank | Each rank | Gather **summary** (total, avg) → `log_table()` |

**Why this split?**
- **Residual history** is used for convergence plots → step-based metrics enable MLflow's built-in charting
- **Timing data** is used for load balancing analysis → need all ranks' data in one place

---

## Hydra Integration

### Design Decision: Code-Centric Defaults

**Defaults live in dataclass, not YAML.** This means:
- Dataclass is the single source of truth
- YAML only specifies experiment-specific overrides
- Resolved config is logged as artifact for reproducibility

### Register with ConfigStore

```python
# In src/Poisson/datastructures.py
from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()
cs.store(name="global_params", node=GlobalParams)
```

### YAML Config (Minimal - Only Overrides)

```yaml
# conf/config.yaml
defaults:
  - _self_

# Only specify what's required or differs from defaults
N: 100

# Uncomment to override defaults:
# solver: fmg
# n_ranks: 4
# strategy: cubic
```

### Runner Entry Point

```python
# run_solver.py
import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Convert OmegaConf → typed dataclass (with validation!)
    params: GlobalParams = OmegaConf.to_object(cfg)

    # Auto-detect environment
    params.environment = (
        "hpc" if os.environ.get("LSB_JOBID") or os.environ.get("SLURM_JOB_ID")
        else "local"
    )

    # Just use logging - Hydra captures it, callback uploads it
    log.info(f"Config: {params}")

    # Create and run solver
    solver = create_solver(params)
    solver.solve()

    # Log solver results to MLflow (Hydra callback handles config/logs)
    log_solver_results(solver)
```

```yaml
# conf/config.yaml
defaults:
  - _self_
  - override hydra/callbacks: mlflow  # Enable MLflow callback

N: 100
```

---

## Solver Integration

### Solver Constructor

```python
class JacobiMPISolver:
    """MPI-parallel Jacobi solver."""

    def __init__(self, params: GlobalParams, comm: MPI.Comm = None):
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

        # Store global params directly (no extraction needed!)
        self.global_params = params

        # Initialize grid/decomposition
        self.grid = DistributedGrid(
            N=params.N,
            comm=self.comm,
            strategy=params.strategy or "sliced",
            halo_exchange=params.communicator or "custom",
        )

        # Populate local params from grid
        self.local_params = LocalParams(
            rank=self.rank,
            hostname=MPI.Get_processor_name(),
            cart_coords=tuple(self.grid.cart_comm.Get_coords(self.rank)),
            neighbors=self.grid.neighbors.copy(),
            local_shape=self.grid.local_shape,
            global_start=self.grid.global_start,
            global_end=self.grid.global_end,
        )

        # Initialize empty metrics
        self.global_metrics = GlobalMetrics()
        self.local_metrics = LocalMetrics()
```

### During Solve

```python
def solve(self):
    t_start = MPI.Wtime()

    for iteration in range(self.global_params.max_iter):
        # Track per-iteration timing
        t_compute = MPI.Wtime()
        self._compute_step()
        self.local_metrics.compute_times.append(MPI.Wtime() - t_compute)

        t_halo = MPI.Wtime()
        self.grid.sync_halos(self.u)
        self.local_metrics.halo_times.append(MPI.Wtime() - t_halo)

        # Check convergence...

    # Populate global metrics
    self.global_metrics.wall_time = MPI.Wtime() - t_start
    self.global_metrics.iterations = iteration + 1
    self._compute_performance_metrics()
```

---

## Logging Strategy

### Use `logging` + Hydra Callbacks (Cleanest)

**Principle:** Use Python's `logging` module everywhere. Hydra captures it automatically
and the callback uploads everything to MLflow.

```python
# In solver code - just use logging
import logging
log = logging.getLogger(__name__)

log.info(f"Starting solve: N={self.global_params.N}, ranks={self.global_params.n_ranks}")
log.debug(f"Iteration {i}: residual={residual:.2e}")
log.info(f"Converged in {iterations} iterations")
```

**What Hydra automatically saves:**
```
outputs/2024-12-03/15-30-00/
  .hydra/
    config.yaml      # Full resolved config
    hydra.yaml       # Hydra settings
    overrides.yaml   # CLI overrides
  main.log           # All logging output
```

### Hydra MLflow Callback

Create a callback that auto-logs everything to MLflow:

```yaml
# conf/hydra/callbacks/mlflow.yaml
mlflow_callback:
  _target_: utils.hydra.callbacks.MLflowCallback
```

```python
# src/utils/hydra/callbacks.py
from dataclasses import asdict
from hydra.core.utils import JobReturn
from hydra.experimental.callback import Callback
from omegaconf import DictConfig
import mlflow
import logging

log = logging.getLogger(__name__)

class MLflowCallback(Callback):
    """Hydra callback that auto-logs to MLflow."""

    def on_job_start(self, config: DictConfig, **kwargs) -> None:
        """Start MLflow run when Hydra job starts."""
        # MLflow run is started by the runner, not here
        pass

    def on_job_end(self, config: DictConfig, job_return: JobReturn, **kwargs) -> None:
        """Upload Hydra outputs to MLflow when job ends."""
        from hydra.core.hydra_config import HydraConfig

        try:
            output_dir = HydraConfig.get().runtime.output_dir

            # Upload entire Hydra output directory (config + logs)
            mlflow.log_artifacts(output_dir, artifact_path="hydra")
            log.info(f"Uploaded Hydra outputs to MLflow: {output_dir}")

        except Exception as e:
            log.warning(f"Could not upload Hydra outputs: {e}")
```

### MLflow Logging Function (Solver Results Only)

The callback handles Hydra outputs. This function handles solver-specific data:

```python
from dataclasses import asdict
import mlflow
import time

def log_solver_results(solver, comm=None):
    """Log solver results to MLflow. Hydra callback handles config/logs."""
    comm = comm or getattr(solver, 'comm', None)
    rank = comm.Get_rank() if comm else 0

    if rank == 0:
        # Params (from dataclass - no manual dict building!)
        mlflow.log_params(asdict(solver.global_params))

        # Metrics (filter None values)
        metrics = {k: v for k, v in asdict(solver.global_metrics).items() if v is not None}
        mlflow.log_metrics(metrics)

        # Residual history → step-based metrics (for MLflow charts)
        _log_residual_history(solver.local_metrics.residual_history)

    # Gather and log per-rank summary (MPI only)
    if comm and solver.global_params.n_ranks > 1:
        import pandas as pd

        # Build DataFrame from timeseries
        df = pd.DataFrame({
            "compute": solver.local_metrics.compute_times,
            "halo": solver.local_metrics.halo_times,
        })

        # Compute all stats at once, flatten to {col}_{stat} keys
        stats = df.agg(["sum", "mean", "std", "min", "max"]).T
        stats_flat = {f"{col}_{stat}": val for col, row in stats.iterrows() for stat, val in row.items()}

        local_data = {**asdict(solver.local_params), **stats_flat}
        all_local = comm.gather(local_data, root=0)

        if rank == 0:
            mlflow.log_table(all_local, artifact_file="rank_topology.json")


def _log_residual_history(residual_history: list):
    """Log residual history as step-based metrics for MLflow charts."""
    if not residual_history or not mlflow.active_run():
        return

    client = mlflow.tracking.MlflowClient()
    run_id = mlflow.active_run().info.run_id
    timestamp = int(time.time() * 1000)

    metrics = [
        mlflow.entities.Metric("residual", float(val), timestamp, step)
        for step, val in enumerate(residual_history)
    ]

    # Log in batches of 1000 (MLflow limit)
    for i in range(0, len(metrics), 1000):
        client.log_batch(run_id=run_id, metrics=metrics[i:i+1000], synchronous=True)
```

### What Gets Logged to MLflow

| Source | What | How | Purpose |
|--------|------|-----|---------|
| **Hydra callback** | `config.yaml`, `overrides.yaml`, `main.log` | Auto via `log_artifacts()` | Reproducibility |
| **Solver** | `GlobalParams` fields | `log_params(asdict(...))` | Experiment tracking |
| **Solver** | `GlobalMetrics` fields | `log_metrics(asdict(...))` | Performance comparison |
| **Solver** | Residual history | Step-based `log_batch()` | Convergence charts |
| **Solver** | Per-rank topology + timeseries | `log_table(all_local, ...)` | Load balancing analysis |

### No More Manual Logging!

```python
# OLD (manual, error-prone)
print(f"INFO: Starting run...")
with open("config.yaml", "w") as f:
    yaml.dump(config, f)
mlflow.log_artifact("config.yaml")

# NEW (automatic, clean)
log.info("Starting run...")  # Captured by Hydra → uploaded by callback
# Config already saved by Hydra → uploaded by callback
```

---

## Complete Flow Example

```python
# ══════════════════════════════════════════════════════════════
# 1. YAML CONFIG (conf/config.yaml)
# ══════════════════════════════════════════════════════════════
# N: 100
# solver: jacobi
# omega: 0.8
# n_ranks: 4
# strategy: cubic
# communicator: custom

# ══════════════════════════════════════════════════════════════
# 2. ENTRY POINT (run_solver.py)
# ══════════════════════════════════════════════════════════════
@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    global_params: GlobalParams = OmegaConf.to_object(cfg)

    if global_params.n_ranks == 1:
        run_sequential(global_params)
    else:
        spawn_mpi(global_params)

# ══════════════════════════════════════════════════════════════
# 3. MPI WORKER
# ══════════════════════════════════════════════════════════════
def run_mpi_solver(params: GlobalParams, comm: MPI.Comm):
    solver = JacobiMPISolver(params, comm)
    solver.solve()
    solver.compute_l2_error()

    with mlflow_run_context(...):
        log_to_mlflow(solver, comm)

# ══════════════════════════════════════════════════════════════
# 4. RESULT IN MLFLOW
# ══════════════════════════════════════════════════════════════
# Params:
#   N: 100
#   solver: jacobi
#   omega: 0.8
#   n_ranks: 4
#   strategy: cubic
#   communicator: custom
#   environment: hpc
#
# Metrics:
#   converged: 1
#   iterations: 1523
#   wall_time: 12.34
#   mlups: 156.7
#   bandwidth_gb_s: 10.03
#
# Artifacts:
#   resolved_config.yaml (full config for reproducibility)
#   rank_topology.json (per-rank timing/topology data)
```

---

## Migration Checklist

### Dataclasses - STATUS: ✅ COMPLETED
- [x] `GlobalParams` - run configuration dataclass ✓
- [x] `GlobalMetrics` - aggregated results dataclass ✓
- [x] `LocalMetrics` - per-iteration timeseries dataclass ✓
- [x] `solver.metrics` instead of `solver.results` ✓

### Dataclasses to Keep
- [x] `GridLevel` (multigrid-specific) ✓

### Files Updated - STATUS: ✅ COMPLETED
- [x] `src/Poisson/datastructures.py` - new dataclass definitions ✓
- [x] `src/Poisson/solvers/base.py` - uses `GlobalMetrics` and `LocalMetrics` ✓
- [x] `src/Poisson/solvers/jacobi.py` - uses `solver.metrics` ✓
- [x] `src/Poisson/solvers/fmg.py` - uses `solver.metrics` ✓
- [x] `src/Poisson/solvers/jacobi_mpi.py` - uses `solver.metrics` ✓
- [x] `src/Poisson/solvers/fmg_mpi.py` - uses `solver.metrics` ✓
- [x] `run_solver.py` - Orchestrator/Worker architecture with native MLflow ✓
- [x] `conf/config.yaml` - flattened to match `GlobalParams` ✓
- [x] `log_to_mlflow()` in run_solver.py - uses `asdict()` pattern ✓

---

## Benefits Summary

| Aspect | Before | After |
|--------|--------|-------|
| Config validation | Runtime errors | Hydra validates at load |
| Parameter extraction | `cfg.get("problem", {}).get("N")` | `params.N` |
| MLflow params | Manual dict building | `asdict(params)` |
| Type hints | None | Full IDE support |
| Adding new param | Edit 5 places | Edit dataclass only |
| Default values | Duplicated in YAML + code | Single source (dataclass) |
| Debugging | Guess what config was used | `resolved_config.yaml` artifact |
| Reproducibility | Hope YAML wasn't changed | Exact config stored per run |

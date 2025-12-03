# Final Sprint Checklist

**Deadline:** December 3, 2025 (midnight)

Cross-referenced from: `todos.md`, GitHub Issues, Course Materials, Assignment-1 Feedback

---

## Phase 1: Code Readiness (BLOCKING)

These must be done before HPC runs.

### 1.1 MLUPS/Bandwidth Calculation (todos.md #1)
- [x] **Fix bytes-per-stencil-update to match course** ✓
  - Changed from 32 to 64 bytes/point in `base.py:41-43`
  - **Course Week 02 (p.17):** "each lup involves 8 memory operations: 7 loads and 1 store"
  - Now: `bytes_per_point = 64` (8 mem-ops × 8 bytes)

### 1.2 Wall Time Verification (todos.md #17)
- [x] **Verify timing methodology** ✓
  - Using `MPI.Wtime()` ✓ (correct for MPI apps)
  - **Course Week 03 (p.7-8):** Be specific about what timing includes/excludes
  - Timing excludes init, measures only solve iterations ✓
  - Document in report what is/isn't timed

- [ ] **Verify timing breakdown sums to wall time**
  ```
  wall_time ≈ compute_time + halo_time + other
  ```
  - **Compute time:** Jacobi stencil updates (the actual kernel work)
  - **Halo time:** Halo exchange (`Sendrecv` for ghost cells)
  - **Other:** Residual computation, convergence Allreduce, loop overhead, array swaps

  **Note:** Residual `Allreduce` is currently **blocking** and **not timed separately**.
  - Currently in `_compute_residual()` without timing wrapper
  - Options: (1) time it separately, (2) use non-blocking `Iallreduce`, (3) accept as "other"
  - For now, accept as "other" - it's convergence checking overhead, not core solve

  - Run a test and compare: `compute_total + halo_total` vs `wall_time`
  - Document breakdown in report (e.g., "90% compute, 7% halo, 3% other")

### 1.3 Numba Thread Control (todos.md #29)
- [x] **Set `NUMBA_NUM_THREADS` explicitly** ✓
  - **Assignment-1 Feedback:** "from the report it is not clear how many threads you are using"
  - "if you introduce another threading/parallel scheme, it is vital to understand the implications"
  - Add: `os.environ["NUMBA_NUM_THREADS"] = str(n_threads)` before numba imports ✓
  - Document in report: MPI ranks × Numba threads ≤ allocated cores

- [x] **Ensure Numba warmup before timing** ✓
  - JIT compilation happens on first call - must not be included in timing
  - `solver.warmup()` called before solve in run_solver.py ✓
  - Warmup is called in all experiment runners ✓

### 1.4 FMG Decomposition (GH #33)
- [x] **Test FMG with cubic decomposition** ✓
  - Tested: `solver=multigrid solver.strategy=cubic n_ranks=8` works correctly
  - FMG works with both sliced and cubic decomposition

### 1.5 MPI Communication Style (Assignment-1 Feedback + mpi4py docs) - CRITICAL
- [ ] **Use uppercase methods in solve loop (performance-critical)**
  - **Assignment-1 Feedback:** "your code uses basically only send and recv (lower-case), these have a performance hit as they have to serialize your data. Always prefer titlelized versions"
  - **mpi4py Tutorial:** "You have to use method names starting with an upper-case letter"

  | Lowercase (pickle) | Uppercase (buffer) | When OK to use lowercase |
  |--------------------|--------------------|--------------------------|
  | `comm.send()` | `comm.Send()` | Setup/teardown only |
  | `comm.recv()` | `comm.Recv()` | Setup/teardown only |
  | `comm.gather()` | `comm.Gather()` | For Python dicts/objects |
  | `comm.allreduce()` | `comm.Allreduce()` | **Never in solve loop** |

  **Rule of thumb:**
  - **In solve loop / hot path:** Always uppercase (NumPy arrays, zero-copy)
  - **Setup / finalize / one-time ops:** Lowercase OK for Python objects (dicts, lists)

- [x] **Verify solve loop uses uppercase** ✓
  - `jacobi_mpi.py:93` → `Allreduce` in `_compute_residual()` (hot path) ✓
  - `fmg_mpi.py:183` → `Allreduce` in residual computation (hot path) ✓
  - `halo.py` → `Sendrecv` for halo exchange (hot path) ✓
  - `grid.py:157` → `allreduce` in `compute_l2_error()` - **OK, only called once at end**

### 1.6 Non-blocking Request Handles (Generic Feedback)
- [x] **Verify proper Wait/Test on all request handles** ✓
  - Generic Feedback: "When you have a request, you should not overwrite it until you have issued a Wait or a Test on it"
  - **Status:** No `Isend`/`Irecv` used - code uses blocking `Sendrecv` pattern in `halo.py`
  - This is safe and doesn't require request handle management

### 1.7 Rank Topology Logging for MLflow (todos.md #3, #15)
- [x] **Extend `RankGeometry` or create new dataclass for topology info** ✓
  - Uses existing `LocalParams` dataclass from `datastructures.py`
  - Added `get_rank_info()` method to `DistributedGrid` (`grid.py:108-118`)
  - Returns `LocalParams` with rank, hostname, cart_coords, neighbors, local_shape, global_start/end

- [x] **Gather to rank 0 and save as artifact** ✓
  - Added `gather_rank_topology()` in `run_solver.py:148-168`
  - Uses `comm.gather()` to collect `LocalParams` from all ranks
  - Logs JSON artifact to `topology/` folder in MLflow

- [ ] **Enable networkx visualization later**
  - Neighbor relationships → graph edges
  - Same hostname → same node cluster
  - Same socket_id → same socket cluster
  - Edge weights → halo communication times

### 1.8 Code Polishing & Verification
- [x] **Run ruff format** - `uv run ruff format src/` ✓ (14 files reformatted)
- [x] **Run ruff check with autofix** - `uv run ruff check src/ --fix` ✓
- [x] **Run ruff lint (final check)** - All checks passed ✓
- [x] **Verify locally** ✓
  - Sequential Jacobi: 186 Mlup/s
  - MPI Jacobi (4 ranks): 311 Mlup/s
  - Sequential FMG: 0.2 Mlup/s (6 iterations to 1.16e-03 error)

---

## Phase 2: HPC Experiments

### 2.1 Scaling Experiments (GH #14, #34) - CRITICAL

**Reference: Week 08 + [HPC Wiki Scaling](https://hpc-wiki.info/hpc/Scaling)** ← "Read every week!"

#### Strong Scaling (Amdahl's Law)
- [ ] Fixed problem sizes: 250³, 500³
- [ ] Sweep ranks: 1, 2, 4, 8, **12**, 16, **24**, 27, 32, **36**, **48**, 64
  - **Use multiples of 12** to match socket size (12 cores/socket)
  - Assignment-1 Feedback: "if you would have used a rank increment of 12, you would have matched the socket size"
- [ ] Plot: **Speedup = T(1)/T(P)** vs P processors (**log-log scale!**)
- [ ] Plot: **Efficiency = Speedup/P** vs P processors
- [ ] Formula: `Speedup = 1 / (s + p/P)` where s=serial fraction, p=parallel fraction

#### Weak Scaling (Gustafson's Law)
- [ ] Constant work per rank (scale problem with ranks)
- [ ] More datapoints needed (GH #34)
- [ ] Plot: **Efficiency = T(1)/T(P)** vs P processors (should stay ~1.0)
- [ ] Formula: `Scaled Speedup = s + p×P`

**CRITICAL FORMAT (from feedback):**
- ❌ wall-time vs processors is **NOT** a scaling plot!
- ✓ Scaling plots: **x = processors, y = T(1)/T(P), log-log scale!**
- ✓ Efficiency/overhead should be **unit-less** (fractions, not raw times)

**Scale Requirements:**
- "Go large-scale, up to at least **4 nodes**, or as high as you can go"
- 4 nodes = 8 sockets = 96 cores
- "Always consider how chunk-size influences work-load"

#### Node Crossing Analysis (Assignment-1 Feedback)
- [ ] **Discuss performance when crossing node boundaries**
  - "how did it perform when crossing from 2 nodes to 3 nodes? Comments there would have fitted"
  - Compare: 12 ranks (1 node) → 24 ranks (2 nodes) → 36 ranks (3 nodes)
  - Explain: inter-node communication via Infiniband vs intra-node shared memory

### 2.2 Communication Experiments (GH #30)

- [ ] **Larger problem sizes** for communication experiment
- [ ] **Communicator comparison:** numpy vs custom MPI datatypes
- [ ] **Socket binding strategy** (todos.md #3)
  - **Week 02 (p.44):** Use `mpirun --map-by ppr:1:package` for max bandwidth
  - Compare compact vs spread placement
  - Use rank counts matching socket: 12, 24, 36...

### 2.3 Rank Placement Logging (todos.md #3, #15)
- [x] **Log per-rank info in solver:** ✓
  - Hostname ✓ (via `MPI.Get_processor_name()`)
  - Cart coords ✓ (via `cart_comm.Get_coords()`)
  - Neighbor mapping ✓ (from `neighbors` dict)
  - Local shape and global indices ✓
  - Logged as JSON artifact in `topology/` folder
- [ ] Consider adding socket_id/cpu_model for HPC runs
- [ ] Consider networkx visualization for rank topology

### 2.4 Hybrid Experiments
- [ ] **Numba threads sweep:** 1, 4, 8 threads
- [ ] Ensure: MPI_ranks × Numba_threads ≤ allocated_cores
- [ ] Document exact configuration in report

---

## Phase 3: Plotting & Analysis

### 3.1 Required Plots

- [ ] **MLUPS vs problem size** (GH #32)
  - Compute bandwidth: `bandwidth_GB_s = mlups × 64 / 1000`
  - Compare to hardware max: ~76.8 GB/s per socket (Week 02, p.6)

- [ ] **Surface-to-volume scaling** (GH #35, todos.md #11)
  - Halo size (surface) vs local volume
  - Communication overhead vs compute ratio

- [ ] **Strong scaling plot**
  - X: Number of processors (log scale)
  - Y: Speedup or Efficiency
  - NOT just wall-time vs ranks!

- [ ] **Weak scaling plot**
  - X: Number of processors
  - Y: Efficiency (should be ~1.0 for ideal)

- [ ] **Halo exchange timing**
  - Wall time per iteration vs problem size
  - Compare numpy vs custom communicator

### 3.2 Plot Quality (Assignment-1 + Generic Feedback)
- [ ] **Readable font sizes** - "plots are practically unreadable in print-out format"
- [ ] **Proper tick marks** - "would have benefited from more ticks", "Number of ticks and tick-labels are also important"
- [ ] **Log-log where appropriate** - "log-log scale!" for scaling plots
- [ ] **Consider binning/normalization** for complex data
- [ ] **Unit-less efficiency/overhead** - "overhead/efficiency plots are typically best in unit-less data. Convert timings into fraction of overhead"
- [ ] **Be clear about overhead baseline** - "be clear whether it is overhead compared to total time, or overhead compared to compute time"

---

## Phase 4: Report Writing

### 4.1 Required Content (CRITICAL - from feedback)

- [ ] **Hardware description** (Assignment-1 Feedback: "no description of hardware/compiler used")
  - CPU type/model: Intel Xeon E5-2650 v4 (12 cores/socket, 2 sockets/node)
  - Memory per node: 256 GB (128 GB per socket)
  - Memory bandwidth: 76.8 GB/s per socket (DDR4-2400, 4 channels)
  - Network: FDR Infiniband (56 Gb/s)
  - MPI version: OpenMPI (specify version from `module list`)
  - Python/Numba versions

- [ ] **Explain Barrier before timing** (generic_feedback.pdf)
  - "Many have the correct usage of a Barrier before the initial timing. But there are no discussions on why"
  - Add: ensures all ranks start timing simultaneously, prevents early ranks from including wait time

- [ ] **Bibliography** (todos.md #7)
  - Multigrid: Briggs, Henson & McCormick - "A Multigrid Tutorial" (2nd ed., SIAM 2000)
  - Alternative: Trottenberg, Oosterlee, Schüller - "Multigrid" (Academic Press 2001)

- [ ] **Parallel IO considerations** (GH #29)
  - Explain result gathering excluded from timings
  - Reference Week 09 for parallel IO approaches

- [ ] **Design decisions** (todos.md #19, #21)
  - Never gather global solution - use parallel validation
  - Custom MPI types vs numpy halo exchange trade-offs

### 4.2 Report Sections Status

**Currently DONE in report:**
- [x] Kernel comparison (NumPy vs Numba) - Section complete with figures
- [x] Domain decomposition (sliced vs cubic) - Section complete with figures
- [x] Communication strategy (datatypes vs buffers) - Section complete with figures
- [x] Solver validation - Section complete with figures

**Currently INCOMPLETE/TODO in report:**
- [ ] **Scaling analysis section** - Only template, no actual content
- [ ] **Rank placement/mapping section** - Only TODO notes
- [ ] **Full Multigrid section** - Only TODO notes
- [ ] **Conclusion/Summary** - Empty

### 4.3 Analysis Points (from feedback)

- [ ] **Socket matching discussion** (Assignment-1 Feedback)
  - "if you would have used a rank increment of 12, you would have matched the socket size"
  - Explain what happens at 12, 24, 36, 48 ranks (socket boundaries)

- [ ] **Sockets vs nodes discussion** (Generic Feedback)
  - "A more thorough discussion on sockets vs. nodes"
  - "touch on the sockets when you hit multiples of 12"
  - Intra-socket: shared L3 cache, fast
  - Inter-socket (same node): QPI link, slower
  - Inter-node: Infiniband, slowest

- [ ] **Negative results** (Generic Feedback)
  - "Negative results are also results, it's typically fair to highlight these results so that others won't do the same mistake!"
  - If cubic doesn't outperform sliced at some rank counts, explain why

- [ ] **Data placement optimization** (Generic Feedback)
  - "All of you did a receive on rank == 0 in a buffer, why? What about receiving data right where you need it?"
  - Discuss: we DON'T gather global solution (parallel validation instead)

- [ ] **Document Numba thread usage clearly** (Assignment-1 Feedback)
  - "from the report it is not clear how many threads you are using"
  - State explicitly: "All experiments use NUMBA_NUM_THREADS=8 unless noted"
  - Document: MPI ranks × Numba threads ≤ allocated cores

---

## Phase 5: Code Quality (GH #28)

Lower priority - do if time permits.

### 5.1 Critical Code Cleanup (HIGH)

| Issue | Location | Fix |
|-------|----------|-----|
| **Duplicate functions** | `logs.py` vs `upload_logs.py` | Both have `upload_logs()` with different interfaces. Consolidate into one. |
| **Print-based logging** | `io.py:43,56,58,82,84,106...` | Replace `print("INFO: ...")` with `logging.info()`. Use `logging.getLogger(__name__)`. |
| **Bare except clause** | `logs.py:66-67` | `except Exception: pass` - at minimum log the error, don't swallow silently. |

### 5.2 Configuration Issues (MEDIUM)

| Issue | Location | Fix |
|-------|----------|-----|
| **Hardcoded workspace prefix** | `io.py:71,200` | `"/Shared/LSM-PoissonMPI-v2"` appears multiple times. Move to config. |
| **String-based filter construction** | `io.py:99` | `f"tags.mlflow.runName = '{parent_run_name}'"` - use parameterized queries if possible. |

### 5.3 OS Compatibility (MEDIUM)

| Issue | Location | Fix |
|-------|----------|-----|
| **Unix-only code** | `main.py:118` | `preexec_fn=os.setsid` only works on Unix. Add Windows fallback or guard. |

### 5.4 Dataclass Architecture Refactor (HIGH - Architectural) ✅ COMPLETED

**Problem:** Current dataclasses are confusingly named and overlap:
- `KernelParams` vs solver params - what's the difference?
- `KernelMetrics` vs `GlobalMetrics` - redundant
- Manual dict-building with conditionals in `_log_results()`

**Solution:** Clean 2×2 matrix: **Params vs Metrics** × **Global vs Local**

**STATUS:** Refactored in run_solver.py and datastructures.py:
- `GlobalParams` dataclass for run configuration ✓
- `GlobalMetrics` dataclass for aggregated results ✓
- `LocalMetrics` dataclass for per-iteration timeseries ✓
- `solver.metrics` instead of `solver.results` ✓
- Clean MLflow logging with `asdict()` pattern ✓
- Flattened Hydra config (removed solver groups) ✓
- Native MLflow batch API for timeseries ✓

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

#### Dataclass Definitions

- [ ] **`GlobalParams`** - Run configuration (identical across ranks):
  ```python
  @dataclass
  class GlobalParams:
      # Problem
      N: int
      # Solver
      solver: str  # "jacobi" | "fmg"
      omega: float = 0.8
      tolerance: float = 1e-6
      max_iter: int = 1000
      # Parallelization
      n_ranks: int = 1
      strategy: str | None = None  # "sliced" | "cubic"
      communicator: str | None = None  # "numpy" | "custom"
      # Numba
      use_numba: bool = False
      numba_threads: int = 1
      # Environment (auto-detected)
      environment: str = "local"  # "local" | "hpc"
  ```

- [ ] **`LocalParams`** - Rank-specific geometry/topology:
  ```python
  @dataclass
  class LocalParams:
      rank: int
      hostname: str
      cart_coords: tuple[int, int, int] | None = None
      neighbors: dict[str, int | None] = field(default_factory=dict)
      local_shape: tuple[int, int, int] | None = None
      global_start: tuple[int, int, int] | None = None
      global_end: tuple[int, int, int] | None = None
  ```

- [ ] **`GlobalMetrics`** - Aggregated results (rank 0):
  ```python
  @dataclass
  class GlobalMetrics:
      converged: bool = False
      iterations: int = 0
      final_error: float | None = None
      wall_time: float | None = None
      total_compute_time: float | None = None
      total_halo_time: float | None = None
      mlups: float | None = None
      bandwidth_gb_s: float | None = None
  ```

- [ ] **`LocalMetrics`** - Per-rank timeseries:
  ```python
  @dataclass
  class LocalMetrics:
      compute_times: list[float] = field(default_factory=list)
      halo_times: list[float] = field(default_factory=list)
      residual_history: list[float] = field(default_factory=list)  # rank 0 only
  ```

#### Solver Integration

- [ ] **Solver stores all four**:
  ```python
  # In solver.__init__()
  self.global_params = GlobalParams(N=N, solver="jacobi", ...)
  self.local_params = LocalParams(
      rank=self.rank,
      hostname=MPI.Get_processor_name(),
      neighbors=self.grid.neighbors,
      local_shape=self.grid.local_shape,
  )
  self.global_metrics = GlobalMetrics()
  self.local_metrics = LocalMetrics()
  ```

#### MLflow Logging

- [ ] **Clean logging** - no more manual dict building:
  ```python
  def _log_to_mlflow(solver, comm):
      # Global params → MLflow params
      mlflow.log_params(asdict(solver.global_params))

      # Global metrics → MLflow metrics
      mlflow.log_metrics(asdict(solver.global_metrics))

      # Gather local data for topology artifact
      local_data = {
          "params": asdict(solver.local_params),
          "metrics": {
              "compute_time_total": sum(solver.local_metrics.compute_times),
              "halo_time_total": sum(solver.local_metrics.halo_times),
          }
      }
      all_local = comm.gather(local_data, root=0)

      if solver.global_params.n_ranks > 1 and comm.Get_rank() == 0:
          # Flatten for DataFrame
          rows = []
          for d in all_local:
              row = d["params"].copy()
              row.update(d["metrics"])
              rows.append(row)
          df = pd.DataFrame(rows)
          df.to_parquet("rank_topology.parquet")
          mlflow.log_artifact("rank_topology.parquet")
  ```

#### Hydra Structured Configs (Seamless Integration)

The key to seamless Hydra ↔ Solver ↔ MLflow integration:

- [ ] **Register `GlobalParams` with Hydra's ConfigStore**:
  ```python
  # In datastructures.py or a new configs.py
  from hydra.core.config_store import ConfigStore

  cs = ConfigStore.instance()
  cs.store(name="global_params", node=GlobalParams)
  ```

- [ ] **Update YAML configs to match dataclass**:
  ```yaml
  # conf/config.yaml
  defaults:
    - _self_

  # These fields now validated against GlobalParams schema
  N: 100
  solver: jacobi
  omega: 0.8
  tolerance: 1e-6
  max_iter: 1000
  n_ranks: 1
  strategy: null
  communicator: null
  use_numba: false
  numba_threads: 1
  ```

- [ ] **Convert OmegaConf → dataclass in runner**:
  ```python
  from omegaconf import OmegaConf

  @hydra.main(config_path="conf", config_name="config")
  def main(cfg: DictConfig):
      # Convert to typed dataclass (with validation!)
      global_params = OmegaConf.to_object(cfg)  # Returns GlobalParams instance

      # Or if nested config:
      # global_params = OmegaConf.structured(GlobalParams(**cfg.solver))

      solver = create_solver(global_params)
      solver.solve()
  ```

- [ ] **Solver accepts dataclass directly**:
  ```python
  class JacobiMPISolver:
      def __init__(self, params: GlobalParams, comm: MPI.Comm = None):
          self.global_params = params
          self.global_metrics = GlobalMetrics()
          # ... rest of init uses self.global_params.N, etc.
  ```

**Result: Complete flow with zero manual wrangling**:
```
conf/config.yaml
      ↓ (Hydra loads + validates)
OmegaConf.to_object(cfg) → GlobalParams
      ↓ (passed to solver)
solver.global_params
      ↓ (logged directly)
mlflow.log_params(asdict(solver.global_params))
```

#### Migration

- [ ] **Remove deprecated dataclasses**:
  - `KernelParams` → absorbed into `GlobalParams`
  - `KernelMetrics` → absorbed into `GlobalMetrics`
  - `KernelSeries` → absorbed into `LocalMetrics`
  - `LocalSeries` → renamed to `LocalMetrics`
  - `RankGeometry` → absorbed into `LocalParams`

- [ ] **Keep `GridLevel`** (multigrid-specific, still needed)

### 5.5 Run Tagging: HPC vs Local (BLOCKING for analysis)

- [x] **Tag all runs with `environment`**: `"hpc"` or `"local"` ✓
  - Implemented in `io.py:114-120`
  - Auto-detects via `LSB_JOBID` (LSF) or `SLURM_JOB_ID` (Slurm)
  - Filter with: `mlflow.search_runs(filter_string="tags.environment = 'hpc'")`

### 5.6 Documentation Standards

- [ ] **Detailed NumPy docstrings only for exposed API functions**
  - Functions included in `API_reference.rst` get full docstrings (Parameters, Returns, Examples)
  - Internal/helper functions get simple one-line docstrings

- [ ] **Use `_` prefix consistently for internal functions**
  - Public API: `solve()`, `compute_l2_error()`, `create_solver()`
  - Internal: `_smooth()`, `_v_cycle()`, `_compute_residual()`, `_sync_halos()`

- [ ] **Simplify `__init__.py` exports**
  - Only expose what users need (solvers, main dataclasses, key utilities)
  - Don't expose internal helpers, base classes, or implementation details
  - Reduces API surface and makes docs cleaner

### 5.7 Non-blocking Halo Exchange (OPTIONAL)

- [ ] **Implement non-blocking halo exchange for compute/comm overlap**

  Current: blocking `Sendrecv` in `halo.py`

  Change to:
  ```python
  def start_exchange(self, arr, comm, neighbors) -> list[MPI.Request]:
      reqs = []
      reqs.append(comm.Irecv(recv_buf, src, tag))  # post receives first
      reqs.append(comm.Isend(send_buf, dest, tag))
      return reqs

  def finish_exchange(self, reqs):
      MPI.Request.Waitall(reqs)
  ```

  Usage in solver:
  ```python
  reqs = halo.start_exchange(u, comm, neighbors)
  kernel.step_interior(u, u_temp, f)  # [2:-2, 2:-2, 2:-2] - no halos needed
  halo.finish_exchange(reqs)
  kernel.step_boundary(u, u_temp, f)  # outer shell - needs halos
  ```

  **Effort:** ~2 hours (kernel split is tedious)
  **Benefit:** Overlap comm with compute - only worthwhile if halo time is significant %

### 5.8 General Cleanup

- [ ] Update `API_reference.rst` to match simplified exports
- [ ] Remove redundant `palettes.py`
- [ ] Consistent figure sizes
- [ ] Better DataFrame column naming
- [ ] stdout/stderr always logged to MLflow
- [ ] Check `allgather` (lowercase) efficiency - should use `Allgather`
- [ ] Clean up git repo (todos.md #25)

---

## Course Material Reference

| Topic | Source | Key Content |
|-------|--------|-------------|
| Memory bandwidth | Week 02, pp.4-19 | 64 bytes/lup, memory-bound scaling |
| Process placement | Week 02, pp.30-47 | `--map-by ppr:n:package`, socket binding |
| Scaling laws | Week 08, pp.3-7 | Amdahl (strong), Gustafson (weak), formulas |
| Scaling reference | [hpc-wiki.info/hpc/Scaling](https://hpc-wiki.info/hpc/Scaling) | "Read every week!" (Week 08) |
| Cartesian maps | Week 07, pp.7-8 | `Create_cart`, `Compute_dims`, `Shift` |
| Custom datatypes | Week 08, pp.9-16 | `Create_contiguous`, `Create_indexed` |
| Parallel IO | Week 09, pp.2-8 | MPI.File, collective vs local writes |
| Profiling | Week 09, pp.9-22 | cProfile, mpiP, per-rank analysis |
| MPI basics | Week 04 | Send/Recv, collectives, non-blocking |
| Jacobi method | Poisson_problem_intro.pdf | 7-point stencil, convergence |
| Assignment reqs | Project2.pdf | Mlup/s, wall-clock, CPU type required |
| **mpi4py uppercase** | [mpi4py Tutorial](https://mpi4py.readthedocs.io/en/stable/tutorial.html) | Uppercase=fast buffer, lowercase=slow pickle |
| **Generic Feedback** | generic_feedback.pdf | Scaling plots, request handles, sockets, Barrier |
| **Assignment-1 Feedback** | Feedback.pdf | Numba threads, plot quality, hardware desc |

---

## GitHub Issues Summary

| GH # | Title | Phase |
|------|-------|-------|
| 35 | Surface-to-volume plot | 3 |
| 34 | Weak scaling more datapoints | 2 |
| 33 | FMG: Only sliced custom decomp | 1 |
| 32 | Create MLUPS plots | 3 |
| 31 | Allow non-perfect cubes | 2 |
| 30 | Communication exp: larger problems | 2 |
| 29 | Report: Parallel IO considerations | 4 |
| 28 | Code structure and quality | 5 |
| 14 | Scaling Analysis | 2,3 |

---

## Key Formulas

```
MLUPS = (N-2)³ × iterations / (wall_time × 10⁶)

Bandwidth (GB/s) = MLUPS × bytes_per_lup / 1000
  where bytes_per_lup = 64 (7 loads + 1 store × 8 bytes)

Strong Scaling Speedup = T(1) / T(N)
Strong Scaling Efficiency = Speedup / N

Weak Scaling Efficiency = T(1) / T(N)  [problem scales with N]

Amdahl's Law: Speedup = 1 / (s + p/N)
Gustafson's Law: Speedup = s + p×N
```

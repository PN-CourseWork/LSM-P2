# Final Sprint Checklist

**Deadline:** December 3, 2025 (midnight)

Cross-referenced from: `todos.md`, GitHub Issues, Course Materials, Assignment-1 Feedback

---

## Phase 1: Code Readiness (BLOCKING)

These must be done before HPC runs.

### 1.1 MLUPS/Bandwidth Calculation (todos.md #1)
- [ ] **Fix bytes-per-stencil-update to match course**
  - Current code: 32 bytes/point - `base.py:41-44`
  - **Course Week 02 (p.17):** "each lup involves 8 memory operations: 7 loads and 1 store"
  - **Decision: Use 64 bytes** (matches course material)

  **Code change needed in `src/Poisson/solvers/base.py:41-44`:**
  ```python
  # OLD (line 41-42):
  # 4 arrays * 8 bytes = 32 bytes per point
  bytes_per_point = 32

  # NEW:
  # 7-point stencil: 6 neighbors + center + f = 8 mem-ops × 8 bytes
  bytes_per_point = 64
  ```

  **Impact:** Reported bandwidth will **double** (this is correct - reflects actual memory traffic)

### 1.2 Wall Time Verification (todos.md #17)
- [ ] **Verify timing methodology**
  - Using `MPI.Wtime()` ✓ (correct for MPI apps)
  - **Course Week 03 (p.7-8):** Be specific about what timing includes/excludes
  - Ensure timing: excludes init, measures only solve iterations
  - Document in report what is/isn't timed

### 1.3 Numba Thread Control (todos.md #29)
- [ ] **Set `NUMBA_NUM_THREADS` explicitly**
  - **Assignment-1 Feedback:** "from the report it is not clear how many threads you are using"
  - "if you introduce another threading/parallel scheme, it is vital to understand the implications"
  - Add: `os.environ["NUMBA_NUM_THREADS"] = str(n_threads)` before numba imports
  - Document in report: MPI ranks × Numba threads ≤ allocated cores

### 1.4 FMG Decomposition (GH #33)
- [ ] **Test FMG with cubic decomposition**
  - Currently only works with sliced + custom?
  - Either fix or document limitation in report

### 1.5 MPI Communication Style (Assignment-1 Feedback + mpi4py docs)
- [ ] **Use uppercase methods in solve loop (performance-critical)**
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
  - `grid.py:157` → `allreduce` in `compute_l2_error()` - **OK, only called once at end**

### 1.6 Non-blocking Request Handles (Generic Feedback)
- [x] **Verify proper Wait/Test on all request handles** ✓
  - Generic Feedback: "When you have a request, you should not overwrite it until you have issued a Wait or a Test on it"
  - **Status:** No `Isend`/`Irecv` used - code uses blocking `Sendrecv` pattern in `halo.py`
  - This is safe and doesn't require request handle management

### 1.7 Rank Topology Logging for MLflow (todos.md #3, #15)
- [ ] **Extend `RankGeometry` or create new dataclass for topology info**

  **Already exists** (`datastructures.py:144-170`, `grid.py:192-201`):
  | Field | Source |
  |-------|--------|
  | `rank` | `RankGeometry.rank` |
  | `local_shape` | `RankGeometry.local_shape` |
  | `neighbors` | `RankGeometry.neighbors` |
  | `halo_time_total` | `GlobalMetrics.total_halo_time` |

  **Needs to be added:**
  | Column | How to get |
  |--------|------------|
  | `hostname` | `MPI.Get_processor_name()` |
  | `cart_coords` | `grid.cart_comm.Get_coords(rank)` |
  | `socket_id` | Parse `/proc/self/status` for `Cpus_allowed` or use `psutil` |
  | `cpu_model` | `platform.processor()` or parse `/proc/cpuinfo` |
  | `halo_time_per_iter` | `total_halo_time / iterations` |

- [ ] **Gather to rank 0 and save as artifact**
  ```python
  # In solver finalize (after solve completes)
  rank_info = {
      'rank': self.rank,
      'hostname': MPI.Get_processor_name(),
      'cart_coords': list(self.grid.cart_comm.Get_coords(self.rank)),
      'neighbors': self.grid.neighbors,
      'local_shape': self.grid.local_shape,
      'halo_time_total': self.results.total_halo_time,
      'halo_time_per_iter': self.results.total_halo_time / self.results.iterations,
  }
  all_rank_info = self.comm.gather(rank_info, root=0)  # Use lowercase gather for dict
  if self.rank == 0:
      df = pd.DataFrame(all_rank_info)
      df.to_parquet("rank_topology.parquet")
      mlflow.log_artifact("rank_topology.parquet")
  ```

- [ ] **Enable networkx visualization later**
  - Neighbor relationships → graph edges
  - Same hostname → same node cluster
  - Same socket_id → same socket cluster
  - Edge weights → halo communication times

### 1.8 Code Polishing & Verification
- [ ] **Run ruff format** - `uv run ruff format src/`
- [ ] **Run ruff check with autofix** - `uv run ruff check src/ --fix`
- [ ] **Run ruff lint (final check)** - `uv run ruff check src/`
- [ ] **Verify locally** - Run experiments to confirm everything works:
  ```bash
  # Sequential test
  uv run python main.py solver=jacobi N=50 max_iter=100

  # MPI test (4 ranks)
  mpiexec -n 4 uv run python main.py solver=jacobi_mpi N=50 max_iter=100
  ```

---

## Phase 2: HPC Experiments

### 2.1 Scaling Experiments (GH #14, #34)

**Reference: Week 08 + [HPC Wiki Scaling](https://hpc-wiki.info/hpc/Scaling)**

#### Strong Scaling (Amdahl's Law)
- [ ] Fixed problem sizes: 250³, 500³
- [ ] Sweep ranks: 1, 2, 4, 8, 12, 16, 24, 27, 32, 48, 64
  - Use multiples of 12 to match socket size (per Assignment-1 feedback)
- [ ] Plot: **Speedup = T(1)/T(N)** vs N processors
- [ ] Plot: **Efficiency = Speedup/N** vs N processors
- [ ] Formula: `Speedup = 1 / (s + p/N)` where s=serial fraction, p=parallel fraction

#### Weak Scaling (Gustafson's Law)
- [ ] Constant work per rank (scale problem with ranks)
- [ ] More datapoints needed (GH #34)
- [ ] Plot: **Efficiency = T(1)/T(N)** vs N processors (should stay ~1.0)
- [ ] Formula: `Scaled Speedup = s + p×N`

**Important:** "wall-time vs. number of processors is NOT a scaling plot" (Assignment-1 Feedback)

**Generic Feedback:**
- "Scaling plots, x is number of processors, y is T(NP)/T(1), log-log scale!"
- "Go large-scale, up to at least 4 nodes, or as high as you can go"
- "Always consider how chunk-size influences work-load"

### 2.2 Communication Experiments (GH #30)

- [ ] **Larger problem sizes** for communication experiment
- [ ] **Communicator comparison:** numpy vs custom MPI datatypes
- [ ] **Socket binding strategy** (todos.md #3)
  - **Week 02 (p.44):** Use `mpirun --map-by ppr:1:package` for max bandwidth
  - Compare compact vs spread placement
  - Use rank counts matching socket: 12, 24, 36...

### 2.3 Rank Placement Logging (todos.md #3, #15)
- [ ] **Log per-rank info in solver:**
  - Hostname
  - Socket ID
  - CPU model
  - Neighbor mapping for halo communication
  - Consider networkx visualization for rank topology

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

### 4.1 Required Content

- [ ] **Hardware description** (Assignment-1 Feedback: "no description of hardware/compiler used")
  - CPU type/model
  - Memory per node
  - Network (Infiniband specs)
  - Compiler/MPI version

- [ ] **Bibliography** (todos.md #7)
  - Multigrid: Briggs, Henson & McCormick - "A Multigrid Tutorial" (2nd ed., SIAM 2000)
  - Alternative: Trottenberg, Oosterlee, Schüller - "Multigrid" (Academic Press 2001)

- [ ] **Parallel IO considerations** (GH #29)
  - Explain result gathering excluded from timings
  - Reference Week 09 for parallel IO approaches

- [ ] **Design decisions** (todos.md #19, #21)
  - Never gather global solution - use parallel validation
  - Custom MPI types vs numpy halo exchange trade-offs

### 4.2 Analysis Points

- [ ] **Node crossing analysis** (Assignment-1 Feedback)
  - "how did it perform when crossing from 2 nodes to 3 nodes? Comments there would have fitted"

- [ ] **Socket matching** (Assignment-1 Feedback)
  - "if you would have used a rank increment of 12, you would have matched the socket size"

- [ ] **Sockets vs nodes discussion** (Generic Feedback)
  - "A more thorough discussion on sockets vs. nodes"
  - "touch on the sockets when you hit multiples of 12"

- [ ] **Barrier before timing** (Generic Feedback)
  - "Many have the correct usage of a Barrier before the initial timing. But there are no discussions on why"
  - Explain: ensures all ranks start timing simultaneously

- [ ] **Negative results** (Generic Feedback)
  - "Negative results are also results, it's typically fair to highlight these results so that others won't do the same mistake!"

- [ ] **Data placement optimization** (Generic Feedback)
  - "All of you did a receive on rank == 0 in a buffer, why? What about receiving data right where you need it?"

---

## Phase 5: Code Quality (GH #28)

Lower priority - do if time permits.

- [ ] Ensure docstrings up to date
- [ ] Update `API_reference.rst`
- [ ] Remove redundant `palettes.py`
- [ ] Consistent figure sizes
- [ ] Better DataFrame column naming
- [ ] stdout/stderr always logged to MLflow
- [ ] Check `allgather` (lowercase) efficiency - should use `Allgather`
- [ ] Clean up git repo (todos.md #25)
- [ ] MLflow run tagging: local vs HPC (todos.md #27)

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

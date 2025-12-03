# Action Plan - Project 2 Final Sprint

**Deadline:** December 3, 2025 (midnight)

---

## PHASE 1: Code Fixes (BLOCKS HPC)
*All code changes must be done before submitting HPC jobs*

### 1.1 Timing & Metrics Verification
- [ ] Verify `bytes_per_point = 64` in `base.py` (already done ✓)
- [ ] Verify timing uses `MPI.Wtime()`
- [ ] Verify timing excludes initialization, measures only solve loop
- [ ] Add timing breakdown verification test:
  ```python
  # After solve, check: compute_time + halo_time ≈ wall_time
  gap = wall_time - (compute_total + halo_total)
  print(f"Timing gap (other): {gap/wall_time*100:.1f}%")
  ```

### 1.2 Numba Configuration
- [ ] Verify `NUMBA_NUM_THREADS` set explicitly before numba imports
- [ ] Verify warmup is called before timing starts in all runners
- [ ] Add warmup call if missing:
  ```python
  solver.warmup()  # Trigger JIT compilation
  comm.Barrier()   # Sync all ranks
  t_start = MPI.Wtime()  # Now start timing
  ```

### 1.3 MPI Communication Verification
- [ ] Verify solve loop uses uppercase MPI methods:
  - `halo.py`: `Sendrecv` ✓
  - `jacobi_mpi.py`: `Allreduce` ✓
  - `fmg_mpi.py`: `Allreduce` ✓
- [ ] Verify `comm.gather()` (lowercase) only used for post-solve Python dicts

### 1.4 Rank Topology Logging
- [ ] Add hostname to rank info: `MPI.Get_processor_name()`
- [ ] Add cart_coords: `grid.cart_comm.Get_coords(rank)`
- [ ] Gather rank info and log as MLflow artifact:
  ```python
  rank_info = {
      'rank': self.rank,
      'hostname': MPI.Get_processor_name(),
      'cart_coords': list(self.grid.cart_comm.Get_coords(self.rank)),
      'local_shape': self.grid.local_shape,
      'neighbors': self.grid.neighbors,
  }
  all_ranks = comm.gather(rank_info, root=0)
  if rank == 0:
      mlflow.log_table(all_ranks, "rank_topology.json")
  ```

### 1.5 Timing Summary Stats
- [ ] Implement per-rank timing summary with pandas:
  ```python
  df = pd.DataFrame({
      "compute": solver.local_metrics.compute_times,
      "halo": solver.local_metrics.halo_times,
  })
  stats = df.agg(["sum", "mean", "std", "min", "max"]).T
  stats_flat = {f"{col}_{stat}": val
                for col, row in stats.iterrows()
                for stat, val in row.items()}
  ```

### 1.6 Code Quality (Quick fixes only)
- [ ] Run `uv run ruff format src/`
- [ ] Run `uv run ruff check src/ --fix`
- [ ] Fix any remaining lint errors

### 1.7 Local Verification
- [ ] Test sequential: `uv run python run_solver.py N=100 n_ranks=1`
- [ ] Test MPI: `mpiexec -n 4 uv run python run_solver.py N=100`
- [ ] Test FMG: `mpiexec -n 8 uv run python run_solver.py solver=fmg N=65`
- [ ] Verify MLUPS output is reasonable
- [ ] Verify MLflow logging works

---

## PHASE 2: Submit HPC Jobs
*Submit all jobs, then work on report while they run*

### 2.1 Strong Scaling Jobs
**Config:** Fixed N, sweep ranks (multiples of 12 for socket boundaries)

```bash
# strong_scaling.sh
N_VALUES="250 500"
RANKS="1 2 4 8 12 16 24 27 32 36 48 64"
DECOMP="cubic"
COMM="custom"
NUMBA_THREADS=1  # Pure MPI scaling
```

- [ ] Create job script: `jobs/strong_scaling_N250.sh`
- [ ] Create job script: `jobs/strong_scaling_N500.sh`
- [ ] Submit jobs

### 2.2 Weak Scaling Jobs
**Config:** Constant work per rank, scale problem with ranks

| Ranks | Local size per rank | Global N |
|-------|---------------------|----------|
| 1 | ~100³ | 100 |
| 8 | ~100³ | 200 |
| 27 | ~100³ | 300 |
| 64 | ~100³ | 400 |

- [ ] Create job script: `jobs/weak_scaling.sh`
- [ ] Submit job

### 2.3 Rank Placement Jobs
**Config:** Compare placement strategies at fixed rank count

| Strategy | Command | Expected |
|----------|---------|----------|
| Compact (1 node) | `span[hosts=1]` | Memory BW limited |
| Spread (multi-socket) | `--map-by ppr:1:package` | Max bandwidth |

- [ ] Create job script: `jobs/placement_comparison.sh`
- [ ] Submit job

### 2.4 Communication Comparison Jobs
**Config:** numpy buffers vs custom MPI datatypes

- [ ] Create job script: `jobs/comm_comparison.sh`
- [ ] Submit job

### 2.5 Decomposition Comparison Jobs
**Config:** sliced vs cubic at various rank counts

- [ ] Create job script: `jobs/decomp_comparison.sh`
- [ ] Submit job

### 2.6 FMG vs Jacobi Jobs (Optional)
- [ ] Create job script: `jobs/fmg_comparison.sh`
- [ ] Submit job

---

## PHASE 3: Report Writing (While HPC runs)
*Work on sections that don't need experimental data*

### 3.1 Hardware Description Section (REQUIRED - from feedback)
Add to methodology:

```latex
\subsection{Hardware Configuration}
Experiments were conducted on the DTU HPC cluster:
\begin{itemize}
  \item \textbf{CPU:} Intel Xeon E5-2650 v4 (12 cores/socket, 2 sockets/node, 2.2 GHz)
  \item \textbf{Memory:} 256 GB per node (128 GB per socket)
  \item \textbf{Memory bandwidth:} 76.8 GB/s per socket (DDR4-2400, 4 channels)
  \item \textbf{Network:} FDR Infiniband (56 Gb/s)
  \item \textbf{Software:} Python 3.12, mpi4py 3.x, Numba 0.x, OpenMPI 4.x
\end{itemize}
```

- [ ] Write hardware section
- [ ] Get exact software versions from HPC (`module list`)

### 3.2 Methodology Additions

- [ ] **Barrier explanation**: Add why we use `MPI.Barrier()` before timing
  > "A barrier synchronization ensures all ranks begin timing simultaneously, preventing faster ranks from including idle wait time in their measurements."

- [ ] **Numba thread documentation**: State explicitly
  > "All experiments use `NUMBA_NUM_THREADS=8` unless otherwise noted. For pure MPI scaling experiments, we set `NUMBA_NUM_THREADS=1` to isolate MPI parallelization effects."

- [ ] **Timing methodology**: Document what is/isn't timed
  > "Wall time includes only the iterative solve phase, excluding initialization, warmup, and result gathering."

### 3.3 Theory/Background Additions

- [ ] **Socket/Node hierarchy explanation**:
  - Intra-socket: shared L3 cache, fastest
  - Inter-socket (same node): QPI/UPI link, ~40 GB/s
  - Inter-node: Infiniband, 56 Gb/s but higher latency

- [ ] **Scaling theory recap**:
  - Strong scaling: Amdahl's law, `S = 1/(s + p/P)`
  - Weak scaling: Gustafson's law, `S = s + p×P`

### 3.4 Discussion Points to Prepare

- [ ] **Socket boundary effects** (12, 24, 36, 48 ranks)
  > "At 12 ranks, all processes fit within a single socket sharing L3 cache. Beyond 12 ranks, inter-socket communication introduces additional latency."

- [ ] **Node crossing analysis** (feedback requirement)
  > "When scaling from 24 ranks (1 node) to 36 ranks (2 nodes), we observe [X] due to inter-node Infiniband communication replacing intra-node shared memory."

- [ ] **Negative results** (if any)
  > "Contrary to expectations, cubic decomposition did not outperform sliced at [X] ranks because [reason]."

### 3.5 Conclusion Draft
- [ ] Draft conclusion structure (fill in numbers after experiments)

---

## PHASE 4: Generate Plots (After HPC completes)

### 4.1 Download Results
- [ ] Download MLflow artifacts from HPC runs
- [ ] Verify all experiments completed successfully
- [ ] Load data into analysis notebook

### 4.2 Strong Scaling Plots

**CRITICAL FORMAT (from feedback):**
```
❌ WRONG: X=processors, Y=wall_time
✓ CORRECT: X=processors, Y=T(1)/T(P), log-log scale
```

```python
# Strong scaling plot
t1 = df[df['n_ranks'] == 1]['wall_time'].values[0]
df['speedup'] = t1 / df['wall_time']
df['efficiency'] = df['speedup'] / df['n_ranks']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Speedup plot (log-log)
ax1.loglog(df['n_ranks'], df['speedup'], 'o-', label='Measured')
ax1.loglog(df['n_ranks'], df['n_ranks'], '--', label='Ideal')
ax1.set_xlabel('Number of Processors')
ax1.set_ylabel('Speedup T(1)/T(P)')
ax1.legend()

# Efficiency plot
ax2.semilogx(df['n_ranks'], df['efficiency'], 'o-')
ax2.axhline(y=1.0, linestyle='--', color='gray', label='Ideal')
ax2.set_xlabel('Number of Processors')
ax2.set_ylabel('Parallel Efficiency')
ax2.set_ylim(0, 1.1)
```

- [ ] Create strong scaling speedup plot (log-log)
- [ ] Create strong scaling efficiency plot
- [ ] Add ideal scaling reference line
- [ ] Mark socket/node boundaries (12, 24, 36, 48)

### 4.3 Weak Scaling Plot

```python
# Weak scaling - efficiency should stay ~1.0
t1 = df[df['n_ranks'] == 1]['wall_time'].values[0]
df['efficiency'] = t1 / df['wall_time']

plt.plot(df['n_ranks'], df['efficiency'], 'o-')
plt.axhline(y=1.0, linestyle='--', label='Ideal')
plt.xlabel('Number of Processors')
plt.ylabel('Weak Scaling Efficiency T(1)/T(P)')
```

- [ ] Create weak scaling efficiency plot

### 4.4 Performance Plots

- [ ] MLUPS vs problem size
- [ ] Bandwidth vs problem size (with 76.8 GB/s reference line)
- [ ] Compute vs halo time breakdown
- [ ] Surface-to-volume ratio (already done ✓)

### 4.5 Comparison Plots

- [ ] Decomposition comparison (sliced vs cubic)
- [ ] Communication comparison (numpy vs custom datatypes)
- [ ] Rank placement comparison (compact vs spread)

### 4.6 Plot Quality Checklist
- [ ] Font sizes ≥ 10pt (readable when printed)
- [ ] Axis labels on all plots
- [ ] Proper tick marks (not too few, not too many)
- [ ] Log-log scale for scaling plots
- [ ] Legends where needed
- [ ] Consistent color scheme
- [ ] Unit-less efficiency values (fractions, not raw times)

---

## PHASE 5: Complete Report (After plots ready)

### 5.1 Scaling Analysis Section
Fill in `03_results.tex` scaling subsection with actual numbers:

- [ ] Present strong scaling results with figure
- [ ] Present weak scaling results with figure
- [ ] Analyze socket boundary effects (12→24→36 ranks)
- [ ] Analyze node crossing effects
- [ ] Compare measured vs theoretical (Amdahl/Gustafson)
- [ ] Compute and report parallel efficiency

### 5.2 Rank Placement Section
- [ ] Present placement comparison results
- [ ] Explain compact vs spread tradeoffs
- [ ] Recommend optimal configuration

### 5.3 FMG Section (if experiments run)
- [ ] Present FMG vs Jacobi comparison
- [ ] Discuss communication reduction benefits

### 5.4 Complete Discussion
- [ ] Socket/node effects analysis
- [ ] Memory bandwidth saturation analysis
- [ ] Negative results (if any) with explanations

### 5.5 Complete Conclusion
- [ ] Key findings summary
- [ ] Achieved performance (X Mlup/s, Y% efficiency at Z ranks)
- [ ] Recommendations for optimal configuration
- [ ] Future work (non-blocking halo, better load balancing, etc.)

---

## PHASE 6: Final Polish & Submit

### 6.1 Report Checklist
- [ ] All figures have captions
- [ ] All figures referenced in text
- [ ] All tables have captions
- [ ] Bibliography complete (Multigrid tutorial, etc.)
- [ ] No TODO/FIXME comments in LaTeX
- [ ] Page limit respected (if any)
- [ ] Compile without errors/warnings

### 6.2 Code Package
- [ ] Create ZIP of source code
- [ ] Include README with:
  - Dependencies (`uv sync`)
  - How to run locally
  - How to run on HPC
  - Example commands
- [ ] Remove unnecessary files:
  - `.pyc`, `__pycache__`
  - `.venv/`
  - `mlruns/` (local MLflow data)
  - Large data files

### 6.3 Final Checks
- [ ] PDF renders correctly
- [ ] All plots readable when printed
- [ ] Code ZIP opens correctly
- [ ] Both files ready for upload

### 6.4 Submit
- [ ] Upload PDF to DTU Learn
- [ ] Upload code ZIP to DTU Learn
- [ ] Verify submission before midnight

---

## Quick Reference

### Scaling Plot Format (CRITICAL)
```
❌ WRONG:  X = processors, Y = wall_time
✓ CORRECT: X = processors, Y = T(1)/T(P), log-log scale
```

### Rank Count → Hardware Mapping
| Ranks | Sockets | Nodes | Key transition |
|-------|---------|-------|----------------|
| 12 | 1 | 1 | Single socket |
| 24 | 2 | 1 | Full node (inter-socket) |
| 36 | 3 | 2 | **Node crossing!** |
| 48 | 4 | 2 | Two full nodes |
| 64 | 6 | 3 | Large scale |

### Key Formulas
```
MLUPS = (N-2)³ × iterations / (wall_time × 10⁶)
Bandwidth (GB/s) = MLUPS × 64 / 1000
Strong Speedup = T(1) / T(P)
Efficiency = Speedup / P
```

### Hardware Specs
- Socket: 12 cores, 76.8 GB/s memory bandwidth
- Node: 2 sockets, 24 cores, 153.6 GB/s total
- Network: 56 Gb/s Infiniband

---

## Time Estimates

| Phase | Time | Can parallelize? |
|-------|------|------------------|
| 1. Code fixes | 1-2 hr | No - blocks everything |
| 2. Submit HPC | 30 min | Then wait for jobs |
| 3. Report writing | 2-3 hr | While HPC runs |
| 4. Generate plots | 1-2 hr | After HPC completes |
| 5. Complete report | 1-2 hr | After plots ready |
| 6. Final polish | 30 min | Last step |
| **Total** | **6-10 hr** | |

**Critical path:** Phase 1 → Phase 2 → (Phase 3 parallel) → Phase 4 → Phase 5 → Phase 6

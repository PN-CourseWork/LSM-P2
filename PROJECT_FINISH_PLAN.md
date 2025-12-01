# Project Finish-Up Plan: MPI Poisson Solver

**Course**: 02616 Large-scale Modelling
**Deadline**: ~2 days
**Branch**: `final-handin`
**Last Updated**: 2025-12-01

---

## Executive Summary

The project is **90% complete** with excellent code quality and architecture. The **critical blocker** is the **missing scaling analysis** (Experiment 05), which is the main deliverable for Project 2. All infrastructure exists - only the experiments need to be run.

### Quick Status
| Component | Status |
|-----------|--------|
| Kernel benchmarks | COMPLETE |
| Domain decomposition | COMPLETE |
| Communication strategy | COMPLETE |
| Solver validation | COMPLETE |
| **Scaling analysis** | **CRITICAL - MISSING** |
| Multigrid/FMG | COMPLETE |
| Report sections 1-4 | 80% (scaling empty) |
| Conclusion | EMPTY |

---

## Open GitHub Issues

| # | Title | Priority | Action Needed |
|---|-------|----------|---------------|
| **14** | **Scaling Analysis** | CRITICAL | Run strong/weak scaling experiments |
| **19** | Geometric multigrid? | LOW | "99% sure not worth the time" - skip |
| **20** | MLops/s or wall time | HIGH | Add Mlup/s metrics to report |
| **18** | Splitting ranks/sockets | MEDIUM | Document node configuration |
| **16** | MLFlow setup revised | LOW | fetch_data_mlflow.py exists |
| **25** | Tools | DONE | HPC job monitor, job packs merged |

---

## Agent Analysis Reports

### 1. Course Requirements Assessment

#### Requirements Checklist

| Requirement | Status | Notes |
|-------------|--------|-------|
| Sequential Jacobi implementation | MET | Section 2.1 |
| Parallel MPI implementation | MET | Full MPI solver |
| Multiple domain decomposition | MET | Sliced + Cubic |
| Validation against analytical solution | MET | Section 3.4, O(h^2) convergence |
| **Scaling plots** | **NOT MET** | Section 3.5 is placeholder |
| **Performance metrics (Mlup/s)** | **NOT MET** | Uses wall-time only |
| **Memory bandwidth analysis** | **NOT MET** | Not discussed |
| **Hardware specifications** | **NOT MET** | No CPU/memory specs |
| **Cartesian topology (Create_cart)** | **NOT MET** | Manual decomposition used |

#### Assignment 1 Feedback Status

| Feedback Item | Addressed? |
|---------------|------------|
| Plot readability | YES - Good fonts/labels |
| Code complexity | YES - Modular design |
| Uppercase MPI calls (Send/Recv) | NEEDS CHECK |
| Scaling plots (not just wall-time) | NO - Missing |
| Hardware documentation | NO - Missing |

#### Critical Gaps (Will Hurt Grade)

1. **Section 3.5 (Scaling)** - Completely empty
2. **No Mlup/s metrics** - Required per assignment
3. **No hardware specs** - Required per feedback
4. **Empty conclusion** - Section 4 needs writing
5. **Multigrid section 2.4** - Mentions FMG but no results in report

---

### 2. Code Quality Review

**Overall Grade: B+ (Good, with fixes needed)**

#### Critical Issues

| Issue | Location | Impact | Fix |
|-------|----------|--------|-----|
| Missing timing aggregation | solver.py:~111 | All timing data shows 0/None | Add sum of timeseries |
| Unnecessary Barrier | communicators.py:51 | Inflates comm costs | Remove barrier |
| Sliced residual bug | solver.py:144 | Wrong convergence check for sliced | Use axis-aware slicing |

#### Good Practices Found
- Excellent modular design with strategy pattern
- Proper MPI patterns (allreduce, gather, Sendrecv)
- Good Numba integration with warmup
- Correct boundary condition handling
- Clean documentation with type hints

#### Code Fixes Required Before Scaling Experiments

```python
# solver.py - Add after _iterate() in solve():
if self.rank == 0:
    self.results.total_compute_time = sum(self.timeseries.compute_times)
    self.results.total_halo_time = sum(self.timeseries.halo_exchange_times)
    self.results.total_mpi_comm_time = sum(self.timeseries.mpi_comm_times)

# communicators.py:51 - Remove unnecessary barrier
# DELETE: comm.Barrier()
```

---

### 3. Project Architecture Assessment

**Completeness: 90%**

#### Module Status
| Module | Status | Notes |
|--------|--------|-------|
| src/Poisson/solver.py | Complete | Jacobi solver |
| src/Poisson/multigrid.py | Complete | V-cycle + FMG |
| src/Poisson/mpi/decomposition.py | Complete | Sliced + Cubic |
| src/Poisson/mpi/communicators.py | Complete | NumPy + Custom datatypes |
| src/utils/* | Complete | All utilities working |

#### Data Flow
```
Experiments/XX-*/ (compute_*.py)
    -> data/XX-*/ (parquet, HDF5)
    -> Experiments/XX-*/ (plot_*.py)
    -> figures/XX-*/ (PDF, PNG)
    -> docs/reports/TexReport/figures/
```

#### No Dead Code Detected
- All imports used
- All functions called
- Clean codebase

---

### 4. Experiments & Data Completeness

| Experiment | Data | Plots | Report |
|------------|------|-------|--------|
| 01-kernels | 2 parquet | 3 PDF | READY |
| 02-decomposition | N/A | 3 PNG | READY |
| 03-communication | 1 parquet (np4 only) | 2 PDF | Partial |
| 04-validation | 12 HDF5 | 2 files | READY |
| **05-scaling** | **1 HDF5 (test only)** | **NONE** | **MISSING** |
| 06-multigrid | 24 HDF5 | 1 PDF | READY |

#### Missing Figures for Report
- `figures/scaling/strong_scaling.pdf` - Does not exist
- `figures/scaling/weak_scaling.pdf` - Does not exist

#### All Other Figures Present
- 10 figures referenced in report Section 3
- All exist in figures/ directory

---

## Prioritized Action Plan

### Priority 1: CRITICAL (Blocking Submission)

#### 1.1 Fix Code Issues Before Running Experiments
```bash
# Files to edit:
# 1. src/Poisson/solver.py - Add timing aggregation
# 2. src/Poisson/mpi/communicators.py - Remove unnecessary barrier
```

#### 1.2 Run Scaling Experiments (Issue #14)
```bash
# Option A: Local testing (small scale)
cd /Users/philipnickel/Documents/GitHub/DTU_Courses/LargeScaleModeling/LSM/LSM-Project_2
mpiexec -n 4 uv run python Experiments/05-scaling/runner.py --N 64 --max-iter 100

# Option B: HPC job submission
uv run python main.py --hpc  # Interactive job generator
```

**Required Configurations:**
- Strong scaling: N=128 or 256, P={1,2,4,8,16,32,64}
- Weak scaling: Local N=32^3, P={1,2,4,8,16,32,64}
- All 4 strategy combos: sliced/cubic x numpy/custom

#### 1.3 Create Scaling Plots
```bash
# Need to create: Experiments/05-scaling/plot_scaling.py
# Should generate:
# - figures/scaling/strong_scaling_efficiency.pdf
# - figures/scaling/weak_scaling_efficiency.pdf
```

#### 1.4 Fill Report Section 3.5
- Strong scaling results with speedup S(P) = T(1)/T(P)
- Weak scaling efficiency E(P) = T(1)/T(P)
- Compare sliced vs cubic decomposition
- Discuss limiting factors

---

### Priority 2: HIGH (Improve Grade)

#### 2.1 Add Performance Metrics (Issue #20)
- Convert timing to Mlup/s: `Mlup/s = (N^3 * iterations) / (time * 10^6)`
- Add to results section
- Create performance table

#### 2.2 Document Hardware Specifications
Add to methodology:
```latex
\subsection{Computational Environment}
Experiments conducted on [HPC Cluster]:
- CPU: [model]
- Cores per node: [X]
- Memory bandwidth: [Y] GB/s
- Network: [type]
```

#### 2.3 Write Conclusion (Section 4)
- Summary of key findings
- Comparison of decomposition strategies
- Optimal configuration recommendation
- Limitations and future work

#### 2.4 Add Multigrid Results to Report
- FMG data exists in data/multigrid_fmg/
- Plot exists: figures/multigrid/fmg_convergence.pdf
- Add section discussing FMG vs Jacobi algorithmic efficiency

---

### Priority 3: MEDIUM (Polish)

#### 3.1 Complete Communication Experiment
- Run with more rank counts (8, 16, 32)
- Currently only np4 data exists

#### 3.2 Verify MPI Uppercase Calls
```bash
grep -r "comm\.send\|comm\.recv" src/Poisson/
# Should use Send/Recv not send/recv
```

#### 3.3 Consider Cartesian Topology
- Currently uses manual neighbor calculation
- Could use MPI.Compute_dims() + Create_cart()
- Low priority - current implementation works

---

### Priority 4: LOW (Nice to Have)

- Add memory bandwidth analysis
- Create presentation slides content
- Expand unit tests
- Add version string to package

---

## Timeline

### Day 1 (Today - Dec 1)
| Time | Task | Owner |
|------|------|-------|
| AM | Fix code issues (timing, barrier) | Dev |
| AM | Start scaling experiments (HPC submit) | Dev |
| PM | Create plot_scaling.py script | Dev |
| PM | Draft Section 3.5 structure | Dev |

### Day 2 (Tomorrow - Dec 2)
| Time | Task | Owner |
|------|------|-------|
| AM | Collect scaling results | HPC |
| AM | Generate scaling plots | Dev |
| PM | Write Section 3.5 + Conclusion | Dev |
| PM | Add hardware specs, Mlup/s | Dev |
| EVE | Final review and polish | All |

### Day 3 (Deadline - Dec 3)
| Time | Task | Owner |
|------|------|-------|
| AM | Final checks | All |
| AM | Commit and submit | You (not Claude) |

---

## Commands Reference

```bash
# Run all compute scripts
uv run python main.py --compute

# Run all plot scripts
uv run python main.py --plot

# Copy plots to report
uv run python main.py --copy-plots

# Build documentation
uv run python main.py --docs

# Clean generated files
uv run python main.py --clean

# Fetch MLflow artifacts
uv run python main.py --fetch

# HPC job generator
uv run python main.py --hpc
```

---

## File Checklist

### Must Edit
- [x] `src/Poisson/solver.py` - Add timing aggregation DONE
- [x] `src/Poisson/mpi/communicators.py` - Remove barrier (line 51) DONE
- [x] `docs/reports/TexReport/Report/sections/03_results.tex` - Section 3.5 structure DONE
- [x] `docs/reports/TexReport/Report/sections/04_conclusion.tex` - Write conclusion DONE
- [ ] `docs/reports/TexReport/Report/sections/02_methodology.tex` - Add hardware specs

### Must Create
- [x] `Experiments/05-scaling/plot_scaling.py` - Scaling visualization DONE
- [x] `figures/scaling/` - Directory for scaling plots DONE

### Must Generate (via experiments)
- [ ] `data/05-scaling/*.h5` - Scaling experiment results
- [ ] `figures/scaling/strong_scaling.pdf`
- [ ] `figures/scaling/weak_scaling.pdf`

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| HPC queue delays | HIGH | Start jobs ASAP, have local fallback |
| Scaling shows poor results | MEDIUM | Analyze and explain in discussion |
| Time runs out | HIGH | Focus on minimum viable: 1 strong + 1 weak plot |
| Code bugs affect results | HIGH | Fix timing issue before experiments |

---

## Success Criteria

### Minimum Viable Submission
- [ ] At least 1 strong scaling plot
- [ ] At least 1 weak scaling plot
- [ ] Section 3.5 with basic analysis
- [ ] Non-empty conclusion

### Good Submission
- [ ] Strong + weak scaling for all 4 configurations
- [ ] Mlup/s metrics
- [ ] Hardware documentation
- [ ] Comprehensive discussion

### Excellent Submission
- [ ] All above plus memory bandwidth analysis
- [ ] Jacobi vs FMG comparison
- [ ] Multi-node scaling analysis
- [ ] Timing breakdown (compute vs communication)

---

*This plan was generated by analyzing the project with 4 specialized agents examining course requirements, code quality, architecture, and data completeness.*

# Code Review - COMPLETED

All phases completed successfully. 68 tests passing.

## Summary of Changes

### Phase 0: Pytest Suite
- `tests/test_kernels.py` - Kernel correctness & O(h²) convergence
- `tests/test_decomposition.py` - Domain decomposition logic
- `tests/test_communicators.py` - Halo exchange (NumpyHaloExchange, CustomHaloExchange)
- `tests/test_solver.py` - JacobiPoisson solver convergence & accuracy (single-rank)
- `tests/test_problems.py` - Grid creation & sinusoidal problem setup
- `tests/test_mpi_integration.py` - End-to-end MPI tests via subprocess runner (1-8 ranks)

### Phase 1: Bug Fixes & Enhancements
- Fixed warmup parameter mismatch in `kernels.py`
- Scaling experiment converted to placeholder
- Added configurable `axis` parameter to sliced decomposition

### Phase 2: Communicators Refactor
- Renamed: `NumpyCommunicator` → `NumpyHaloExchange`
- Renamed: `DatatypeCommunicator` → `CustomHaloExchange`
- Extracted `_get_neighbor_ranks()` helper (~60 lines reduced)
- Created `_BaseHaloExchange` base class

### Phase 3: Experiments Refactor
- `plot_decompositions.py`: 134 → 84 lines (extracted `visualize_decomposition`)
- `compute_all.py`: 175 → 106 lines (extracted `run_kernel`, `kernel_to_df`)

### Phase 4: MPI Correctness
- Fixed hardcoded byte stride → `MPI.DOUBLE.Get_size()`
- Added offset validation assertions in `CustomHaloExchange`

### Phase 5: Documentation & Terminology
- Standardized on "halo" terminology (replaced "ghost")
- Renamed method: `exchange_ghosts` → `exchange_halos`
- Renamed field: `ghost_cells_total` → `halo_cells_total`

# Plan: Multigrid Acceleration for 3D Poisson Solver

## Objective
Implement a Full Multigrid (FMG) or V-Cycle algorithm to accelerate the convergence of the existing 3D Poisson solver. The goal is to achieve mesh-independent convergence rates ($O(N)$ instead of $O(N^3)$ or $O(N^5)$ depending on method).

## 1. Architecture & Design

We will introduce a new solver class `MultigridPoisson` that manages a hierarchy of grids. This class will use the existing `JacobiPoisson` (or a lighter variant) for smoothing on each level.

### 1.1 New File: `src/Poisson/multigrid_operators.py` (Completed)
*   Contains standalone Numba-optimized functions for grid transfer operations.
*   `restrict(fine_u, coarse_u)`: Full Weighting.
*   `prolong(coarse_u, fine_u)`: Trilinear Interpolation.

### 1.2 New File: `src/Poisson/multigrid.py`
*   **Class `MultigridPoisson`**:
    *   **Initialization**: Takes `N` (fine grid size), `levels` (depth of V-cycle), and other config.
    *   **Grid Hierarchy**: Creates a list of `GridLevel` objects. Each `GridLevel` contains:
        *   `u` (solution)
        *   `f` (RHS)
        *   `h` (grid spacing)
        *   `communicator` (for halo exchange)
        *   `decomposition` (for domain info) - *Note: MPI decomposition handling for coarse grids is complex; strictly checking if local size allows coarsening.*
    *   **Method `solve()`**: Orchestrates the V-cycles until convergence.
    *   **Method `v_cycle(level)`**: Recursive function for the V-cycle.

## 2. Implementation Steps

### Step 1: Add Multigrid Kernels (Completed)
*   Implemented `restrict` and `prolong` in `src/Poisson/multigrid_operators.py`.
*   Verified with `tests/test_multigrid_operators.py`.

### Step 2: Create Multigrid Solver Structure (Next)
*   Create `src/Poisson/multigrid.py`.
*   Define `MultigridPoisson`.
*   Implement the initialization logic to set up the grid hierarchy.
    *   *Constraint*: Coarsest grid local size must be $\ge 2$ (interior) to allow stencil operations.

### Step 3: Implement V-Cycle Logic
*   Implement the recursive V-cycle:
    1.  **Pre-smoothing**: Run $v_1$ Jacobi steps on current level.
    2.  **Residual Computation**: $r = f - Au$.
    3.  **Restriction**: $f_{coarse} = R(r)$.
    4.  **Recursion**: Solve for error $e$ on coarse grid: $Ae = f_{coarse}$.
    5.  **Prolongation**: $e_{fine} = P(e)$.
    6.  **Correction**: $u = u + e_{fine}$.
    7.  **Post-smoothing**: Run $v_2$ Jacobi steps.
*   *Base Case*: On the coarsest level, solve exactly (or run many smoothing steps).

### Step 4: Integration & CLI
*   Update `main.py` or add a new CLI entry point to run multigrid experiments.
*   Add configuration options for `levels`, `pre_smooth`, `post_smooth`.

## 3. MPI Considerations
*   **Halos**: Restriction and Prolongation require valid ghost cells. We must perform halo exchanges *before* these operations if the stencil reaches across boundaries.
*   **Global Reductions**: Residual computation still requires global reduction, but only on the current active level.

## 4. Verification & Validation
*   **Convergence Test**: Verify that the number of V-cycles required to reach a fixed tolerance is roughly independent of $N$.
*   **Benchmark**: Compare time-to-solution of `JacobiPoisson` vs. `MultigridPoisson`.
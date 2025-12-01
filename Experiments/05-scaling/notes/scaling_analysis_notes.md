Nice, this is the money question: what should we actually expect the plots to look like when we throw Jacobi and FMG at these experiments?

I’ll separate it into two things:
	•	Parallel efficiency expectations (time per work unit, your actual measurements)
	•	Algorithmic efficiency expectations (error reduction per work unit, mostly conceptual for the report)

⸻

1. Parallel efficiency: time per work unit

You defined:
	•	Jacobi work unit = 1 Jacobi iteration (one sweep + halo exchange)
	•	FMG work unit = 1 full multigrid cycle (V/FMG cycle with smoothing, restriction, interpolation, coarse solve, prolongation, etc.)

Cost per work unit (single rank)

On a single rank, ignoring MPI:
	•	A Jacobi iteration is just:
stencil update over all N³ points → cheap, very regular.
	•	An FMG cycle:
	•	several smooths on fine grid
	•	restrict down through levels
	•	smooth/solve on coarse grid
	•	prolong back up, smoothing on each level

Total operations per FMG cycle is still O(N³) with a small constant (like 2–5 Jacobi-sweep equivalents depending on how you set it up), but:
	•	Time per FMG cycle > time per Jacobi iteration by some constant factor C > 1.

So on 1 rank, in raw timings you should expect:
	•	T_Jacobi_per_iter = baseline
	•	T_FMG_per_cycle ≈ C * T_Jacobi_per_iter  (C maybe 3–10-ish depending on your implementation choices)

That’s fine; FMG is “heavier” per work unit.

Strong scaling: time per work unit vs P

For strong scaling (fixed global N³, P going up):
	•	Both Jacobi and FMG work units perform the same kind of halo exchanges on the finest grid, just different numbers of sweeps.
	•	FMG additionally does halo exchanges on coarser grids, but those grids:
	•	Have many fewer points.
	•	So total comm volume at coarse levels is tiny.

So for strong scaling:
	•	Shape of Jacobi and FMG curves vs P should be similar:
	•	Initially near-ideal T ∝ 1/P.
	•	Then flattening as halo comm + latency starts to dominate.
	•	Cubic > sliced for both solvers:
	•	cubic: better surface/volume, less halo per rank → better scaling
	•	sliced: big interfaces → more comm → earlier saturation
	•	Custom MPI > NumPy for both solvers:
	•	especially for sliced decomposition where memory isn’t contiguous

What differs mainly is absolute level:
	•	Jacobi curve lower: smaller time per work unit.
	•	FMG curve higher: same scaling trend, but scaled by the constant factor C.

So a “healthy” plot of time per work unit vs P:
	•	Two families of curves (Jacobi, FMG),
	•	FMG curves sit above Jacobi curves,
	•	Both families show:
	•	cubic+custom best
	•	sliced+NumPy worst
	•	parallel efficiency dropping as P increases.

Weak scaling: time per work unit vs P

For weak scaling (local size fixed, P and global N grow):
	•	Ideal world: time per work unit is flat as P increases.
	•	Reality:
	•	Jacobi: mostly flat for moderate P, then slowly increases when latency and global reductions (if any) start to bite.
	•	FMG: similar, but with an extra twist:
	•	At very large P, coarse levels become tiny per rank.
	•	Coarse-grid work is dominated by communication/latency, not flops.
	•	This can make FMG’s time-per-cycle curve bend upwards earlier than Jacobi’s.

So:
	•	Expect Jacobi weak scaling curve to stay slightly flatter for large P.
	•	FMG weak scaling curve may show worse efficiency at the highest P because of coarse-level work being comm-bound.

But still:
	•	For both solvers under weak scaling:
	•	cubic+custom will be the flattest curves,
	•	sliced+NumPy will degrade fastest.

⸻

2. Algorithmic efficiency: error reduction per work unit

You don’t want to run Jacobi to convergence (good choice, that would be painful), but you do want to reason about algorithmic efficiency.

Jacobi

For the 3D Poisson problem:
	•	Classical Jacobi has a convergence factor per iteration \rho_J that:
	•	is < 1 (so it converges),
	•	but gets closer to 1 as N increases (i.e. convergence slows down on finer grids).
	•	In practice, per iteration it only shaves off a bit of residual, mostly killing high-frequency error.

So:
	•	Error reduction per Jacobi work unit is small:
\|r_{k+1}\| \approx \rho_J \|r_k\|, \quad \rho_J \approx 0.9\text{–}0.99
and \rho_J \to 1 as N grows.
	•	Effective iterations to reduce residual by 10⁸ grows like O(N²).

So algorithmically:
	•	Jacobi: cheap work unit, but you need many, many work units for the same accuracy.
	•	And that number worsens with problem size.

FMG

Multigrid (especially full multigrid / good V-cycle):
	•	Attacks all frequencies by moving error to coarser grids.
	•	Has grid-independent convergence:
	•	Convergence factor per cycle \rho_M is roughly independent of N.
	•	Typical \rho_M \sim 0.1 or better with reasonable smoothing.

So:
	•	Error reduction per FMG work unit is huge:
\|r_{k+1}\| \approx \rho_M \|r_k\|, \quad \rho_M \approx 0.1\text{ or }0.01
	•	The number of cycles needed to reduce residual by 10⁸ is:
k_\text{FMG} \approx \frac{\log 10^{-8}}{\log \rho_M},
which is O(1), independent of N.

So algorithmically:
	•	FMG: expensive work unit vs a single Jacobi iteration, but you need O(1) work units to reach a useful tolerance, regardless of N.
	•	Jacobi: cheap unit, but O(N²) units grow with problem size.

This is exactly the “parallel efficiency vs algorithmic efficiency” story you want:
	•	Jacobi might even have slightly nicer parallel scaling (especially at very high P) because it doesn’t have coarse-level overheads.
	•	But algorithmically it’s hopeless: scaling is destroyed by the fact that it needs more and more work units as N grows.

⸻

3. How to phrase expectations in the report

You can literally write something like:

Parallel efficiency.
For both Jacobi and FMG, the time per work unit decreases roughly as 1/P initially and then saturates due to halo communication. Cubic domain decompositions and custom MPI datatypes consistently reduce halo cost and yield better strong and weak scaling than sliced decompositions and plain NumPy-based communication. FMG cycles are more expensive than single Jacobi iterations by a constant factor, so FMG curves lie above Jacobi curves but exhibit similar parallel trends.

Algorithmic efficiency.
A Jacobi iteration reduces the residual only slightly and its convergence factor deteriorates as the grid is refined, leading to an iteration count that grows roughly like O(N^2) for a fixed error reduction. In contrast, a full multigrid cycle reduces the residual by a much larger, nearly grid-independent factor, so FMG requires O(1) cycles to reach a given tolerance. Thus, Jacobi is more parallel “friendly” per work unit but algorithmically inefficient, whereas FMG achieves nearly optimal O(N^3) complexity with essentially grid-independent convergence.

So in short, what you should expect to see:
	•	Time per work unit:
	•	FMG > Jacobi by a constant factor.
	•	Similar strong/weak scaling shapes.
	•	Cubic+custom best, sliced+NumPy worst, for both solvers.
	•	Error reduction:
	•	Jacobi: tiny progress per work unit, worse as N↑.
	•	FMG: big, almost N-independent drop per cycle.

Your plots then nicely show:
	•	Parallel story: “Look, for a single work unit, how do decomposition and communication impact scaling?”
	•	Algorithmic story: “Given what we know about convergence factors, FMG crushes Jacobi in total work to reach a physically meaningful solution.”

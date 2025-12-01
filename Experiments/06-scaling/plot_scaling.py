"""
Visualization of Scaling Experiments
=====================================

Strong and weak scaling analysis for the MPI Poisson solver.
Generates efficiency plots comparing decomposition and communication strategies.
Also compares Jacobi vs FMG solvers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from Poisson import get_project_root

# %%
# Setup
# -----

sns.set_theme()

# Get paths
repo_root = get_project_root()
data_dir = repo_root / "data" / "06-scaling"
fmg_dir = data_dir / "fmg"
fig_dir = repo_root / "figures" / "scaling"
fig_dir.mkdir(parents=True, exist_ok=True)

# %%
# Load Data
# ---------

def load_scaling_data(data_dir: Path, solver_type: str = "jacobi") -> pd.DataFrame:
    """Load all HDF5 scaling results into a DataFrame."""
    results = []

    for h5_file in data_dir.glob("*.h5"):
        try:
            df = pd.read_hdf(h5_file, key="results")
            df["solver"] = solver_type
            results.append(df)
        except Exception as e:
            print(f"Warning: Could not load {h5_file}: {e}")

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


# Load Jacobi data
df_jacobi = load_scaling_data(data_dir, "Jacobi")
print(f"Loaded {len(df_jacobi)} Jacobi results")

# Load FMG data
df_fmg = pd.DataFrame()
if fmg_dir.exists():
    df_fmg = load_scaling_data(fmg_dir, "FMG")
    print(f"Loaded {len(df_fmg)} FMG results")

# Combine
df_all = pd.concat([df_jacobi, df_fmg], ignore_index=True) if not df_fmg.empty else df_jacobi

if df_all.empty:
    print(f"No data found in {data_dir}")
    print("Run scaling experiments first:")
    print("  mpiexec -n P uv run python Experiments/06-scaling/jacobi_runner.py --N 64")
    import sys
    sys.exit(0)

print(f"\nTotal data points: {len(df_all)}")
print(f"Solvers: {df_all['solver'].unique().tolist()}")
print(f"Problem sizes: {sorted(df_all['N'].unique())}")
print(f"Rank counts: {sorted(df_all['mpi_size'].unique())}")

# %%
# Create Method Labels
# --------------------

def create_method_label(row):
    """Create a descriptive method label for legend."""
    decomp = row.get("decomposition", "unknown")
    comm = row.get("communicator", "unknown")
    return f"{decomp}/{comm}"

df_all["method"] = df_all.apply(create_method_label, axis=1)

# %%
# Compute Strong Scaling Metrics
# ------------------------------

def compute_strong_scaling(df: pd.DataFrame) -> pd.DataFrame:
    """Compute strong scaling speedup and efficiency.

    Strong scaling: Fixed problem size N, varying P.
    Speedup S(P) = T(1) / T(P)
    Efficiency E(P) = S(P) / P

    Uses ONE sequential baseline (p=1) per solver/N - decomposition doesn't
    matter at p=1 since there's no actual domain splitting.
    """
    results = []

    for solver in df["solver"].unique():
        df_solver = df[df["solver"] == solver]

        for N in df_solver["N"].unique():
            df_N = df_solver[df_solver["N"] == N].copy()

            # Get SINGLE sequential baseline (p=1) - method doesn't matter
            baseline_df = df_N[df_N["mpi_size"] == 1]
            if baseline_df.empty:
                # Fall back to minimum P
                min_p = df_N["mpi_size"].min()
                baseline_df = df_N[df_N["mpi_size"] == min_p]

            if baseline_df.empty:
                continue

            T1 = baseline_df["wall_time"].values[0]
            print(f"  {solver} N={N}: Sequential baseline T(1) = {T1:.4f}s")

            # Process all data points
            for _, row in df_N.iterrows():
                P = row["mpi_size"]
                T_P = row["wall_time"]
                speedup = T1 / T_P if T_P > 0 else 0
                efficiency = speedup / P if P > 0 else 0

                results.append({
                    "solver": solver,
                    "N": N,
                    "P": P,
                    "method": row["method"],
                    "decomposition": row.get("decomposition", "unknown"),
                    "communicator": row.get("communicator", "unknown"),
                    "wall_time": T_P,
                    "speedup": speedup,
                    "efficiency": efficiency * 100,  # as percentage
                    "iterations": row.get("iterations", 0),
                    "compute_time": row.get("total_compute_time", 0),
                    "halo_time": row.get("total_halo_time", 0),
                    "mpi_time": row.get("total_mpi_comm_time", 0),
                })

    return pd.DataFrame(results)


df_strong = compute_strong_scaling(df_all)
print(f"\nStrong scaling data points: {len(df_strong)}")

# %%
# Plot 1: Strong Scaling by Method (Jacobi only)
# ---------------------------------------------

df_jacobi_strong = df_strong[df_strong["solver"] == "Jacobi"]

if not df_jacobi_strong.empty:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Color map for methods
    methods = df_jacobi_strong["method"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    color_map = dict(zip(methods, colors))
    markers = {"sliced/numpy": "o", "sliced/custom": "s", "cubic/numpy": "^", "cubic/custom": "D"}

    # Left: Speedup vs P
    ax = axes[0]
    N_max = df_jacobi_strong["N"].max()
    df_plot = df_jacobi_strong[df_jacobi_strong["N"] == N_max]

    for method in sorted(methods):
        df_m = df_plot[df_plot["method"] == method].sort_values("P")
        marker = markers.get(method, "o")
        ax.plot(df_m["P"], df_m["speedup"], marker=marker, linestyle="-",
                color=color_map[method], label=method, markersize=8)

    # Ideal scaling line
    P_range = np.array(sorted(df_plot["P"].unique()))
    ax.plot(P_range, P_range, "k--", alpha=0.5, label="Ideal (S=P)")

    ax.set_xlabel("Number of Ranks (P)")
    ax.set_ylabel("Speedup S(P) = T(1)/T(P)")
    ax.set_title(f"Strong Scaling: Jacobi (N={N_max})")
    ax.legend(loc="upper left")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.grid(True, alpha=0.3)

    # Right: Efficiency vs P
    ax = axes[1]
    for method in sorted(methods):
        df_m = df_plot[df_plot["method"] == method].sort_values("P")
        marker = markers.get(method, "o")
        ax.plot(df_m["P"], df_m["efficiency"], marker=marker, linestyle="-",
                color=color_map[method], label=method, markersize=8)

    ax.axhline(y=100, color="k", linestyle="--", alpha=0.5, label="Ideal (100%)")
    ax.set_xlabel("Number of Ranks (P)")
    ax.set_ylabel("Parallel Efficiency (%)")
    ax.set_title(f"Strong Scaling: Jacobi (N={N_max})")
    ax.legend(loc="upper right")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(fig_dir / "01_strong_scaling_jacobi.pdf")
    print(f"Saved: {fig_dir / '01_strong_scaling_jacobi.pdf'}")

# %%
# Plot 2: Jacobi vs FMG Comparison
# --------------------------------

df_fmg_strong = df_strong[df_strong["solver"] == "FMG"]

if not df_jacobi_strong.empty and not df_fmg_strong.empty:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Find comparable problem sizes
    jacobi_N = df_jacobi_strong["N"].max()
    fmg_N = df_fmg_strong["N"].max()

    # Left: Wall time comparison
    ax = axes[0]

    # Jacobi - best method (cubic/custom typically)
    df_j = df_jacobi_strong[
        (df_jacobi_strong["N"] == jacobi_N) &
        (df_jacobi_strong["method"] == "cubic/custom")
    ].sort_values("P")

    # FMG (only has cubic/custom)
    df_f = df_fmg_strong[df_fmg_strong["N"] == fmg_N].sort_values("P")

    if not df_j.empty:
        ax.plot(df_j["P"], df_j["wall_time"], "o-", color="tab:blue",
                label=f"Jacobi 100iter (N={jacobi_N})", markersize=8)
    if not df_f.empty:
        ax.plot(df_f["P"], df_f["wall_time"], "s-", color="tab:orange",
                label=f"FMG 1cycle (N={fmg_N})", markersize=8)

    ax.set_xlabel("Number of Ranks (P)")
    ax.set_ylabel("Wall Time (seconds)")
    ax.set_title("Jacobi vs FMG: Wall Time")
    ax.legend()
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)

    # Right: Efficiency comparison
    ax = axes[1]

    if not df_j.empty:
        ax.plot(df_j["P"], df_j["efficiency"], "o-", color="tab:blue",
                label=f"Jacobi (N={jacobi_N})", markersize=8)
    if not df_f.empty:
        ax.plot(df_f["P"], df_f["efficiency"], "s-", color="tab:orange",
                label=f"FMG (N={fmg_N})", markersize=8)

    ax.axhline(y=100, color="k", linestyle="--", alpha=0.5, label="Ideal (100%)")
    ax.set_xlabel("Number of Ranks (P)")
    ax.set_ylabel("Parallel Efficiency (%)")
    ax.set_title("Jacobi vs FMG: Strong Scaling Efficiency")
    ax.legend()
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(fig_dir / "02_jacobi_vs_fmg.pdf")
    print(f"Saved: {fig_dir / '02_jacobi_vs_fmg.pdf'}")

# %%
# Plot 3: Timing Breakdown (Jacobi)
# ---------------------------------

if not df_jacobi_strong.empty and "compute_time" in df_jacobi_strong.columns:
    has_timing = df_jacobi_strong["compute_time"].notna().any() and (df_jacobi_strong["compute_time"] > 0).any()

    if has_timing:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Use cubic/custom for breakdown
        N_max = df_jacobi_strong["N"].max()
        df_plot = df_jacobi_strong[
            (df_jacobi_strong["method"] == "cubic/custom") &
            (df_jacobi_strong["N"] == N_max)
        ].sort_values("P")

        if df_plot.empty:
            # Fall back to any available method
            df_plot = df_jacobi_strong[df_jacobi_strong["N"] == N_max].sort_values("P")

        # Stack plot
        P = df_plot["P"].values
        compute = df_plot["compute_time"].fillna(0).values
        halo = df_plot["halo_time"].fillna(0).values
        mpi = df_plot["mpi_time"].fillna(0).values

        x = np.arange(len(P))
        width = 0.6

        ax.bar(x, compute, width, label="Compute", color="tab:blue")
        ax.bar(x, halo, width, bottom=compute, label="Halo Exchange", color="tab:orange")
        ax.bar(x, mpi, width, bottom=compute + halo, label="MPI Allreduce", color="tab:green")

        ax.set_xticks(x)
        ax.set_xticklabels([str(int(p)) for p in P])
        ax.set_xlabel("Number of Ranks (P)")
        ax.set_ylabel("Time (seconds)")
        ax.set_title(f"Timing Breakdown: Jacobi cubic/custom (N={N_max})")
        ax.legend()

        plt.tight_layout()
        fig.savefig(fig_dir / "03_timing_breakdown.pdf")
        print(f"Saved: {fig_dir / '03_timing_breakdown.pdf'}")

# %%
# Plot 4: Decomposition Comparison
# --------------------------------

if not df_jacobi_strong.empty:
    fig, ax = plt.subplots(figsize=(8, 5))

    N_max = df_jacobi_strong["N"].max()
    df_N = df_jacobi_strong[df_jacobi_strong["N"] == N_max]

    # Sliced methods (dashed)
    for comm in ["numpy", "custom"]:
        df_m = df_N[(df_N["decomposition"] == "sliced") & (df_N["communicator"] == comm)].sort_values("P")
        if not df_m.empty:
            marker = "o" if comm == "numpy" else "s"
            ax.plot(df_m["P"], df_m["efficiency"], marker=marker, linestyle="--",
                    label=f"sliced/{comm}", alpha=0.8, markersize=8)

    # Cubic methods (solid)
    for comm in ["numpy", "custom"]:
        df_m = df_N[(df_N["decomposition"] == "cubic") & (df_N["communicator"] == comm)].sort_values("P")
        if not df_m.empty:
            marker = "^" if comm == "numpy" else "D"
            ax.plot(df_m["P"], df_m["efficiency"], marker=marker, linestyle="-",
                    label=f"cubic/{comm}", alpha=0.8, markersize=8)

    ax.axhline(y=100, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("Number of Ranks (P)")
    ax.set_ylabel("Parallel Efficiency (%)")
    ax.set_title(f"Decomposition Strategy Comparison (N={N_max})")
    ax.legend(loc="best")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(fig_dir / "04_decomposition_comparison.pdf")
    print(f"Saved: {fig_dir / '04_decomposition_comparison.pdf'}")

# %%
# Summary Statistics
# ------------------

print("\n" + "=" * 60)
print("SCALING ANALYSIS SUMMARY")
print("=" * 60)

for solver in df_strong["solver"].unique():
    df_s = df_strong[df_strong["solver"] == solver]
    print(f"\n{solver} Strong Scaling Results:")
    print("-" * 40)

    summary = df_s.groupby(["N", "method"]).agg({
        "P": "max",
        "speedup": "max",
        "efficiency": lambda x: x.iloc[-1] if len(x) > 0 else 0  # efficiency at max P
    }).round(2)
    print(summary.to_string())

print("\n" + "=" * 60)

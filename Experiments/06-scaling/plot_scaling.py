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
data_base = repo_root / "data"
# MLflow downloads to separate directories per experiment
jacobi_dirs = [
    data_base / "06-scaling",  # Original expected location
    data_base / "06-scaling-strong_scaling",
    data_base / "06-scaling-weak_scaling",
    data_base / "06-scaling-single_socket_strong",
    data_base / "06-scaling-single_socket_weak",
    data_base / "06-scaling-test",  # Local test data with timeseries
]
fmg_dirs = [
    data_base / "06-scaling" / "fmg",  # Original expected location
    data_base / "06-scaling-fmg_strong",
    data_base / "06-scaling-fmg_weak",
]
fig_dir = repo_root / "figures" / "scaling"
fig_dir.mkdir(parents=True, exist_ok=True)

# %%
# Load Data
# ---------

def load_scaling_data(data_dirs: list[Path], solver_type: str = "jacobi") -> pd.DataFrame:
    """Load all HDF5 scaling results from multiple directories into a DataFrame."""
    results = []

    for data_dir in data_dirs:
        if not data_dir.exists():
            continue
        for h5_file in data_dir.glob("*.h5"):
            try:
                df = pd.read_hdf(h5_file, key="results")
                df["solver"] = solver_type
                df["source_file"] = str(h5_file)
                results.append(df)
            except Exception as e:
                print(f"Warning: Could not load {h5_file}: {e}")

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


def load_timeseries_data(data_dirs: list[Path], solver_type: str = "jacobi", warmup_iters: int = 5) -> pd.DataFrame:
    """Load per-iteration timeseries data for CI plotting.

    Returns DataFrame with columns: mpi_size, N, method, iteration, iter_time, speedup
    """
    all_timeseries = []
    baselines = {}  # (N, solver) -> mean baseline iter_time

    for data_dir in data_dirs:
        if not data_dir.exists():
            continue
        for h5_file in data_dir.glob("*.h5"):
            try:
                # Check if timeseries exists
                df_ts = pd.read_hdf(h5_file, key="timeseries")
                df_res = pd.read_hdf(h5_file, key="results")

                N = int(df_res["N"].iloc[0])
                P = int(df_res["mpi_size"].iloc[0])
                decomp = str(df_res["decomposition"].iloc[0])
                comm = str(df_res["communicator"].iloc[0])
                method = f"{decomp}/{comm}"

                # Compute total iteration time (mpi_comm_times may not exist with non-blocking)
                df_ts["iter_time"] = (
                    df_ts["compute_times"].fillna(0) +
                    df_ts["halo_exchange_times"].fillna(0)
                )
                # Add mpi_comm_times if present (legacy data)
                if "mpi_comm_times" in df_ts.columns:
                    df_ts["iter_time"] += df_ts["mpi_comm_times"].fillna(0)
                df_ts["mpi_size"] = P
                df_ts["N"] = N
                df_ts["method"] = method
                df_ts["solver"] = solver_type
                df_ts["iteration"] = range(len(df_ts))

                # Skip warmup iterations
                df_ts = df_ts[df_ts["iteration"] >= warmup_iters].copy()

                # Store baseline (P=1) for later speedup computation
                if P == 1:
                    key = (N, solver_type)
                    baselines[key] = df_ts["iter_time"].mean()

                all_timeseries.append(df_ts)

            except KeyError:
                # No timeseries data in this file
                pass
            except Exception as e:
                print(f"Warning: Could not load timeseries from {h5_file}: {e}")

    if not all_timeseries:
        return pd.DataFrame()

    df = pd.concat(all_timeseries, ignore_index=True)

    # Compute speedup using baseline
    def compute_speedup(row):
        key = (row["N"], row["solver"])
        baseline = baselines.get(key)
        if baseline and baseline > 0:
            return baseline / row["iter_time"]
        return np.nan

    df["speedup"] = df.apply(compute_speedup, axis=1)
    df["efficiency"] = df["speedup"] / df["mpi_size"] * 100

    return df


# Load Jacobi data
df_jacobi = load_scaling_data(jacobi_dirs, "Jacobi")
print(f"Loaded {len(df_jacobi)} Jacobi results")

# Load FMG data
df_fmg = load_scaling_data(fmg_dirs, "FMG")
print(f"Loaded {len(df_fmg)} FMG results")

# Load timeseries data for CI plots (if available)
df_ts_jacobi = load_timeseries_data(jacobi_dirs, "Jacobi")
df_ts_fmg = load_timeseries_data(fmg_dirs, "FMG")
df_ts_all = pd.concat([df_ts_jacobi, df_ts_fmg], ignore_index=True) if not df_ts_fmg.empty else df_ts_jacobi
has_timeseries = not df_ts_all.empty
print(f"Loaded {len(df_ts_all)} timeseries data points (for CI plots): {'Yes' if has_timeseries else 'No'}")

# Combine
df_all = pd.concat([df_jacobi, df_fmg], ignore_index=True) if not df_fmg.empty else df_jacobi

if df_all.empty:
    print(f"No data found in scaling directories")
    print("Run scaling experiments first or fetch from MLflow:")
    print("  uv run python main.py --fetch")
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

    # Filter out rows with invalid wall_time
    df = df[df["wall_time"].apply(lambda x: x not in (None, 'None', ''))]
    df = df.copy()
    df["wall_time"] = df["wall_time"].astype(float)

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

            T1 = float(baseline_df["wall_time"].values[0])
            print(f"  {solver} N={N}: Sequential baseline T(1) = {T1:.4f}s")

            # Process all data points
            for _, row in df_N.iterrows():
                P = int(row["mpi_size"])
                T_P = float(row["wall_time"])
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
# Uses seaborn CI plots when timeseries data is available

df_jacobi_strong = df_strong[df_strong["solver"] == "Jacobi"]

if not df_jacobi_strong.empty:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Color map for methods
    methods = df_jacobi_strong["method"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    color_map = dict(zip(methods, colors))
    markers = {"sliced/numpy": "o", "sliced/custom": "s", "cubic/numpy": "^", "cubic/custom": "D"}
    palette = {m: color_map[m] for m in methods}

    # For strong scaling, find N with most P values (not just max N)
    N_counts = df_jacobi_strong.groupby("N")["P"].nunique()
    N_strong = N_counts.idxmax()  # N with most unique P values
    df_plot = df_jacobi_strong[df_jacobi_strong["N"] == N_strong]
    P_range = np.array(sorted(df_plot["P"].unique()))

    # Check if we have timeseries data for CI plots
    df_ts_jacobi_N = df_ts_all[(df_ts_all["solver"] == "Jacobi") & (df_ts_all["N"] == N_strong)] if has_timeseries else pd.DataFrame()
    use_ci = not df_ts_jacobi_N.empty

    # Left: Speedup vs P
    ax = axes[0]
    if use_ci:
        # Use seaborn lineplot with confidence intervals
        sns.lineplot(
            data=df_ts_jacobi_N, x="mpi_size", y="speedup", hue="method",
            style="method", markers=markers, dashes=False,
            palette=palette, errorbar=("ci", 95), ax=ax, markersize=8
        )
        ax.set_xlabel("Number of Ranks (P)")
        ax.set_ylabel("Speedup S(P) = T(1)/T(P)")
    else:
        # Fall back to simple line plot
        for method in sorted(methods):
            df_m = df_plot[df_plot["method"] == method].sort_values("P")
            marker = markers.get(method, "o")
            ax.plot(df_m["P"], df_m["speedup"], marker=marker, linestyle="-",
                    color=color_map[method], label=method, markersize=8)
        ax.set_xlabel("Number of Ranks (P)")
        ax.set_ylabel("Speedup S(P) = T(1)/T(P)")

    # Ideal scaling line
    ax.plot(P_range, P_range, "k--", alpha=0.5, label="Ideal (S=P)")

    ax.set_title(f"Strong Scaling: Jacobi (N={N_strong})" + (" [95% CI]" if use_ci else ""))
    ax.legend(loc="upper left")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(P_range)
    ax.set_xticklabels([str(int(p)) for p in P_range])
    ax.set_yticks(P_range)
    ax.set_yticklabels([str(int(p)) for p in P_range])
    ax.grid(True, alpha=0.3)

    # Right: Efficiency vs P
    ax = axes[1]
    if use_ci:
        sns.lineplot(
            data=df_ts_jacobi_N, x="mpi_size", y="efficiency", hue="method",
            style="method", markers=markers, dashes=False,
            palette=palette, errorbar=("ci", 95), ax=ax, markersize=8
        )
        ax.set_xlabel("Number of Ranks (P)")
        ax.set_ylabel("Parallel Efficiency (%)")
    else:
        for method in sorted(methods):
            df_m = df_plot[df_plot["method"] == method].sort_values("P")
            marker = markers.get(method, "o")
            ax.plot(df_m["P"], df_m["efficiency"], marker=marker, linestyle="-",
                    color=color_map[method], label=method, markersize=8)
        ax.set_xlabel("Number of Ranks (P)")
        ax.set_ylabel("Parallel Efficiency (%)")

    ax.axhline(y=100, color="k", linestyle="--", alpha=0.5, label="Ideal (100%)")
    ax.set_title(f"Strong Scaling: Jacobi (N={N_strong})" + (" [95% CI]" if use_ci else ""))
    ax.legend(loc="upper right")
    ax.set_xscale("log", base=2)
    ax.set_xticks(P_range)
    ax.set_xticklabels([str(int(p)) for p in P_range])
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
# Plot 5: Timing Fractions vs Ranks
# ---------------------------------
# Shows how compute/halo/other fractions change with increasing parallelism
# "Other" = iter_time - compute - halo (captures MPI overhead, sync, etc.)

if has_timeseries:
    # Get Jacobi timeseries data for timing fractions
    df_ts_jacobi = df_ts_all[df_ts_all["solver"] == "Jacobi"].copy()

    if not df_ts_jacobi.empty:
        # Find N with most P values for strong scaling analysis
        N_counts = df_ts_jacobi.groupby("N")["mpi_size"].nunique()
        N_strong = N_counts.idxmax()
        df_ts_N = df_ts_jacobi[df_ts_jacobi["N"] == N_strong].copy()

        # Compute fractions: Compute, Halo, Other (remainder)
        compute = df_ts_N["compute_times"].fillna(0)
        halo = df_ts_N["halo_exchange_times"].fillna(0)
        total = df_ts_N["iter_time"]  # Already computed in load_timeseries_data
        other = (total - compute - halo).clip(lower=0)  # Clip to handle floating point

        df_ts_N["compute_frac"] = compute / total * 100
        df_ts_N["halo_frac"] = halo / total * 100
        df_ts_N["other_frac"] = other / total * 100

        # Filter for a single method (cubic/custom is typically best)
        methods_available = df_ts_N["method"].unique()
        target_method = "cubic/custom" if "cubic/custom" in methods_available else methods_available[0]
        df_ts_method = df_ts_N[df_ts_N["method"] == target_method]

        if not df_ts_method.empty:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Melt data for seaborn
            df_fracs = df_ts_method[["mpi_size", "compute_frac", "halo_frac", "other_frac"]].melt(
                id_vars=["mpi_size"],
                value_vars=["compute_frac", "halo_frac", "other_frac"],
                var_name="component",
                value_name="fraction"
            )

            # Rename for legend
            component_names = {
                "compute_frac": "Compute",
                "halo_frac": "Halo Exchange",
                "other_frac": "Other (MPI, sync)"
            }
            df_fracs["component"] = df_fracs["component"].map(component_names)

            # Plot with CI
            sns.lineplot(
                data=df_fracs, x="mpi_size", y="fraction", hue="component",
                style="component", markers=True, dashes=False,
                palette={"Compute": "tab:blue", "Halo Exchange": "tab:orange", "Other (MPI, sync)": "tab:green"},
                errorbar=("ci", 95), ax=ax, markersize=8
            )

            P_range = np.array(sorted(df_ts_method["mpi_size"].unique()))
            ax.set_xlabel("Number of Ranks (P)")
            ax.set_ylabel("Time Fraction (%)")
            ax.set_title(f"Timing Breakdown: {target_method} (N={N_strong}) [95% CI]")
            ax.set_xscale("log", base=2)
            ax.set_xticks(P_range)
            ax.set_xticklabels([str(int(p)) for p in P_range])
            ax.set_ylim(0, 105)
            ax.legend(title="Component")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            fig.savefig(fig_dir / "05_timing_fractions.pdf")
            print(f"Saved: {fig_dir / '05_timing_fractions.pdf'}")

# %%
# Plot 6: Weak Scaling (if data available)
# ----------------------------------------
# Weak scaling: N scales with P to keep work per rank constant

def compute_weak_scaling(df: pd.DataFrame) -> pd.DataFrame:
    """Compute weak scaling efficiency.

    Weak scaling: Problem size N grows with P.
    Efficiency E(P) = T(1) / T(P) where N(P) ~ P * N(1)
    """
    results = []

    # Filter out rows with invalid wall_time
    df = df[df["wall_time"].apply(lambda x: x not in (None, 'None', ''))]
    df = df.copy()
    df["wall_time"] = df["wall_time"].astype(float)

    for solver in df["solver"].unique():
        df_solver = df[df["solver"] == solver]

        # Group by method
        for method in df_solver["method"].unique():
            df_method = df_solver[df_solver["method"] == method]

            # Get baseline at P=1
            baseline = df_method[df_method["mpi_size"] == 1]
            if baseline.empty:
                continue

            T1 = float(baseline["wall_time"].values[0])
            N1 = int(baseline["N"].values[0])

            for _, row in df_method.iterrows():
                P = int(row["mpi_size"])
                T_P = float(row["wall_time"])
                N = int(row["N"])

                # For weak scaling, efficiency is T1/T(P)
                efficiency = (T1 / T_P) * 100 if T_P > 0 else 0

                results.append({
                    "solver": solver,
                    "N": N,
                    "P": P,
                    "method": method,
                    "wall_time": T_P,
                    "efficiency": efficiency,
                    "work_per_rank": (N ** 3) / P,  # For reference
                })

    return pd.DataFrame(results)


# Check if we have weak scaling data (N varies with P)
df_jacobi_only = df_all[df_all["solver"] == "Jacobi"]
if not df_jacobi_only.empty:
    # Group by method and check if N varies
    has_weak = False
    for method in df_jacobi_only["method"].unique():
        df_m = df_jacobi_only[df_jacobi_only["method"] == method]
        N_values = df_m["N"].nunique()
        P_values = df_m["mpi_size"].nunique()
        if N_values > 1 and P_values > 1:
            has_weak = True
            break

    if has_weak:
        df_weak = compute_weak_scaling(df_jacobi_only)

        if not df_weak.empty and len(df_weak) > 2:
            fig, ax = plt.subplots(figsize=(8, 5))

            methods = df_weak["method"].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
            color_map = dict(zip(methods, colors))
            base_markers = {"sliced/numpy": "o", "sliced/custom": "s", "cubic/numpy": "^", "cubic/custom": "D"}
            # Ensure all methods have a marker (fallback to 'o')
            markers = {m: base_markers.get(m, "o") for m in methods}
            palette = {m: color_map[m] for m in methods}

            # Check if we have timeseries data for weak scaling CI
            df_ts_weak = df_ts_all[df_ts_all["solver"] == "Jacobi"].copy() if has_timeseries else pd.DataFrame()
            use_ci_weak = not df_ts_weak.empty and df_ts_weak["N"].nunique() > 1

            if use_ci_weak:
                # Compute weak scaling efficiency per iteration
                # Get baseline (P=1) iter_time per method
                baselines_weak = {}
                for method in df_ts_weak["method"].unique():
                    df_m = df_ts_weak[(df_ts_weak["method"] == method) & (df_ts_weak["mpi_size"] == 1)]
                    if not df_m.empty:
                        baselines_weak[method] = df_m["iter_time"].mean()

                def compute_weak_eff(row):
                    baseline = baselines_weak.get(row["method"])
                    if baseline and baseline > 0:
                        return (baseline / row["iter_time"]) * 100
                    return np.nan

                df_ts_weak["weak_efficiency"] = df_ts_weak.apply(compute_weak_eff, axis=1)

                sns.lineplot(
                    data=df_ts_weak, x="mpi_size", y="weak_efficiency", hue="method",
                    style="method", markers=markers, dashes=False,
                    palette=palette, errorbar=("ci", 95), ax=ax, markersize=8
                )
                ax.set_ylabel("Weak Scaling Efficiency (%)")
                ax.set_title("Weak Scaling: Jacobi [95% CI]")
            else:
                for method in sorted(methods):
                    df_m = df_weak[df_weak["method"] == method].sort_values("P")
                    marker = markers.get(method, "o")
                    ax.plot(df_m["P"], df_m["efficiency"], marker=marker, linestyle="-",
                            color=color_map[method], label=method, markersize=8)
                ax.set_ylabel("Weak Scaling Efficiency (%)")
                ax.set_title("Weak Scaling: Jacobi")

            ax.axhline(y=100, color="k", linestyle="--", alpha=0.5, label="Ideal (100%)")
            P_range = np.array(sorted(df_weak["P"].unique()))
            ax.set_xlabel("Number of Ranks (P)")
            ax.legend(loc="best")
            ax.set_xscale("log", base=2)
            ax.set_xticks(P_range)
            ax.set_xticklabels([str(int(p)) for p in P_range])
            ax.set_ylim(0, 110)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            fig.savefig(fig_dir / "06_weak_scaling.pdf")
            print(f"Saved: {fig_dir / '06_weak_scaling.pdf'}")

# %%
# Plot 7: Operation Timeseries (Jacobi vs FMG comparison)
# -------------------------------------------------------
# Shows compute and halo times per operation - FMG reveals grid hierarchy

if has_timeseries:
    # Load FMG data with level_indices first to check if available
    df_fmg_ts = pd.DataFrame()
    for data_dir in fmg_dirs + [data_base / "06-scaling-test"]:
        if not data_dir.exists():
            continue
        for h5_file in data_dir.glob("*.h5"):
            try:
                df_ts = pd.read_hdf(h5_file, key="timeseries")
                if "level_indices" in df_ts.columns:
                    df_res = pd.read_hdf(h5_file, key="results")
                    df_ts["N"] = int(df_res["N"].iloc[0])
                    df_ts["mpi_size"] = int(df_res["mpi_size"].iloc[0])
                    df_fmg_ts = pd.concat([df_fmg_ts, df_ts], ignore_index=True)
            except:
                pass

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Jacobi timeseries (should be flat)
    df_ts_jac = df_ts_all[df_ts_all["solver"] == "Jacobi"].copy()
    if not df_ts_jac.empty:
        # Pick largest N
        N_jac = df_ts_jac["N"].max()
        P_jac = df_ts_jac[df_ts_jac["N"] == N_jac]["mpi_size"].max()
        df_jac = df_ts_jac[(df_ts_jac["N"] == N_jac) & (df_ts_jac["mpi_size"] == P_jac)].head(100).copy()
        df_jac = df_jac.reset_index(drop=True)
        df_jac["operation"] = range(len(df_jac))

        ax = axes[0, 0]
        ax.plot(df_jac["operation"], df_jac["compute_times"] * 1000, "b-", label="Compute", alpha=0.8)
        ax.plot(df_jac["operation"], df_jac["halo_exchange_times"] * 1000, "r-", label="Halo", alpha=0.8)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Time (ms)")
        ax.set_title(f"Jacobi Timeseries (N={N_jac}, P={P_jac})")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Top-right & Bottom: FMG timeseries
    if not df_fmg_ts.empty:
        N_fmg = df_fmg_ts["N"].max()
        P_fmg = df_fmg_ts[df_fmg_ts["N"] == N_fmg]["mpi_size"].min()
        df_fmg = df_fmg_ts[(df_fmg_ts["N"] == N_fmg) & (df_fmg_ts["mpi_size"] == P_fmg)].copy()
        df_fmg = df_fmg.reset_index(drop=True)
        df_fmg["operation"] = range(len(df_fmg))

        # Top-right: FMG compute times (full view, log scale)
        ax = axes[0, 1]
        ax.plot(df_fmg["operation"], df_fmg["compute_times"] * 1000, "b-", label="Compute", alpha=0.8, linewidth=0.5)
        ax.plot(df_fmg["operation"], df_fmg["halo_exchange_times"] * 1000, "r-", label="Halo", alpha=0.8, linewidth=0.5)
        ax.set_xlabel("Operation")
        ax.set_ylabel("Time (ms) - log scale")
        ax.set_yscale("log")
        ax.set_title(f"FMG Timeseries (N={N_fmg}, P={P_fmg}) - Full")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Bottom-left: FMG level traversal (shows ladder/V-cycle pattern)
        ax = axes[1, 0]
        n_levels = int(df_fmg["level_indices"].max()) + 1
        ax.plot(df_fmg["operation"], df_fmg["level_indices"], "k-", linewidth=0.8)
        ax.set_xlabel("Operation")
        ax.set_ylabel("Grid Level (0=finest, 4=coarsest)")
        ax.set_title("FMG Level Traversal - V-Cycle Pattern")
        ax.set_yticks(range(n_levels))
        ax.set_yticklabels([f"Level {i}" for i in range(n_levels)])
        ax.invert_yaxis()  # Fine at top
        ax.grid(True, alpha=0.3)

        # Bottom-right: Zoomed FMG around first fine grid ops
        ax = axes[1, 1]
        if "level_indices" in df_fmg.columns:
            fine_mask = df_fmg["level_indices"] == 0
            if fine_mask.any():
                first_fine = fine_mask.idxmax()
                start_idx = max(0, first_fine - 50)
                df_zoom = df_fmg.iloc[start_idx:start_idx + 200].copy()
                df_zoom = df_zoom.reset_index(drop=True)
                df_zoom["op"] = range(len(df_zoom))

                ax.plot(df_zoom["op"], df_zoom["compute_times"] * 1000, "b-", label="Compute", alpha=0.8)
                ax.plot(df_zoom["op"], df_zoom["halo_exchange_times"] * 1000, "r-", label="Halo", alpha=0.8)
                ax.set_xlabel("Operation (zoomed)")
                ax.set_ylabel("Time (ms) - log scale")
                ax.set_yscale("log")
                ax.set_title("FMG Zoomed - V-Cycles at Fine Grid")
                ax.legend()
                ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(fig_dir / "07_timeseries.pdf")
    print(f"Saved: {fig_dir / '07_timeseries.pdf'}")

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

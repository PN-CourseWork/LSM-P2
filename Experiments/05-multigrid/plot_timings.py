"""
Plotting script for Jacobi vs FMG timings.
Reveals grid traversal patterns.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from Poisson import get_project_root

# Setup paths
repo_root = get_project_root()
data_dir = repo_root / "data" / "05-multigrid"
figure_dir = repo_root / "figures" / "multigrid"
figure_dir.mkdir(parents=True, exist_ok=True)

# Input files
jacobi_file = data_dir / "timings_jacobi.h5"
fmg_file = data_dir / "timings_fmg.h5"

def load_timeseries(path, method_name):
    if not path.exists():
        print(f"Warning: {path} not found.")
        return pd.DataFrame()
    
    try:
        # Read the 'timeseries' key from HDF5
        df = pd.read_hdf(path, key="timeseries")
        df["method"] = method_name
        df["step"] = df.index
        return df
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return pd.DataFrame()

# Load data
df_jacobi = load_timeseries(jacobi_file, "Jacobi")
df_fmg = load_timeseries(fmg_file, "FMG")

if df_jacobi.empty or df_fmg.empty:
    print("Error: Missing data. Run compute_timings.py first.")
    exit(1)

# Combine
df_all = pd.concat([df_jacobi, df_fmg], ignore_index=True)

# Melt to have "Timing Type" (Compute vs Halo)
# We are interested in 'compute_times' and 'halo_exchange_times'
df_melted = df_all.melt(
    id_vars=["method", "step"], 
    value_vars=["compute_times", "halo_exchange_times"],
    var_name="timing_type", 
    value_name="time_sec"
)

# Rename for cleaner legend
df_melted["timing_type"] = df_melted["timing_type"].replace({
    "compute_times": "Compute",
    "halo_exchange_times": "Halo Exchange"
})

# Plot
print("Generating plot...")
sns.set_theme(style="whitegrid")

# Relplot: separate columns for Method
g = sns.relplot(
    data=df_melted,
    x="step",
    y="time_sec",
    hue="timing_type",
    col="method",
    kind="line",
    height=5,
    aspect=1.5,
    facet_kws={"sharex": False, "sharey": False} # FMG has fewer steps but varying times, Jacobi has many steps constant time
)

g.set_titles("{col_name}")
g.set_axis_labels("Step / Iteration", "Time (s)")

# Adjust titles/layout
plt.subplots_adjust(top=0.85)
g.fig.suptitle("Grid Traversal Patterns: Jacobi vs FMG (N=257, 8 Ranks)", fontsize=16)

output_path = figure_dir / "timings_traversal_pattern.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved plot to {output_path}")

"""
Communication Analysis: Contiguous vs Non-Contiguous
=====================================================

Compares NumPy array copies vs MPI custom datatypes for halo exchange,
demonstrating the benefit of zero-copy communication for non-contiguous data.

Uses per-iteration timeseries data for tight confidence intervals.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from Poisson import get_project_root

# %%
# Setup
# -----

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100

repo_root = get_project_root()
data_dir = repo_root / "data" / "03-communication"
fig_dir = repo_root / "figures" / "communication"
fig_dir.mkdir(parents=True, exist_ok=True)

# %%
# Load Data
# ---------

parquet_files = list(data_dir.glob("communication_*.parquet"))
if not parquet_files:
    raise FileNotFoundError(
        f"No data found in {data_dir}. Run compute_communication.py first."
    )

df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

print(f"Loaded {len(df)} per-iteration measurements")
print(f"Configurations: {df['label'].unique()}")
print(f"Problem sizes: {sorted(df['N'].unique())}")

# %%
# Halo Exchange Time vs Problem Size
# ----------------------------------
# Shows all 4 configurations with 95% CI from per-iteration data

fig, ax = plt.subplots(figsize=(10, 6))

palette = {
    "NumPy (Z-axis, contiguous)": "#1f77b4",
    "Custom (Z-axis, contiguous)": "#2ca02c",
    "NumPy (X-axis, non-contiguous)": "#d62728",
    "Custom (X-axis, non-contiguous)": "#ff7f0e",
}

sns.lineplot(
    data=df,
    x="local_N",
    y="halo_time_us",
    hue="label",
    style="label",
    markers=True,
    dashes=False,
    palette=palette,
    ax=ax,
    errorbar=("ci", 95),
    markersize=8,
    linewidth=2,
)

ax.set_xlabel("Local Subdomain Size (N / nprocs)", fontsize=12)
ax.set_ylabel("Halo Exchange Time (μs)", fontsize=12)
ax.set_title(
    "Halo Exchange Performance: Contiguous vs Non-Contiguous Memory", fontsize=13
)
ax.legend(title="Configuration", fontsize=9, loc="upper left")
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_file = fig_dir / "communication_comparison.pdf"
plt.savefig(output_file, bbox_inches="tight")
print(f"Saved: {output_file}")
#plt.show()

# %%
# Summary Statistics
# ------------------

print("\n" + "=" * 70)
print("Summary: Mean halo time (μs) with 95% CI by local subdomain size")
print("=" * 70)
summary = df.groupby(["local_N", "label"])["halo_time_us"].agg(["mean", "std", "count"])
summary["ci95"] = 1.96 * summary["std"] / (summary["count"] ** 0.5)
summary["display"] = summary.apply(
    lambda r: f"{r['mean']:.1f} ± {r['ci95']:.1f}", axis=1
)
print(summary["display"].unstack().to_string())

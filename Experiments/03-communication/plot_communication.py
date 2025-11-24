"""
Communication Method Visualization
====================================

Visualize communication overhead comparison between MPI datatypes and NumPy arrays.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup
sns.set_context("paper")
sns.set_style("whitegrid")

# Get paths
repo_root = Path(__file__).resolve().parent.parent.parent
data_dir = repo_root / "data" / "communication"
fig_dir = repo_root / "figures" / "communication"
fig_dir.mkdir(parents=True, exist_ok=True)

# %%
# Load all communication data
# ----------------------------

parquet_files = list(data_dir.glob("communication_*.parquet"))
if not parquet_files:
    print("No data found. Run compute_communication.py first.")
    exit(1)

dfs = [pd.read_parquet(f) for f in parquet_files]
df = pd.concat(dfs, ignore_index=True)

# Convert time to microseconds for plotting
df['time_us'] = df['time'] * 1e6

print(f"Loaded {len(df)} measurements")
print(f"Strategies: {df['strategy'].unique()}")
print(f"Sizes: {sorted(df['size'].unique())}")
print(f"Problem sizes N: {sorted(df['N'].unique())}")

# %%
# Plot 1: Communication Time vs Problem Size
# --------------------------------------------
# Combined plot: hue=method, style=strategy

fig, ax = plt.subplots(figsize=(10, 6))

# Plot with seaborn lineplot (automatically computes error bars across repetitions)
sns.lineplot(
    data=df,
    x='N',
    y='time_us',
    hue='method',
    style='strategy',
    markers=True,
    dashes=False,
    ax=ax,
    errorbar=('ci', 95),  # 95% confidence interval
    markersize=8,
    linewidth=2
)

ax.set_xlabel('Grid Size N')
ax.set_ylabel('Time per Exchange (μs)')
ax.set_title('Communication Overhead: Method and Strategy Comparison')
ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.legend(title='', fontsize=10, loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_file = fig_dir / "01_communication_scaling.pdf"
plt.savefig(output_file, bbox_inches='tight')
print(f"Saved: {output_file}")
plt.close()

# %%
# Plot 2: Speedup Relative to NumPy Baseline (with error bars)
# ---------------------------------------------------------------

# Compute normalized performance for each individual measurement
# Normalized to each strategy's numpy baseline
normalized_data = []

for (N, strategy), group in df.groupby(['N', 'strategy']):
    # Get mean numpy time for this configuration as baseline
    numpy_times = group[group['method'] == 'numpy']['time']
    if len(numpy_times) == 0:
        continue

    baseline_time = numpy_times.mean()

    # Normalize all measurements to baseline
    for _, row in group.iterrows():
        normalized_data.append({
            'N': row['N'],
            'strategy': row['strategy'],
            'method': row['method'],
            'repetition': row['repetition'],
            'speedup': baseline_time / row['time']  # >1 means faster than baseline
        })

df_normalized = pd.DataFrame(normalized_data)

fig, ax = plt.subplots(figsize=(10, 6))

# Use relplot-style lineplot with hue and style
sns.lineplot(
    data=df_normalized,
    x='N',
    y='speedup',
    hue='strategy',
    style='method',
    markers=True,
    dashes=False,
    ax=ax,
    errorbar=('ci', 95),  # 95% confidence interval
    markersize=8,
    linewidth=2
)

# Add reference line at speedup=1 (numpy baseline)
ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.4, linewidth=1.5, label='NumPy baseline')

ax.set_xlabel('Grid Size N', fontsize=12)
ax.set_ylabel('Speedup vs NumPy Baseline', fontsize=12)
ax.set_title('Communication Performance: Normalized to NumPy Baseline', fontsize=13)
ax.set_xscale('log', base=2)
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_file = fig_dir / "02_speedup_vs_baseline.pdf"
plt.savefig(output_file, bbox_inches='tight')
print(f"Saved: {output_file}")
plt.close()

print("\nVisualization complete!")

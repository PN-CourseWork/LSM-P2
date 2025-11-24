"""
Communication Method Visualization
====================================

Visualize communication overhead comparison between MPI datatypes and NumPy arrays.
"""
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
# Plot 1: Communication Time vs Problem Size (by strategy)
# ---------------------------------------------------------

strategies = sorted(df['strategy'].unique())
n_strategies = len(strategies)

fig, axes = plt.subplots(1, n_strategies, figsize=(6*n_strategies, 4))
if n_strategies == 1:
    axes = [axes]

for idx, strategy in enumerate(strategies):
    ax = axes[idx]

    df_strategy = df[df['strategy'] == strategy]

    # Plot with seaborn lineplot (automatically computes error bars across repetitions)
    sns.lineplot(
        data=df_strategy,
        x='N',
        y='time_us',
        hue='method',
        style='size',
        markers=True,
        dashes=False,
        ax=ax,
        errorbar='sd'  # Standard deviation across repetitions
    )

    ax.set_xlabel('Grid Size N')
    ax.set_ylabel('Time per Exchange (μs)')
    ax.set_title(f'{strategy.capitalize()} Decomposition')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.legend(title='', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
output_file = fig_dir / "01_communication_scaling.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Saved: {output_file}")
plt.close()

# %%
# Plot 2: Speedup Analysis (only for configurations with both methods)
# ----------------------------------------------------------------------

# Compute speedup by taking ratio of means per (N, size, strategy)
speedup_data = []
for (N, size, strategy), group in df.groupby(['N', 'size', 'strategy']):
    methods = group['method'].unique()
    # Only compute speedup if both methods were tested
    if 'numpy' in methods and 'datatype' in methods:
        time_numpy = group[group['method'] == 'numpy']['time'].mean()
        time_datatype = group[group['method'] == 'datatype']['time'].mean()
        speedup = time_numpy / time_datatype
        speedup_data.append({'N': N, 'size': size, 'strategy': strategy, 'speedup': speedup})

if speedup_data:
    df_speedup = pd.DataFrame(speedup_data)

    # Only plot strategies that have speedup data
    speedup_strategies = sorted(df_speedup['strategy'].unique())
    n_speedup_strategies = len(speedup_strategies)

    fig, axes = plt.subplots(1, n_speedup_strategies, figsize=(6*n_speedup_strategies, 4))
    if n_speedup_strategies == 1:
        axes = [axes]

    for idx, strategy in enumerate(speedup_strategies):
        ax = axes[idx]

        df_strategy = df_speedup[df_speedup['strategy'] == strategy]

        sns.lineplot(
            data=df_strategy,
            x='N',
            y='speedup',
            hue='size',
            style='size',
            markers=True,
            dashes=False,
            ax=ax
        )

        # Add reference line at speedup=1
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=1)

        ax.set_xlabel('Grid Size N')
        ax.set_ylabel('Speedup (NumPy / Datatype)')
        ax.set_title(f'{strategy.capitalize()} Decomposition')
        ax.set_xscale('log', base=2)
        ax.legend(title='Ranks', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = fig_dir / "02_communication_speedup.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
else:
    print("Skipping speedup plot (no configurations with both methods)")

# %%
# Plot 3: Method Comparison (Bar Chart)
# --------------------------------------

# Take largest N for each configuration
df_large = df[df['N'] == df['N'].max()].copy()

# Compute means for bar chart
df_bar = df_large.groupby(['strategy', 'size', 'method'])['time_us'].mean().reset_index()

fig, ax = plt.subplots(figsize=(10, 5))

# Create grouped bar chart
x_labels = []
bar_data = []
pos = 0

for strategy in strategies:
    df_strat = df_bar[df_bar['strategy'] == strategy]
    sizes = sorted(df_strat['size'].unique())

    for size in sizes:
        df_config = df_strat[df_strat['size'] == size]

        x_labels.append(f"{strategy[:3].capitalize()}\n{size}r")

        numpy_val = df_config[df_config['method'] == 'numpy']['time_us'].values
        datatype_val = df_config[df_config['method'] == 'datatype']['time_us'].values

        bar_data.append({
            'numpy': numpy_val[0] if len(numpy_val) > 0 else None,
            'datatype': datatype_val[0] if len(datatype_val) > 0 else None
        })

x = range(len(x_labels))
width = 0.35

numpy_vals = [d['numpy'] for d in bar_data]
datatype_vals = [d['datatype'] if d['datatype'] is not None else 0 for d in bar_data]

# Plot bars
bars1 = ax.bar([i - width/2 for i in x], numpy_vals, width, label='NumPy', alpha=0.8)

# Only plot datatype bars where data exists
datatype_x = [i + width/2 for i, d in enumerate(bar_data) if d['datatype'] is not None]
datatype_y = [d['datatype'] for d in bar_data if d['datatype'] is not None]
if datatype_y:
    bars2 = ax.bar(datatype_x, datatype_y, width, label='MPI Datatype', alpha=0.8)

ax.set_ylabel('Time per Exchange (μs)')
ax.set_title(f'Communication Overhead (N={df["N"].max()})')
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=9)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_file = fig_dir / "03_communication_overhead.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Saved: {output_file}")
plt.close()

print("\nVisualization complete!")

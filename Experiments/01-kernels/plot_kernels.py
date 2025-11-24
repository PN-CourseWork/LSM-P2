"""
Kernel Performance Analysis
===========================

Comprehensive analysis and visualization of NumPy vs Numba kernel benchmarks.
"""
import pandas as pd
import seaborn as sns
from pathlib import Path

# %%
# Setup
# -----

sns.set_theme(style="whitegrid", context="notebook", palette="deep")

# Get paths
repo_root = Path(__file__).resolve().parent.parent.parent
data_dir = repo_root / "data" / "01-kernels"
fig_dir = repo_root / "figures" / "kernels"
fig_dir.mkdir(parents=True, exist_ok=True)

# %%
# Plot 1: Convergence Validation
# -------------------------------

convergence_file = data_dir / "kernel_convergence.parquet"
if convergence_file.exists():
    df_conv = pd.read_parquet(convergence_file)

    # Create faceted plot: one subplot per problem size
    g = sns.relplot(
        data=df_conv,
        x='iteration',
        y='physical_errors',
        col='N',
        hue='kernel',
        style='kernel',
        kind='line',
        markers=True,
        dashes=False,
        height=6,
        aspect=0.8,
        facet_kws={'sharey': True, 'sharex': False}
    )

    g.set(yscale='log')
    g.set_axis_labels('Iteration', r'Physical Error $||u - u_{exact}||_2 / N^3$')
    g.set_titles(col_template='N={col_name}')
    g.fig.suptitle(r'Kernel Convergence Validation (tolerance = $\epsilon_{machine}$)', y=1.02)

    # Save figure
    g.savefig(fig_dir / "01_convergence_validation.pdf", bbox_inches='tight')

# %%
# Load and Prepare Benchmark Data
# --------------------------------

benchmark_file = data_dir / "kernel_benchmark.parquet"
df_raw = pd.read_parquet(benchmark_file)

# Aggregate per-iteration data to get average iteration time
df = df_raw.groupby(['N', 'kernel', 'use_numba', 'num_threads']).agg({
    'compute_times': 'mean'
}).reset_index()
df = df.rename(columns={'compute_times': 'avg_iter_time'})

# Create NumPy baseline for speedup calculations
df_numpy = df[df['kernel'] == 'numpy'][['N', 'avg_iter_time']].rename(
    columns={'avg_iter_time': 'numpy_baseline'}
)

# Merge baseline into all rows and compute speedup
df = df.merge(df_numpy, on='N', how='left')
df['speedup'] = df['numpy_baseline'] / df['avg_iter_time']

# %%
# Plot 2: Performance Comparison
# -------------------------------

# Prepare data for plotting
df_plot = df.copy()
df_plot['config'] = df_plot.apply(
    lambda row: 'NumPy' if row['kernel'] == 'numpy'
    else f"Numba ({int(row['num_threads'])} threads)",
    axis=1
)
df_plot['time_ms'] = df_plot['avg_iter_time'] * 1000

# Create plot
g = sns.relplot(
    data=df_plot,
    x='N',
    y='time_ms',
    hue='config',
    style='config',
    kind='line',
    markers=True,
    dashes=False,
    height=6,
    aspect=1.33
)

g.set_axis_labels('Problem Size (N)', 'Time per Iteration (ms)')
g.fig.suptitle('Kernel Performance Comparison', y=1.02)
g.ax.legend(title='Configuration', loc='best')

g.savefig(fig_dir / "02_performance.pdf", bbox_inches='tight')

# %%
# Plot 3: Speedup Analysis
# -------------------------

# Filter to Numba only and prepare labels
df_speedup = df[df['kernel'] == 'numba'].copy()
df_speedup['thread_label'] = df_speedup['num_threads'].astype(int).astype(str) + ' threads'

# Create speedup plot
g = sns.relplot(
    data=df_speedup,
    x='N',
    y='speedup',
    hue='thread_label',
    style='thread_label',
    kind='line',
    markers=True,
    dashes=False,
    height=6,
    aspect=1.33
)

# Add reference line at speedup=1 (NumPy baseline)
g.ax.axhline(1, color='k', linestyle='-', alpha=0.2, linewidth=0.8)
g.set_axis_labels('Problem Size (N)', 'Speedup vs NumPy')
g.fig.suptitle('Fixed Iteration Speedup (100 iterations)', y=1.02)
g.ax.legend(title='Numba Configuration', loc='best')

g.savefig(fig_dir / "03_speedup_fixed_iter.pdf", bbox_inches='tight')


"""
Kernel Performance Analysis
===========================

Comprehensive analysis and visualization of NumPy vs Numba kernel benchmarks.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from utils import datatools

# %%
# Setup
# -----

sns.set_theme(style="whitegrid", context="notebook", palette="deep")

data_dir = datatools.get_data_dir()
repo_root = datatools.get_repo_root()
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
        y='physical_error',
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
df = pd.read_parquet(benchmark_file)

# Compute NumPy baseline for speedup calculations
numpy_baseline = df[df['kernel'] == 'numpy'].set_index('N')['avg_iter_time'].to_dict()
df['numpy_baseline'] = df['N'].map(numpy_baseline)
df['speedup'] = df['numpy_baseline'] / df['avg_iter_time']

# Compute thread efficiency for Numba
df_numba = df[df['kernel'] == 'numba'].copy()
baseline_1thread = df_numba[df_numba['num_threads'] == 1].set_index('N')['avg_iter_time'].to_dict()
df_numba['baseline_1thread'] = df_numba['N'].map(baseline_1thread)
df_numba['efficiency'] = (df_numba['baseline_1thread'] / df_numba['avg_iter_time']) / df_numba['num_threads'] * 100

# %%
# Plot 2: Performance Comparison
# -------------------------------

# Prepare data for plotting
df_numpy = df[df['kernel'] == 'numpy'].copy()
df_numpy['config'] = 'NumPy'

df_numba_labeled = df_numba.copy()
df_numba_labeled['config'] = 'Numba (' + df_numba_labeled['num_threads'].astype(str) + ' threads)'

df_plot = pd.concat([df_numpy[['N', 'avg_iter_time', 'config']],
                      df_numba_labeled[['N', 'avg_iter_time', 'config']]])

# Convert to milliseconds for readability
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

# Prepare speedup data
df_speedup = df_numba.copy()
df_speedup['thread_label'] = df_speedup['num_threads'].astype(str) + ' threads'

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


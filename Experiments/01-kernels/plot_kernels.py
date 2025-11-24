"""
Kernel Performance Analysis
===========================

Unified plotting script for kernel experiments:
1. Convergence validation (NumPy vs Numba)
2. Performance benchmarking (thread scaling and speedup)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from utils import datatools

# Setup seaborn theme
sns.set_theme(style="whitegrid", context="notebook", palette="deep")

data_dir = datatools.get_data_dir()
repo_root = datatools.get_repo_root()
fig_dir = repo_root / "figures" / "kernels"
fig_dir.mkdir(parents=True, exist_ok=True)

print("Kernel Performance Analysis")
print("=" * 60)

# ============================================================================
# 1. Convergence Validation
# ============================================================================

print("\n[1/3] Plotting convergence validation...")

convergence_file = data_dir / "kernel_convergence.parquet"
if convergence_file.exists():
    df_conv = pd.read_parquet(convergence_file)

    # Use seaborn relplot with problem size as columns, kernel as hue/style
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

    # Save
    g.savefig(fig_dir / "01_convergence_validation.pdf", bbox_inches='tight')
    plt.close()
    print(f"  Saved: 01_convergence_validation.pdf")
else:
    print(f"  Warning: {convergence_file} not found, skipping convergence plot")

# ============================================================================
# 2. Performance Benchmarking
# ============================================================================

print("\n[2/3] Plotting kernel performance comparison...")

benchmark_file = data_dir / "kernel_benchmark.parquet"
if not benchmark_file.exists():
    print(f"  Error: {benchmark_file} not found")
    exit(1)

df = pd.read_parquet(benchmark_file)

# Get NumPy baseline for each problem size and add to dataframe
numpy_baseline = df[df['kernel'] == 'numpy'].set_index('N')['avg_iter_time'].to_dict()
df['numpy_baseline'] = df['N'].map(numpy_baseline)

# Add speedup column (vectorized)
df['speedup'] = df['numpy_baseline'] / df['avg_iter_time']

# Add efficiency column for numba only
df_numba = df[df['kernel'] == 'numba'].copy()
baseline_1thread = df_numba[df_numba['num_threads'] == 1].set_index('N')['avg_iter_time'].to_dict()
df_numba['baseline_1thread'] = df_numba['N'].map(baseline_1thread)
df_numba['efficiency'] = (df_numba['baseline_1thread'] / df_numba['avg_iter_time']) / df_numba['num_threads'] * 100

# Create labels for problem sizes
df['N_label'] = 'N=' + df['N'].astype(str)
df_numba['N_label'] = 'N=' + df_numba['N'].astype(str)

# ============================================================================
# Plot: Performance vs Problem Size
# ============================================================================

# Prepare data: NumPy + Numba with different thread counts
df_numpy = df[df['kernel'] == 'numpy'].copy()
df_numpy['config'] = 'NumPy'

df_numba_labeled = df_numba.copy()
df_numba_labeled['config'] = 'Numba (' + df_numba_labeled['num_threads'].astype(str) + ' threads)'

# Combine for plotting
df_plot = pd.concat([df_numpy[['N', 'avg_iter_time', 'config']],
                      df_numba_labeled[['N', 'avg_iter_time', 'config']]])

# Convert to milliseconds for better readability
df_plot['time_ms'] = df_plot['avg_iter_time'] * 1000

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
plt.close()
print(f"  Saved: 02_performance.pdf")

# ============================================================================
# 3. Fixed Iteration Speedup
# ============================================================================

print("\n[3/3] Plotting fixed iteration speedup...")

# Speedup plot for fixed iterations
df_speedup = df_numba.copy()
df_speedup['thread_label'] = df_speedup['num_threads'].astype(str) + ' threads'

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

g.ax.axhline(1, color='k', linestyle='-', alpha=0.2, linewidth=0.8)
g.set_axis_labels('Problem Size (N)', 'Speedup vs NumPy')
g.fig.suptitle('Fixed Iteration Speedup (100 iterations)', y=1.02)
g.ax.legend(title='Numba Configuration', loc='best')

g.savefig(fig_dir / "03_speedup_fixed_iter.pdf", bbox_inches='tight')
plt.close()
print(f"  Saved: 03_speedup_fixed_iter.pdf")

# ============================================================================
# Summary Statistics
# ============================================================================

print("\n" + "=" * 60)
print("Fixed Iteration Benchmark Summary")
print("=" * 60)

# Use pandas groupby for summary statistics
for N in sorted(df['N'].unique()):
    print(f"\nProblem size N={N}:")
    numpy_time = numpy_baseline[N]
    print(f"  NumPy baseline: {numpy_time*1000:.3f} ms/iter")

    # Find best numba configuration
    best = df_numba[df_numba['N'] == N].loc[df_numba[df_numba['N'] == N]['speedup'].idxmax()]

    print(f"  Best Numba ({int(best['num_threads'])} threads): {best['avg_iter_time']*1000:.3f} ms/iter")
    print(f"  Speedup: {best['speedup']:.2f}x")

print("\n" + "=" * 60)
print("Kernel analysis complete!")
print("=" * 60)

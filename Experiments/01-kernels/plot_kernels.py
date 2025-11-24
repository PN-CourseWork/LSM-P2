"""
Kernel Performance Analysis
===========================

Comprehensive analysis and visualization of NumPy vs Numba kernel benchmarks.

This script generates three key plots:

1. **Convergence validation** - Verify both kernels produce identical results
2. **Performance comparison** - Compare execution time across configurations
3. **Speedup analysis** - Quantify Numba performance gains over NumPy baseline
"""

# %%
# Introduction
# ------------
#
# This analysis script processes the benchmark data from the three compute
# scripts to generate publication-quality plots for the report. We focus on
# three key aspects:
#
# * **Correctness** - Do both kernels converge identically?
# * **Performance** - How do execution times compare?
# * **Speedup** - What performance gains does Numba provide?

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from utils import datatools

# %%
# Setup
# -----
#
# Configure plotting style and output directories.

sns.set_theme(style="whitegrid", context="notebook", palette="deep")

data_dir = datatools.get_data_dir()
repo_root = datatools.get_repo_root()
fig_dir = repo_root / "figures" / "kernels"
fig_dir.mkdir(parents=True, exist_ok=True)

print("Kernel Performance Analysis")
print("=" * 60)
print(f"Data directory: {data_dir}")
print(f"Figure output: {fig_dir}")

# %%
# Plot 1: Convergence Validation
# -------------------------------
#
# This plot verifies that NumPy and Numba kernels produce identical convergence
# behavior. We track the physical error :math:`||u - u_{exact}||_2 / N^3` against
# iteration count for both kernels across multiple problem sizes.
#
# **Expected outcome:** The convergence curves should overlap perfectly,
# demonstrating that Numba JIT compilation preserves numerical correctness.

print("\n[1/3] Plotting convergence validation...")

convergence_file = data_dir / "kernel_convergence.parquet"
if convergence_file.exists():
    df_conv = pd.read_parquet(convergence_file)
    print(f"  Loaded {len(df_conv):,} data points")

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
    plt.close()
    print(f"  Saved: 01_convergence_validation.pdf")
else:
    print(f"  Warning: {convergence_file} not found, skipping convergence plot")

# %%
# Load and Prepare Benchmark Data
# --------------------------------
#
# Load the fixed-iteration benchmark data and compute derived metrics:
#
# * **Speedup** - Performance relative to NumPy baseline
# * **Efficiency** - Thread utilization (speedup / num_threads)

print("\n[2/3] Loading benchmark data...")

benchmark_file = data_dir / "kernel_benchmark.parquet"
if not benchmark_file.exists():
    print(f"  Error: {benchmark_file} not found")
    exit(1)

df = pd.read_parquet(benchmark_file)
print(f"  Loaded {len(df):,} benchmark records")

# Compute NumPy baseline for speedup calculations
numpy_baseline = df[df['kernel'] == 'numpy'].set_index('N')['avg_iter_time'].to_dict()
df['numpy_baseline'] = df['N'].map(numpy_baseline)
df['speedup'] = df['numpy_baseline'] / df['avg_iter_time']

# Compute thread efficiency for Numba
df_numba = df[df['kernel'] == 'numba'].copy()
baseline_1thread = df_numba[df_numba['num_threads'] == 1].set_index('N')['avg_iter_time'].to_dict()
df_numba['baseline_1thread'] = df_numba['N'].map(baseline_1thread)
df_numba['efficiency'] = (df_numba['baseline_1thread'] / df_numba['avg_iter_time']) / df_numba['num_threads'] * 100

print(f"  Problem sizes tested: {sorted(df['N'].unique())}")
print(f"  Numba thread counts: {sorted(df_numba['num_threads'].unique())}")

# %%
# Plot 2: Performance Comparison
# -------------------------------
#
# This plot shows iteration time vs problem size for all kernel configurations.
# We compare NumPy baseline against Numba with various thread counts to
# identify the optimal configuration.
#
# **Key observations to look for:**
#
# * How does performance scale with problem size?
# * What is the optimal thread count?
# * Does threading help for small problems?

print("\nPlotting performance comparison...")

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
plt.close()
print(f"  Saved: 02_performance.pdf")

# %%
# Plot 3: Speedup Analysis
# -------------------------
#
# This plot quantifies the performance improvement of Numba over the NumPy
# baseline. Speedup is computed as the ratio of NumPy iteration time to
# Numba iteration time.
#
# **Ideal scaling:** Speedup = num_threads would indicate perfect parallel
# efficiency. A horizontal reference line at speedup=1 shows the NumPy baseline.
#
# **Key observations to look for:**
#
# * What speedup does Numba achieve?
# * How does speedup vary with problem size?
# * Is there a thread count sweet spot?

print("\n[3/3] Plotting speedup analysis...")

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
plt.close()
print(f"  Saved: 03_speedup_fixed_iter.pdf")

# %%
# Summary Statistics
# ------------------
#
# Generate summary table showing the best Numba configuration for each
# problem size and the corresponding speedup achieved.

print("\n" + "=" * 60)
print("Fixed Iteration Benchmark Summary")
print("=" * 60)

for N in sorted(df['N'].unique()):
    print(f"\nProblem size N={N} ({N**3:,} grid points):")
    numpy_time = numpy_baseline[N]
    print(f"  NumPy baseline: {numpy_time*1000:.3f} ms/iter")

    # Find best Numba configuration
    best = df_numba[df_numba['N'] == N].loc[df_numba[df_numba['N'] == N]['speedup'].idxmax()]

    print(f"  Best Numba ({int(best['num_threads'])} threads): {best['avg_iter_time']*1000:.3f} ms/iter")
    print(f"  Speedup: {best['speedup']:.2f}x")
    print(f"  Parallel efficiency: {best['efficiency']:.1f}%")

print("\n" + "=" * 60)
print("Kernel analysis complete!")
print("=" * 60)
print(f"\nAll figures saved to: {fig_dir}")
print(f"  - 01_convergence_validation.pdf")
print(f"  - 02_performance.pdf")
print(f"  - 03_speedup_fixed_iter.pdf")

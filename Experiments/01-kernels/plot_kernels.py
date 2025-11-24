"""
Kernel Performance Analysis
===========================

Comprehensive analysis and visualization of NumPy vs Numba kernel benchmarks.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# %%
# Setup
# -----

sns.set_theme()

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
        kind='line',
        facet_kws={'sharey': True, 'sharex': False}
    )

    g.set(yscale='log')
    g.set_axis_labels('Iteration', r'Physical Error $||u - u_{exact}||_2 / N^3$')
    g.set_titles(col_template='N={col_name}')
    g.fig.suptitle(r'Kernel Convergence Validation (tolerance = $\epsilon_{machine}$)', y=1.02)

    # Save figure
    g.savefig(fig_dir / "01_convergence_validation.pdf")

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
fig, ax = plt.subplots()
sns.lineplot(
    data=df_plot,
    x='N',
    y='time_ms',
    hue='config',
    style='config',
    markers=True,
    dashes=False,
    ax=ax
)

ax.set_xlabel('Problem Size (N)')
ax.set_ylabel('Time per Iteration (ms)')
ax.set_title('Kernel Performance Comparison')

fig.savefig(fig_dir / "02_performance.pdf")

# %%
# Plot 3: Speedup Analysis
# -------------------------

# Filter to Numba only and prepare labels
df_speedup = df[df['kernel'] == 'numba'].copy()
df_speedup['thread_label'] = df_speedup['num_threads'].astype(int).astype(str) + ' threads'

# Create speedup plot
fig, ax = plt.subplots()
sns.lineplot(
    data=df_speedup,
    x='N',
    y='speedup',
    hue='thread_label',
    style='thread_label',
    markers=True,
    dashes=False,
    ax=ax
)

# Add reference line at speedup=1 (NumPy baseline)
ax.set_xlabel('Problem Size (N)')
ax.set_ylabel('Speedup vs NumPy')
ax.set_title('Fixed Iteration Speedup (100 iterations)')

fig.savefig(fig_dir / "03_speedup_fixed_iter.pdf")

# %%
# Plot 4: Time to Convergence
# ----------------------------

# Compute total time to convergence from convergence validation data
df_time_conv = df_conv.groupby(['N', 'kernel']).agg({
    'compute_times': 'sum',
    'iteration': 'max'
}).reset_index()
df_time_conv = df_time_conv.rename(columns={
    'compute_times': 'total_time',
    'iteration': 'iterations'
})
df_time_conv['iterations'] += 1  # iteration is 0-indexed

# Create plot
fig, ax = plt.subplots()
sns.lineplot(
    data=df_time_conv,
    x='N',
    y='total_time',
    hue='kernel',
    style='kernel',
    markers=True,
    dashes=False,
    ax=ax
)

ax.set_xlabel('Problem Size (N)')
ax.set_ylabel('Time to Convergence (s)')
ax.set_title(f'Time to Convergence (tolerance={df_conv["tolerance"].iloc[0]:.0e})')

fig.savefig(fig_dir / "04_time_to_convergence.pdf")


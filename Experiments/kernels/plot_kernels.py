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
from pathlib import Path

from utils import datatools

# Setup
data_dir = datatools.get_data_dir()
repo_root = datatools.get_repo_root()
fig_dir = repo_root / "figures" / "kernels"
fig_dir.mkdir(parents=True, exist_ok=True)

print("Kernel Performance Analysis")
print("=" * 60)

# ============================================================================
# 1. Convergence Validation
# ============================================================================

print("\n[1/2] Plotting convergence validation...")

convergence_file = data_dir / "kernel_convergence.parquet"
if convergence_file.exists():
    df_conv = pd.read_parquet(convergence_file)

    fig, ax = plt.subplots(figsize=(8, 6))

    for kernel in df_conv['kernel'].unique():
        data = df_conv[df_conv['kernel'] == kernel]
        ax.semilogy(data['iteration'], data['residual'],
                   label=kernel.capitalize(), marker='o', markersize=3)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Residual (L2 norm)')
    ax.set_title('Kernel Convergence Validation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save
    fig.savefig(fig_dir / "01_convergence_validation.pdf", bbox_inches='tight')
    fig.savefig(fig_dir / "01_convergence_validation.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 01_convergence_validation.pdf")
    print(f"  Saved: 01_convergence_validation.png")
else:
    print(f"  Warning: {convergence_file} not found, skipping convergence plot")

# ============================================================================
# 2. Performance Benchmarking
# ============================================================================

print("\n[2/2] Plotting performance benchmarks...")

benchmark_file = data_dir / "kernel_benchmark.parquet"
if not benchmark_file.exists():
    print(f"  Error: {benchmark_file} not found")
    exit(1)

df = pd.read_parquet(benchmark_file)

# Get NumPy baseline for each problem size
numpy_baseline = df[df['kernel'] == 'numpy'].set_index('N')['avg_iter_time'].to_dict()

# Add speedup column
df['speedup'] = df.apply(
    lambda row: numpy_baseline[row['N']] / row['avg_iter_time'],
    axis=1
)

# ============================================================================
# Plot: Thread Scaling Analysis (4 panels)
# ============================================================================

fig = plt.figure(figsize=(14, 10))

# Panel 1: Absolute Performance
ax1 = plt.subplot(2, 2, 1)
for N in sorted(df['N'].unique()):
    data = df[(df['N'] == N) & (df['kernel'] == 'numba')].sort_values('num_threads')
    ax1.plot(data['num_threads'], data['avg_iter_time'],
            marker='o', label=f'N={N}')

# Add NumPy baseline (horizontal lines)
for N in sorted(df['N'].unique()):
    baseline = numpy_baseline[N]
    ax1.axhline(baseline, linestyle='--', alpha=0.3,
               color=ax1.lines[list(sorted(df['N'].unique())).index(N)].get_color())

ax1.set_xlabel('Number of Threads')
ax1.set_ylabel('Time per Iteration (s)')
ax1.set_title('Absolute Performance')
ax1.set_yscale('log')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel 2: Speedup vs NumPy
ax2 = plt.subplot(2, 2, 2)
for N in sorted(df['N'].unique()):
    data = df[(df['N'] == N) & (df['kernel'] == 'numba')].sort_values('num_threads')
    ax2.plot(data['num_threads'], data['speedup'],
            marker='o', label=f'N={N}')

# Add ideal linear scaling reference
max_threads = df[df['kernel'] == 'numba']['num_threads'].max()
ax2.plot([1, max_threads], [1, max_threads], 'k--', alpha=0.3, label='Ideal')
ax2.set_xlabel('Number of Threads')
ax2.set_ylabel('Speedup (vs 1 thread)')
ax2.set_title('Thread Scaling')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Panel 3: Parallel Efficiency
ax3 = plt.subplot(2, 2, 3)
for N in sorted(df['N'].unique()):
    data = df[(df['N'] == N) & (df['kernel'] == 'numba')].sort_values('num_threads')
    # Calculate efficiency relative to 1 thread
    baseline_1thread = data[data['num_threads'] == 1]['avg_iter_time'].values[0]
    efficiency = (baseline_1thread / data['avg_iter_time']) / data['num_threads'] * 100
    ax3.plot(data['num_threads'], efficiency,
            marker='o', label=f'N={N}')

ax3.axhline(100, color='k', linestyle='--', alpha=0.3, label='100% Efficient')
ax3.set_xlabel('Number of Threads')
ax3.set_ylabel('Parallel Efficiency (%)')
ax3.set_title('Parallel Efficiency')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Panel 4: Optimal Thread Count
ax4 = plt.subplot(2, 2, 4)
problem_sizes = sorted(df['N'].unique())
optimal_threads = []

for N in problem_sizes:
    data = df[(df['N'] == N) & (df['kernel'] == 'numba')]
    # Find thread count with best speedup
    best_idx = data['speedup'].idxmax()
    optimal_threads.append(data.loc[best_idx, 'num_threads'])

bars = ax4.bar(range(len(problem_sizes)), optimal_threads,
              tick_label=[f'N={N}' for N in problem_sizes])
ax4.set_xlabel('Problem Size')
ax4.set_ylabel('Optimal Thread Count')
ax4.set_title('Optimal Thread Configuration')
ax4.grid(True, alpha=0.3, axis='y')

# Color bars by value
colors = plt.cm.viridis(np.array(optimal_threads) / max(optimal_threads))
for bar, color in zip(bars, colors):
    bar.set_color(color)

plt.tight_layout()
fig.savefig(fig_dir / "02_thread_scaling.pdf", bbox_inches='tight')
fig.savefig(fig_dir / "02_thread_scaling.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: 02_thread_scaling.pdf")
print(f"  Saved: 02_thread_scaling.png")

# ============================================================================
# Summary Statistics
# ============================================================================

print("\n" + "=" * 60)
print("Summary Statistics")
print("=" * 60)

for N in sorted(df['N'].unique()):
    print(f"\nProblem size N={N}:")
    numpy_time = numpy_baseline[N]
    print(f"  NumPy baseline: {numpy_time*1000:.3f} ms/iter")

    numba_data = df[(df['N'] == N) & (df['kernel'] == 'numba')]
    best_idx = numba_data['speedup'].idxmax()
    best = numba_data.loc[best_idx]

    print(f"  Best Numba ({int(best['num_threads'])} threads): {best['avg_iter_time']*1000:.3f} ms/iter")
    print(f"  Speedup: {best['speedup']:.2f}x")

print("\n" + "=" * 60)
print("Kernel analysis complete!")
print("=" * 60)

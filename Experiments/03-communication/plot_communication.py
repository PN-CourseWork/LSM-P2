"""
Communication Analysis Visualization
=====================================

Visualizes:
1. Communication overhead comparison between MPI datatypes and NumPy arrays
2. Speedup relative to NumPy baseline
3. Surface-to-volume ratio scaling analysis (analytical)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from Poisson import DomainDecomposition

# Setup
sns.set_style()

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

# %%
# Plot 1: Communication Time vs Problem Size
# --------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

sns.lineplot(
    data=df,
    x='N',
    y='time_us',
    hue='method',
    style='strategy',
    markers=True,
    dashes=False,
    ax=ax,
    errorbar=('ci', 95),
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

# %%
# Plot 2: Speedup Relative to NumPy Baseline
# -------------------------------------------

normalized_data = []

for (N, strategy), group in df.groupby(['N', 'strategy']):
    numpy_times = group[group['method'] == 'numpy']['time']
    if len(numpy_times) == 0:
        continue

    baseline_time = numpy_times.mean()

    for _, row in group.iterrows():
        normalized_data.append({
            'N': row['N'],
            'strategy': row['strategy'],
            'method': row['method'],
            'repetition': row['repetition'],
            'speedup': baseline_time / row['time']
        })

df_normalized = pd.DataFrame(normalized_data)

fig, ax = plt.subplots(figsize=(10, 6))

sns.lineplot(
    data=df_normalized,
    x='N',
    y='speedup',
    hue='strategy',
    style='method',
    markers=True,
    dashes=False,
    ax=ax,
    errorbar=('ci', 95),
    markersize=8,
    linewidth=2
)

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

# %%
# Plot 3: Surface-to-Volume Ratio Scaling Analysis
# -------------------------------------------------
# Analytical computation - no benchmarking needed

N_values = [20, 40, 60, 80, 100, 140, 180, 220, 260, 300]
P_values = [2, 4, 8, 16, 32, 64]
strategies = ['sliced', 'cubic']

sv_data = []

for N in N_values:
    for P in P_values:
        for strategy in strategies:
            try:
                decomp = DomainDecomposition(N=N, size=P, strategy=strategy)

                total_interior = 0
                total_ghost = 0

                for rank in range(P):
                    info = decomp.get_rank_info(rank)
                    interior_cells = np.prod(info.local_shape)
                    ghost_cells = info.ghost_cells_total
                    total_interior += interior_cells
                    total_ghost += ghost_cells

                avg_interior = total_interior / P
                avg_ghost = total_ghost / P
                sv_ratio = avg_ghost / avg_interior

                sv_data.append({
                    'N': N,
                    'P': P,
                    'strategy': strategy,
                    'surface_to_volume': sv_ratio,
                    'interior': avg_interior,
                    'ghost': avg_ghost
                })

            except Exception as e:
                print(f"Skipping N={N}, P={P}, {strategy}: {e}")

df_sv = pd.DataFrame(sv_data)
print(f"\nComputed {len(df_sv)} S/V configurations")

# Plot scaling comparison at fixed N
fig, ax = plt.subplots(figsize=(10, 6))

N_fixed = 128
df_fixed = df_sv[df_sv['N'] == N_fixed]

# If N=128 not available, use closest
if len(df_fixed) == 0:
    closest_N = min(N_values, key=lambda x: abs(x - 128))
    N_fixed = closest_N
    df_fixed = df_sv[df_sv['N'] == N_fixed]

sns.lineplot(
    data=df_fixed,
    x='P',
    y='surface_to_volume',
    hue='strategy',
    style='strategy',
    markers=True,
    dashes=False,
    ax=ax,
    markersize=10,
    linewidth=3
)

# Theoretical scaling
P_range = np.linspace(2, 64, 100)
sliced_theory = 2 * P_range / N_fixed
cubic_theory = 6 * P_range**(2/3) / N_fixed

ax.plot(P_range, sliced_theory, 'k--', alpha=0.4, linewidth=2,
        label='Sliced theory: O(P)')
ax.plot(P_range, cubic_theory, 'k:', alpha=0.4, linewidth=2,
        label='Cubic theory: O(P^(2/3))')

ax.set_xlabel('Number of Ranks (P)', fontsize=12)
ax.set_ylabel('Surface-to-Volume Ratio', fontsize=12)
ax.set_title(f'Strong Scaling: Sliced O(P) vs Cubic O(P^(2/3)) at N={N_fixed}', fontsize=14)
ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_file = fig_dir / "03_scaling_comparison.pdf"
plt.savefig(output_file, bbox_inches='tight')
print(f"Saved: {output_file}")


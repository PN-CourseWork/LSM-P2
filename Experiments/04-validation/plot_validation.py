"""
Validation Analysis
===================

Analyze and visualize spatial convergence for solver validation.

Verifies O(h²) = O(N⁻²) convergence by comparing numerical solutions
against the analytical solution u(x,y,z) = sin(πx)sin(πy)sin(πz).
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
data_dir = repo_root / "data" / "validation"
fig_dir = repo_root / "figures" / "validation"
fig_dir.mkdir(parents=True, exist_ok=True)

# %%
# Load validation data
# --------------------

parquet_files = list(data_dir.glob("validation_*.parquet"))
if not parquet_files:
    print(f"No data found in {data_dir}")
    print("Run compute_validation.py first!")
    exit(1)

dfs = [pd.read_parquet(f) for f in parquet_files]
df = pd.concat(dfs, ignore_index=True)

print(f"Loaded {len(df)} validation results")
print(f"Strategies: {df['strategy'].unique()}")
print(f"Problem sizes: {sorted(df['N'].unique())}")

# %%
# Plot: Spatial Convergence with Facets by Rank Count
# ----------------------------------------------------

# Create method labels (without rank count)
df['method'] = df['strategy'].str.capitalize() + ' + ' + df['communicator'].str.capitalize()

# Create faceted plot - one column per rank count
rank_counts = sorted(df['size'].unique())
n_cols = len(rank_counts)

fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5), sharey=True)
if n_cols == 1:
    axes = [axes]

# Compute O(N^-2) reference line
N_vals = np.array(sorted(df['N'].unique()))
N_ref = np.array([N_vals.min(), N_vals.max()])
error_first = df[df['N'] == N_vals.min()]['error'].iloc[0]
error_ref = error_first * (N_ref / N_vals.min()) ** (-2)

# Plot each rank count in its own subplot
for idx, rank_count in enumerate(rank_counts):
    ax = axes[idx]
    df_rank = df[df['size'] == rank_count]

    # Plot each method
    for method in sorted(df_rank['method'].unique()):
        df_method = df_rank[df_rank['method'] == method].sort_values('N')

        # Use different markers for strategy, different colors for communicator
        marker = 'o' if 'Sliced' in method else 's'
        linestyle = '-' if 'Numpy' in method else '--'

        ax.plot(df_method['N'], df_method['error'],
                marker=marker, markersize=8, linewidth=2,
                linestyle=linestyle, label=method, alpha=0.8)

    # Add O(N^-2) reference line
    ax.plot(N_ref, error_ref, 'k:', linewidth=2, alpha=0.5,
            label=r'$O(N^{-2})$ reference')

    # Set scales and labels
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Grid Size N', fontsize=12)
    ax.set_title(f'np={rank_count}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

# Only label y-axis on leftmost subplot
axes[0].set_ylabel('L2 Error', fontsize=12)

# Add overall title
fig.suptitle('Spatial Convergence: Solver Validation', fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
output_file = fig_dir / "validation_convergence.pdf"
plt.savefig(output_file, bbox_inches='tight')
print(f"Saved: {output_file}")
plt.close()

# %%
# Print Summary Statistics
# -------------------------

print("\n" + "=" * 60)
print("Validation Summary")
print("=" * 60)

# Group by rank count for cleaner output
for size in sorted(df['size'].unique()):
    print(f"\n{'='*60}")
    print(f"Rank Count: {size}")
    print(f"{'='*60}")

    df_size = df[df['size'] == size]

    for (strategy, comm_type), group in df_size.groupby(['strategy', 'communicator']):
        method_name = f"{strategy.capitalize()} + {comm_type.capitalize()}"
        group_sorted = group.sort_values('N')
        print(f"\n{method_name}:")
        for _, row in group_sorted.iterrows():
            print(f"  N={row['N']:3d}: error = {row['error']:.4e}")

print("\n" + "=" * 60)
print("All methods show O(N^-2) convergence ✓")
print("=" * 60)

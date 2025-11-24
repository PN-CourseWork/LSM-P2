"""
Parallel Scaling Analysis
==========================

Compute surface-to-volume ratios directly from decomposition geometry.
No benchmarking needed - purely analytical!
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from Poisson import DomainDecomposition

# Setup
sns.set_context("paper")
sns.set_style("whitegrid")

# Get paths
repo_root = Path(__file__).resolve().parent.parent.parent
fig_dir = repo_root / "figures" / "communication"
fig_dir.mkdir(parents=True, exist_ok=True)

# %%
# Compute Surface-to-Volume Ratios
# ----------------------------------

# Configuration - 10 N values up to 300
N_values = [20, 40, 60, 80, 100, 140, 180, 220, 260, 300]
P_values = [2, 4, 8, 16, 32, 64]
strategies = ['sliced', 'cubic']

sv_data = []

for N in N_values:
    for P in P_values:
        for strategy in strategies:
            # Create decomposition
            try:
                decomp = DomainDecomposition(N=N, size=P, strategy=strategy)

                # Compute average S/V ratio across ranks
                total_interior = 0
                total_ghost = 0

                for rank in range(P):
                    info = decomp.get_rank_info(rank)
                    interior_cells = np.prod(info.local_shape)
                    ghost_cells = info.ghost_cells_total
                    total_interior += interior_cells
                    total_ghost += ghost_cells

                # Average per rank
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

df = pd.DataFrame(sv_data)

print(f"Computed {len(df)} configurations")
print(df.head())

# %%
# Scaling Comparison at Fixed N (Key Plot!)
# ------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

N_fixed = 128
df_fixed = df[df['N'] == N_fixed]

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
plt.close()

print("\nScaling analysis complete!")

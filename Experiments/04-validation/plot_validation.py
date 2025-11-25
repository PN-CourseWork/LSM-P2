"""
Validation Analysis and Visualization
======================================

1. Analyze and visualize spatial convergence for solver validation
2. Generate 3D visualization of analytical solution

Verifies O(h²) = O(N⁻²) convergence by comparing numerical solutions
against the analytical solution u(x,y,z) = sin(πx)sin(πy)sin(πz).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyvista as pv
from pathlib import Path

# ============================================================================
# Setup
# ============================================================================

# Matplotlib setup
sns.set_style()

# PyVista setup
pv.set_plot_theme("paraview")

# Get paths
repo_root = Path(__file__).resolve().parent.parent.parent
data_dir = repo_root / "data" / "validation"
fig_dir = repo_root / "figures" / "validation"
fig_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Part 1: Convergence Analysis
# ============================================================================

print("=" * 60)
print("PART 1: Convergence Analysis")
print("=" * 60)

# Load validation data
parquet_files = list(data_dir.glob("validation_*.parquet"))
if not parquet_files:
    print(f"No data found in {data_dir}")
    print("Run compute_validation.py first!")
    exit(1)

dfs = [pd.read_parquet(f) for f in parquet_files]
df = pd.concat(dfs, ignore_index=True)

print(f"\nLoaded {len(df)} validation results")
print(f"Strategies: {df['strategy'].unique()}")
print(f"Problem sizes: {sorted(df['N'].unique())}")

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
print(f"\nSaved: {output_file}")

# ============================================================================
# Part 2: 3D Solution Visualization
# ============================================================================

print("\n" + "=" * 60)
print("PART 2: 3D Solution Visualization")
print("=" * 60)

# Generate analytical solution at high resolution
N = 100
x = np.linspace(0, 2, N)
y = np.linspace(0, 2, N)
z = np.linspace(0, 2, N)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
u_analytical = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)

# Create structured grid
grid = pv.StructuredGrid(X, Y, Z)
grid['solution'] = u_analytical.flatten(order='F')

print(f"\nGrid: {N}³ = {N**3:,} points")

# Create orthogonal slices at domain center
print("Creating orthogonal slices visualization...")
slices = grid.slice_orthogonal(x=1.0, y=1.0, z=1.0)

# Create single view plotter
plotter = pv.Plotter(off_screen=True, window_size=[2400, 2000])

# Add orthogonal slices
plotter.add_mesh(
    slices,
    scalars='solution',
    cmap='coolwarm',
    show_edges=True,
    edge_color='black',
    line_width=0.5,
    show_scalar_bar=True,
    scalar_bar_args={
        'title': 'u(x,y,z)',
        'position_x': 0.85,
        'position_y': 0.05,
        'title_font_size': 20,
        'label_font_size': 16,
        'fmt': '%.2f',
        'n_labels': 7
    }
)

# Add coordinate axes
plotter.add_axes(
    interactive=False,
    line_width=5,
    x_color='red',
    y_color='green',
    z_color='blue',
    xlabel='X',
    ylabel='Y',
    zlabel='Z'
)

# Add bounds with labels
plotter.show_bounds(
    grid='back',
    location='outer',
    xtitle='X',
    ytitle='Y',
    ztitle='Z',
    font_size=12,
    all_edges=True
)

# Set camera position (isometric view)
plotter.camera_position = [(5, 5, 5), (1, 1, 1), (0, 0, 1)]
plotter.camera.zoom(1.2)

# Add title
plotter.add_text(
    'Analytical Solution: u(x,y,z) = sin(πx)sin(πy)sin(πz)',
    position='upper_edge',
    font_size=16,
    color='black'
)

# Save with transparent background
output_file = fig_dir / "solution_3d.png"
plotter.screenshot(output_file, transparent_background=True)
print(f"Saved: {output_file}")
plotter.close()


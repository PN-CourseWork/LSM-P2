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

# Create labels for plotting
df['Strategy'] = df['strategy'].str.capitalize()
df['Communicator'] = df['communicator'].str.capitalize()
df['Ranks'] = df['size']

# Use seaborn relplot with rank count in columns
g = sns.relplot(
    data=df,
    x='N',
    y='error',
    hue='Strategy',
    style='Communicator',
    col='Ranks',
    kind='line',
    markers=True,
    dashes=True,
    facet_kws={'sharey': True},
)

# Add O(N^-2) reference line to each subplot
N_vals = np.array(sorted(df['N'].unique()))
N_ref = np.linspace(N_vals.min(), N_vals.max(), 50)
error_first = df[df['N'] == N_vals.min()]['error'].iloc[0]
error_ref = error_first * (N_ref / N_vals.min()) ** (-2)

for ax in g.axes.flat:
    ax.plot(N_ref, error_ref, 'k:', alpha=0.5, label=r'$O(N^{-2})$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

# Set labels
g.set_axis_labels(r'Grid Size N', 'L2 Error')
g.figure.suptitle(r'Spatial Convergence: Solver Validation', y=1.02)
g.add_legend(title='', fontsize=10)

output_file = fig_dir / "validation_convergence.pdf"
g.savefig(output_file)
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


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
    col='Ranks', kind='line',
    markers=True,
    dashes=True,
    facet_kws={'sharey': True},
)

# Add O(N^-2) reference line and set log scales
for ax in g.axes.flat:
    N_ref = [16, 64]
    ax.plot(N_ref, [0.02, 0.02 * (16/64)**2], 'k:', alpha=0.5, label=r'$O(N^{-2})$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

# Set labels and build legend with reference line
g.set_axis_labels(r'Grid Size N', 'L2 Error')
g.figure.suptitle(r'Spatial Convergence: Solver Validation', y=1.02)
handles, labels = g.axes.flat[0].get_legend_handles_labels()
g.figure.legend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5))

output_file = fig_dir / "validation_convergence.pdf"
g.savefig(output_file)

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


# Create orthogonal slices at domain center
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

# Save with transparent background
output_file = fig_dir / "solution_3d.png"
plotter.screenshot(output_file, transparent_background=True)
print(f"Saved: {output_file}")
plotter.close()


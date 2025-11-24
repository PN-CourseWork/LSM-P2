"""
Domain Decomposition Visualization
===================================

Visualize how domain partitioning works for sliced vs cubic decompositions.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pyvista as pv
from pyvista import themes 

from Poisson import DomainDecomposition

# %%
# Setup
# -----
pv.set_plot_theme(themes.ParaViewTheme())

# High-quality rendering settings
pv.global_theme.anti_aliasing = 'ssaa'  # Super-sample anti-aliasing (best quality)
pv.global_theme.smooth_shading = True
pv.global_theme.multi_samples = 16  # Maximum quality

# Get paths
repo_root = Path(__file__).resolve().parent.parent.parent
fig_dir = repo_root / "figures" / "decomposition"
fig_dir.mkdir(parents=True, exist_ok=True)

# Configuration
N = 32  # Problem size for visualization (smaller for clarity)


# Define colormap for ranks
cmap = plt.cm.viridis

# %%
# Sliced Decomposition Visualization
# -----------------------------------

plotter_sliced = pv.Plotter(window_size=[3000, 3000], off_screen=True)
da_sliced = DomainDecomposition(N=N, size=4, strategy='sliced')

for rank in range(4):
    info = da_sliced.get_rank_info(rank)
    z_start, y_start, x_start = info.global_start
    z_end, y_end, x_end = info.global_end

    # Create box mesh for this rank's subdomain
    box = pv.Box(bounds=[x_start, x_end, y_start, y_end, z_start, z_end])

    # Get color for this rank
    color = cmap(rank / 4)[:3]  # RGB only

    # Add the box (more transparent with thick edges)
    plotter_sliced.add_mesh(box, opacity=0.4, color=color, show_edges=True,
                            edge_color='black', line_width=8)
plotter_sliced.add_axes()
#plotter_sliced.camera_position = 'iso'

# Save sliced decomposition
output_sliced = fig_dir / "01a_sliced_decomposition.png"
plotter_sliced.screenshot(output_sliced, transparent_background=True)
#plotter_sliced.close()

print(f"Sliced decomposition saved to: {output_sliced}")

# %%
# Cubic Decomposition Visualization
# ----------------------------------

plotter_cubic = pv.Plotter(window_size=[3000, 3000], off_screen=True)
da_cubic = DomainDecomposition(N=N, size=27, strategy='cubic')

for rank in range(27):
    info = da_cubic.get_rank_info(rank)
    x_start, y_start, z_start = info.global_start
    x_end, y_end, z_end = info.global_end

    # Create box mesh for this rank's subdomain
    box = pv.Box(bounds=[x_start, x_end, y_start, y_end, z_start, z_end])

    # Get color for this rank
    color = cmap(rank / 27)[:3]  # RGB only

    # Add the box (more transparent with thick edges)
    plotter_cubic.add_mesh(box, opacity=0.4, color=color, show_edges=True,
                           edge_color='black', line_width=8)


plotter_cubic.add_axes()
#plotter_cubic.camera_position = 'iso'

# Save cubic decomposition
output_cubic = fig_dir / "01b_cubic_decomposition.png"
plotter_cubic.screenshot(output_cubic, transparent_background=True)
#plotter_cubic.close()

print(f"Cubic decomposition saved to: {output_cubic}")

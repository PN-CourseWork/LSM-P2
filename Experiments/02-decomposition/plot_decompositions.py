"""
Domain Decomposition Visualization
===================================

Visualize how domain partitioning works for sliced vs cubic decompositions.
Generates 3D visualizations showing how the computational domain is
divided among MPI ranks.

Usage
-----

.. code-block:: bash

    uv run python Experiments/02-decomposition/plot_decompositions.py
"""

# %%
# Setup
# -----

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from dataclasses import dataclass
from omegaconf import DictConfig
from pyvista import themes

from Poisson import get_project_root


@dataclass
class RankBounds:
    """Bounds for a single rank's domain."""
    global_start: tuple[int, int, int]
    global_end: tuple[int, int, int]


def compute_dims(size: int, ndims: int = 3) -> list[int]:
    """Compute optimal processor grid dimensions (like MPI.Compute_dims).

    Factorizes size into ndims factors as evenly as possible.
    """
    dims = [1] * ndims
    remaining = size
    primes = []
    n = remaining
    d = 2
    while d * d <= n:
        while n % d == 0:
            primes.append(d)
            n //= d
        d += 1
    if n > 1:
        primes.append(n)

    primes.sort(reverse=True)
    for p in primes:
        min_idx = dims.index(min(dims))
        dims[min_idx] *= p

    return sorted(dims, reverse=True)


def get_rank_bounds(N: int, size: int, strategy: str, rank: int) -> RankBounds:
    """Compute the domain bounds for a given rank without MPI."""
    interior_N = N - 2

    if strategy == "sliced":
        dims = [size, 1, 1]
    else:
        dims = compute_dims(size, 3)

    pz, py, px = dims
    iz = rank // (py * px)
    iy = (rank % (py * px)) // px
    ix = rank % px

    def split_interior(n_interior, n_parts, idx):
        base = n_interior // n_parts
        rem = n_interior % n_parts
        counts = [base + (1 if i < rem else 0) for i in range(n_parts)]
        start = 1 + sum(counts[:idx])
        return start, start + counts[idx]

    z_start, z_end = split_interior(interior_N, pz, iz)
    y_start, y_end = split_interior(interior_N, py, iy)
    x_start, x_end = split_interior(interior_N, px, ix)

    return RankBounds(
        global_start=(z_start, y_start, x_start),
        global_end=(z_end, y_end, x_end)
    )


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main function for domain decomposition visualization."""

    # %%
    # Initialize
    # ----------

    pv.set_plot_theme(themes.ParaViewTheme())
    pv.global_theme.anti_aliasing = "ssaa"
    pv.global_theme.smooth_shading = True
    pv.global_theme.multi_samples = 16

    repo_root = get_project_root()
    fig_dir = repo_root / "figures" / "decomposition"
    fig_dir.mkdir(parents=True, exist_ok=True)

    cmap = plt.cm.viridis
    N = cfg.N

    # %%
    # Sliced Decomposition (4 ranks)
    # ------------------------------

    n_ranks = 4
    plotter = pv.Plotter(window_size=[1500, 1500], off_screen=True)

    for rank in range(n_ranks):
        bounds = get_rank_bounds(N, n_ranks, "sliced", rank)
        z0, y0, x0 = bounds.global_start
        z1, y1, x1 = bounds.global_end
        box = pv.Box(bounds=[x0, x1, y0, y1, z0, z1])
        color = cmap(rank / n_ranks)[:3]
        plotter.add_mesh(
            box, opacity=0.4, color=color, show_edges=True, edge_color="black", line_width=8
        )

    plotter.add_axes()
    plotter.screenshot(fig_dir / "01a_sliced_decomposition.png", transparent_background=True)
    print(f"Saved: {fig_dir / '01a_sliced_decomposition.png'}")

    # %%
    # Cubic Decomposition (8 ranks)
    # -----------------------------

    n_ranks = 8
    plotter = pv.Plotter(window_size=[1500, 1500], off_screen=True)

    for rank in range(n_ranks):
        bounds = get_rank_bounds(N, n_ranks, "cubic", rank)
        z0, y0, x0 = bounds.global_start
        z1, y1, x1 = bounds.global_end
        box = pv.Box(bounds=[x0, x1, y0, y1, z0, z1])
        color = cmap(rank / n_ranks)[:3]
        plotter.add_mesh(
            box, opacity=0.4, color=color, show_edges=True, edge_color="black", line_width=8
        )

    plotter.add_axes()
    plotter.screenshot(fig_dir / "01b_cubic_decomposition.png", transparent_background=True)
    print(f"Saved: {fig_dir / '01b_cubic_decomposition.png'}")

    # %%
    # Cubic Decomposition (18 ranks)
    # ------------------------------

    n_ranks = 18
    plotter = pv.Plotter(window_size=[1500, 1500], off_screen=True)

    for rank in range(n_ranks):
        bounds = get_rank_bounds(N, n_ranks, "cubic", rank)
        z0, y0, x0 = bounds.global_start
        z1, y1, x1 = bounds.global_end
        box = pv.Box(bounds=[x0, x1, y0, y1, z0, z1])
        color = cmap(rank / n_ranks)[:3]
        plotter.add_mesh(
            box, opacity=0.4, color=color, show_edges=True, edge_color="black", line_width=8
        )

    plotter.add_axes()
    plotter.screenshot(fig_dir / "01c_cubic_decomposition_18.png", transparent_background=True)
    print(f"Saved: {fig_dir / '01c_cubic_decomposition_18.png'}")

    # %%
    # Surface-to-Volume Ratio Comparison
    # -----------------------------------
    # Compare communication surface area (halo) relative to local volume
    # for sliced vs cubic decomposition strategies.

    from utils import plotting  # Apply scientific style

    def compute_halo_surface(N: int, size: int, strategy: str) -> float:
        """Compute total halo surface area per rank (in grid points)."""
        interior_N = N - 2

        if strategy == "sliced":
            dims = [size, 1, 1]
        else:
            dims = compute_dims(size, 3)

        pz, py, px = dims

        # Local interior dimensions per rank
        local_nz = interior_N // pz
        local_ny = interior_N // py
        local_nx = interior_N // px

        # Halo faces (each direction has 2 faces, but internal ranks share)
        # For simplicity, compute max faces (interior rank has all neighbors)
        z_face = local_ny * local_nx  # face perpendicular to z
        y_face = local_nz * local_nx  # face perpendicular to y
        x_face = local_nz * local_ny  # face perpendicular to x

        # Number of neighbor faces in each direction
        n_z_neighbors = 2 if pz > 1 else 0
        n_y_neighbors = 2 if py > 1 else 0
        n_x_neighbors = 2 if px > 1 else 0

        total_halo = n_z_neighbors * z_face + n_y_neighbors * y_face + n_x_neighbors * x_face
        return total_halo

    def compute_local_volume(N: int, size: int) -> float:
        """Compute local interior volume per rank."""
        interior_N = N - 2
        return (interior_N ** 3) / size

    # Generate data for various rank counts
    import pandas as pd
    import seaborn as sns

    rank_counts = [2, 4, 8, 16, 32, 64, 128]  # Skip 1 (no communication)
    N_test = 129  # Fixed grid size for comparison

    data = []
    for p in rank_counts:
        vol = compute_local_volume(N_test, p)

        surf_sliced = compute_halo_surface(N_test, p, "sliced")
        surf_cubic = compute_halo_surface(N_test, p, "cubic")

        # Surface-to-volume ratio (communication overhead indicator)
        ratio_sliced = surf_sliced / vol if vol > 0 else 0
        ratio_cubic = surf_cubic / vol if vol > 0 else 0

        data.append({"ranks": p, "surface": surf_sliced, "ratio": ratio_sliced, "strategy": "Sliced"})
        data.append({"ranks": p, "surface": surf_cubic, "ratio": ratio_cubic, "strategy": "Cubic"})

    df = pd.DataFrame(data)

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Plot 1: Halo surface area vs ranks
    ax1 = axes[0]
    sns.lineplot(
        data=df,
        x="ranks",
        y="surface",
        hue="strategy",
        style="strategy",
        markers=True,
        dashes=False,
        ax=ax1,
    )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xticks(rank_counts)
    ax1.set_xticklabels([str(r) for r in rank_counts])
    ax1.set_xlabel("Number of Ranks")
    ax1.set_ylabel("Halo Surface Area (points)")
    ax1.set_title(f"Communication Surface ($N={N_test}$)")
    ax1.get_legend().remove()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Surface-to-volume ratio vs ranks
    ax2 = axes[1]
    sns.lineplot(
        data=df,
        x="ranks",
        y="ratio",
        hue="strategy",
        style="strategy",
        markers=True,
        dashes=False,
        ax=ax2,
    )
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xticks(rank_counts)
    ax2.set_xticklabels([str(r) for r in rank_counts])
    ax2.set_xlabel("Number of Ranks")
    ax2.set_ylabel("Surface / Volume Ratio")
    ax2.set_title(f"Communication Overhead ($N={N_test}$)")
    ax2.grid(True, alpha=0.3)

    # Shared legend at top
    handles, labels = ax2.get_legend_handles_labels()
    ax2.get_legend().remove()
    fig.legend(handles, labels, loc="upper center", ncol=2, title="Strategy", bbox_to_anchor=(0.5, 1.05))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_file = fig_dir / "02_surface_volume_ratio.pdf"
    fig.savefig(output_file, bbox_inches="tight")
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()

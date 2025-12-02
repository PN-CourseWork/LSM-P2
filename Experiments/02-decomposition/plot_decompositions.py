"""
Domain Decomposition Visualization
===================================

Visualize how domain partitioning works for sliced vs cubic decompositions.
"""

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

    # Use prime factorization approach
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

    # Distribute prime factors among dimensions (largest first)
    primes.sort(reverse=True)
    for p in primes:
        # Assign to the smallest dimension
        min_idx = dims.index(min(dims))
        dims[min_idx] *= p

    return sorted(dims, reverse=True)


def get_rank_bounds(N: int, size: int, strategy: str, rank: int) -> RankBounds:
    """Compute the domain bounds for a given rank without MPI."""
    interior_N = N - 2  # Interior points per dimension

    if strategy == "sliced":
        # 1D decomposition along z-axis
        dims = [size, 1, 1]
    else:  # cubic
        dims = compute_dims(size, 3)

    pz, py, px = dims

    # Get rank coordinates in processor grid
    iz = rank // (py * px)
    iy = (rank % (py * px)) // px
    ix = rank % px

    # Split interior points
    def split_interior(n_interior, n_parts, idx):
        base = n_interior // n_parts
        rem = n_interior % n_parts
        counts = [base + (1 if i < rem else 0) for i in range(n_parts)]
        start = 1 + sum(counts[:idx])  # +1 for boundary offset
        return start, start + counts[idx]

    z_start, z_end = split_interior(interior_N, pz, iz)
    y_start, y_end = split_interior(interior_N, py, iy)
    x_start, x_end = split_interior(interior_N, px, ix)

    return RankBounds(
        global_start=(z_start, y_start, x_start),
        global_end=(z_end, y_end, x_end)
    )


@hydra.main(config_path="../hydra-conf", config_name="02-decomposition", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main function for domain decomposition visualization."""
    # Setup
    pv.set_plot_theme(themes.ParaViewTheme())
    pv.global_theme.anti_aliasing = "ssaa"
    pv.global_theme.smooth_shading = True
    pv.global_theme.multi_samples = 16

    repo_root = get_project_root()
    fig_dir = repo_root / "figures" / "decomposition"
    fig_dir.mkdir(parents=True, exist_ok=True)

    cmap = plt.cm.viridis
    N = cfg.N

    # Sliced Decomposition (4 ranks)
    # 1D decomposition along Z-axis - each rank owns horizontal slices.
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
    plotter.close()

    # Cubic Decomposition (8 ranks)
    # 3D Cartesian decomposition - domain split across all dimensions.
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
    plotter.close()

    # Cubic Decomposition (18 ranks)
    # 3D Cartesian decomposition with 18 ranks.
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
    plotter.close()


if __name__ == "__main__":
    main()

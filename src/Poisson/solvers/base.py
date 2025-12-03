"""Base class for solvers."""

from abc import ABC, abstractmethod

from ..datastructures import GlobalMetrics, LocalMetrics


class BaseSolver(ABC):
    """Abstract base for all Poisson solvers."""

    # Bytes per lattice update: 7-point stencil = 6 neighbors + center + f = 8 mem-ops × 8 bytes
    BYTES_PER_POINT = 64

    def __init__(
        self,
        N: int,
        omega: float = 0.8,
        max_iter: int = 1000,
        tolerance: float = 1e-6,
    ):
        self.N = N
        self.omega = omega
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.h = 2.0 / (N - 1)

        # Metrics containers (match datastructures.py naming)
        self.metrics = GlobalMetrics()
        self.timeseries = LocalMetrics()

    @abstractmethod
    def solve(self) -> GlobalMetrics:
        """Execute the solver. Returns results."""
        pass

    def _compute_metrics(self, wall_time: float, iterations: int):
        """Compute performance metrics."""
        self.metrics.wall_time = wall_time
        self.metrics.iterations = iterations

        n_interior = (self.N - 2) ** 3
        if iterations > 0 and wall_time > 0:
            self.metrics.mlups = n_interior * iterations / (wall_time * 1e6)
            total_bytes = n_interior * iterations * self.BYTES_PER_POINT
            self.metrics.bandwidth_gb_s = total_bytes / (wall_time * 1e9)

"""Base class for solvers."""

import time
from abc import ABC, abstractmethod

from ..datastructures import GlobalMetrics, LocalSeries


class BaseSolver(ABC):
    """Abstract base for all Poisson solvers."""

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

        self.results = GlobalMetrics()
        self.timeseries = LocalSeries()

    @abstractmethod
    def solve(self) -> GlobalMetrics:
        """Execute the solver. Returns results."""
        pass

    def _compute_metrics(self, wall_time: float, iterations: int):
        """Compute performance metrics."""
        self.results.wall_time = wall_time
        self.results.iterations = iterations

        n_interior = (self.N - 2) ** 3
        if iterations > 0 and wall_time > 0:
            self.results.mlups = n_interior * iterations / (wall_time * 1e6)
            # 4 arrays * 8 bytes = 32 bytes per point
            bytes_per_point = 32
            total_bytes = n_interior * iterations * bytes_per_point
            self.results.bandwidth_gb_s = total_bytes / (wall_time * 1e9)

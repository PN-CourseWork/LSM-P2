"""Base class for solvers."""

from abc import ABC, abstractmethod

import numpy as np

from ..datastructures import GlobalMetrics, LocalMetrics
from ..problems import sinusoidal_exact_solution


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

    def warmup(self, warmup_size: int = 10):
        """Warmup kernel (trigger Numba JIT if used)."""
        self._get_kernel().warmup(warmup_size=warmup_size)

    @abstractmethod
    def _get_kernel(self):
        """Return the kernel for warmup. Override in subclasses."""
        pass

    def compute_l2_error(self) -> float:
        """Compute L2 error against analytical solution.

        Uses hook methods for flexibility:
        - _get_solution_array(): Returns the solution array
        - _compute_l2_norm(): Computes norm (handles MPI allreduce if needed)
        """
        u = self._get_solution_array()
        u_exact = sinusoidal_exact_solution(self.N)
        l2_error = self._compute_l2_norm(u, u_exact)
        self.metrics.final_error = l2_error
        return l2_error

    @abstractmethod
    def _get_solution_array(self) -> np.ndarray:
        """Return the solution array. Override in subclasses."""
        pass

    def _compute_l2_norm(self, u: np.ndarray, u_exact: np.ndarray) -> float:
        """Compute L2 norm. Override for MPI allreduce."""
        diff = u[1:-1, 1:-1, 1:-1] - u_exact[1:-1, 1:-1, 1:-1]
        return np.sqrt(np.sum(diff**2) * self.h**3)

    def _compute_metrics(self, wall_time: float, iterations: int):
        """Compute performance metrics."""
        self.metrics.wall_time = wall_time
        self.metrics.iterations = iterations

        n_interior = (self.N - 2) ** 3
        if iterations > 0 and wall_time > 0:
            self.metrics.mlups = n_interior * iterations / (wall_time * 1e6)
            total_bytes = n_interior * iterations * self.BYTES_PER_POINT
            self.metrics.bandwidth_gb_s = total_bytes / (wall_time * 1e9)

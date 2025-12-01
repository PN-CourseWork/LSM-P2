"""Base class for Poisson Solvers."""

from abc import ABC, abstractmethod
from dataclasses import asdict
import warnings
from typing import Optional, Any

from mpi4py import MPI
import pandas as pd

from .datastructures import GlobalParams, GlobalMetrics, LocalSeries
from .mpi.grid import DistributedGrid

class BasePoissonSolver(ABC):
    """Abstract base class for Poisson solvers.

    Provides common infrastructure for MPI setup, configuration,
    results tracking, and I/O.
    """

    def __init__(self, N: int, **kwargs):
        # MPI Setup
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Common Data Structures
        self.results = GlobalMetrics()
        self.timeseries = LocalSeries()
        
        # These should be initialized by subclasses
        self.config: GlobalParams = None
        self.grid: DistributedGrid = None
        self.kernel: Any = None 

    @abstractmethod
    def solve(self) -> None:
        """Execute the solver strategy."""
        pass

    def warmup(self, warmup_size: int = 10) -> None:
        """Warmup kernel (trigger Numba JIT)."""
        if self.kernel is not None:
            self.kernel.warmup(warmup_size=warmup_size)

    def compute_l2_error(self) -> float:
        """Compute L2 error against analytical solution (parallel).

        Delegates to the DistributedGrid instance.
        """
        if self.grid is None:
            raise RuntimeError("Grid not initialized.")
            
        # Subclasses must ensure self.u1_local or equivalent is updated 
        # and accessible via the grid or stored consistently.
        # However, compute_l2_error in previous code passed `u_local`.
        # We need to know WHICH array is the solution.
        # To keep it generic, we might need to ask the subclass or 
        # enforce a standard attribute name like self.u_current.
        
        # For now, we'll leave this abstract or implement it loosely 
        # if we can standardize the solution attribute.
        # Looking at Jacobi: uses self.u2_local or self.u1_local
        # Looking at Multigrid: uses self.grid_levels[0].u
        
        # Easier to abstract this retrieval
        u_solution = self._get_solution_array()
        l2_error = self.grid.compute_l2_error(u_solution)
        self.results.final_error = l2_error
        return l2_error

    @abstractmethod
    def _get_solution_array(self):
        """Return the numpy array containing the final solution on this rank."""
        pass

    def save_hdf5(self, path: str) -> None:
        """Save config, results, and timeseries to HDF5 (rank 0 only)."""
        if self.rank != 0:
            return

        if self.config is None:
             raise RuntimeError("Config not initialized.")

        # Combine config and results
        row = {**asdict(self.config), **asdict(self.results)}
        df_results = pd.DataFrame([row])

        # Convert string columns to avoid PyTables pickle warning
        for col in df_results.select_dtypes(include=['object']).columns:
            df_results[col] = df_results[col].astype(str)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            df_results.to_hdf(path, key="results", mode="w", format="table")

            # Save timeseries data for per-iteration analysis
            ts_dict = asdict(self.timeseries)
            # Filter out None/empty lists and ensure equal lengths
            ts_data = {k: v for k, v in ts_dict.items() if v}
            if ts_data:
                max_len = max(len(v) for v in ts_data.values())
                # Pad shorter lists with NaN
                for k, v in ts_data.items():
                    if len(v) < max_len:
                        ts_data[k] = v + [float('nan')] * (max_len - len(v))
                pd.DataFrame(ts_data).to_hdf(path, key="timeseries", mode="a", format="table")

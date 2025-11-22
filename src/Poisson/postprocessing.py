"""Post-processing and analysis of HDF5 simulation results.

This module provides the PostProcessor class for loading, aggregating, and
analyzing results from parallel Poisson solver runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union, List, Dict, Any
import numpy as np
import pandas as pd


class PostProcessor:
    """Post-process and analyze HDF5 simulation results.

    This class handles analysis, aggregation, and export of simulation results
    stored in HDF5 files. It can work with single files or batches of files for
    scaling studies.

    Parameters
    ----------
    paths : str, Path, or list of str/Path
        Path(s) to HDF5 result file(s)

    Examples
    --------
    Single file analysis:
    >>> pp = PostProcessor("results/experiment.h5")
    >>> timings = pp.aggregate_timings()
    >>> df = pp.to_dataframe()

    Multi-file scaling study:
    >>> pp = PostProcessor(["results/np2.h5", "results/np4.h5", "results/np6.h5"])
    >>> df = pp.to_dataframe()
    >>> pp.plot_strong_scaling()
    """

    def __init__(self, paths: Union[str, Path, List[Union[str, Path]]]):
        # Normalize to list of Path objects
        if not isinstance(paths, list):
            paths = [paths]
        self.paths = [Path(p) for p in paths]

        # Load data from all files
        self.runs = [self._load_hdf5(p) for p in self.paths]

    def _load_hdf5(self, path: Path) -> Dict[str, Any]:
        """Load complete simulation data from HDF5 file.

        Parameters
        ----------
        path : Path
            Path to HDF5 file

        Returns
        -------
        dict
            Dictionary with keys: 'config', 'fields', 'results', 'timings'
        """
        import h5py

        data = {}

        with h5py.File(path, 'r') as f:
            # Load config (scalar attributes)
            data['config'] = dict(f['config'].attrs)

            # Load fields (solution array)
            data['fields'] = {
                'u': f['fields']['u'][:]
            }

            # Load results (convergence info)
            data['results'] = dict(f['results'].attrs)

            # Load per-rank timings
            data['timings'] = {}
            for rank_name in f['timings'].keys():
                rank_id = int(rank_name.split('_')[1])
                data['timings'][rank_id] = {
                    'compute_times': f[f'timings/{rank_name}/compute_times'][:],
                    'mpi_comm_times': f[f'timings/{rank_name}/mpi_comm_times'][:],
                    'halo_exchange_times': f[f'timings/{rank_name}/halo_exchange_times'][:],
                }

                # Rank 0 additionally has residual history
                if rank_id == 0 and 'residual_history' in f[f'timings/{rank_name}']:
                    data['timings'][rank_id]['residual_history'] = \
                        f[f'timings/{rank_name}/residual_history'][:]

        return data

    def aggregate_timings(self, run_index: int = 0) -> Dict[str, float]:
        """Aggregate per-rank timings to global statistics.

        Parameters
        ----------
        run_index : int, optional
            Index of run to aggregate (default: 0)

        Returns
        -------
        dict
            Dictionary with aggregated timing statistics:
            - total_compute_time: Sum of all compute times across all ranks
            - total_mpi_comm_time: Sum of all MPI communication times
            - total_halo_exchange_time: Sum of all halo exchange times
            - mean_compute_time: Mean compute time per rank
            - mean_mpi_comm_time: Mean MPI communication time per rank
            - mean_halo_exchange_time: Mean halo exchange time per rank
            - max_compute_time: Maximum compute time across ranks
            - max_mpi_comm_time: Maximum MPI communication time across ranks
            - max_halo_exchange_time: Maximum halo exchange time across ranks
        """
        timings = self.runs[run_index]['timings']

        # Sum timings across all ranks
        total_compute = sum(np.sum(t['compute_times']) for t in timings.values())
        total_mpi_comm = sum(np.sum(t['mpi_comm_times']) for t in timings.values())
        total_halo = sum(np.sum(t['halo_exchange_times']) for t in timings.values())

        # Mean timings per rank
        n_ranks = len(timings)
        mean_compute = total_compute / n_ranks
        mean_mpi_comm = total_mpi_comm / n_ranks
        mean_halo = total_halo / n_ranks

        # Max timings across ranks
        max_compute = max(np.sum(t['compute_times']) for t in timings.values())
        max_mpi_comm = max(np.sum(t['mpi_comm_times']) for t in timings.values())
        max_halo = max(np.sum(t['halo_exchange_times']) for t in timings.values())

        return {
            'total_compute_time': total_compute,
            'total_mpi_comm_time': total_mpi_comm,
            'total_halo_exchange_time': total_halo,
            'mean_compute_time': mean_compute,
            'mean_mpi_comm_time': mean_mpi_comm,
            'mean_halo_exchange_time': mean_halo,
            'max_compute_time': max_compute,
            'max_mpi_comm_time': max_mpi_comm,
            'max_halo_exchange_time': max_halo,
        }

    def get_config(self, run_index: int = 0) -> Dict[str, Any]:
        """Get configuration for a run.

        Parameters
        ----------
        run_index : int, optional
            Index of run (default: 0)

        Returns
        -------
        dict
            Configuration dictionary
        """
        return self.runs[run_index]['config']

    def get_convergence(self, run_index: int = 0) -> Dict[str, Any]:
        """Get convergence information for a run.

        Parameters
        ----------
        run_index : int, optional
            Index of run (default: 0)

        Returns
        -------
        dict
            Results dictionary with iterations, converged, final_error
        """
        return self.runs[run_index]['results']

    def get_residual_history(self, run_index: int = 0) -> np.ndarray:
        """Get residual history for a run.

        Parameters
        ----------
        run_index : int, optional
            Index of run (default: 0)

        Returns
        -------
        np.ndarray
            Residual history array
        """
        return self.runs[run_index]['timings'][0]['residual_history']

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all runs to a single DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with one row per run, columns for config, results, and
            aggregated timings
        """
        rows = []
        for i, run in enumerate(self.runs):
            row = {}

            # Add config
            row.update(run['config'])

            # Add results
            row.update(run['results'])

            # Add aggregated timings
            timings = self.aggregate_timings(run_index=i)
            row.update(timings)

            rows.append(row)

        return pd.DataFrame(rows)

    def log_to_mlflow(self, experiment_name: str, run_index: int = 0):
        """Log a single run to MLflow.

        Parameters
        ----------
        experiment_name : str
            MLflow experiment name
        run_index : int, optional
            Index of run to log (default: 0)
        """
        import mlflow

        mlflow.set_experiment(experiment_name)

        with mlflow.start_run():
            # Log config as parameters
            config = self.get_config(run_index)
            for key, value in config.items():
                mlflow.log_param(key, value)

            # Log results as metrics
            results = self.get_convergence(run_index)
            for key, value in results.items():
                mlflow.log_metric(key, value)

            # Log aggregated timings as metrics
            timings = self.aggregate_timings(run_index)
            for key, value in timings.items():
                mlflow.log_metric(key, value)

    def plot_residual_convergence(self, run_index: int = 0, ax=None):
        """Plot residual convergence history.

        Parameters
        ----------
        run_index : int, optional
            Index of run to plot (default: 0)
        ax : matplotlib axes, optional
            Axes to plot on (creates new figure if None)

        Returns
        -------
        matplotlib axes
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))

        residuals = self.get_residual_history(run_index)
        ax.semilogy(residuals, linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Residual')
        ax.set_title('Convergence History')
        ax.grid(True, alpha=0.3)

        return ax

    def plot_per_rank_breakdown(self, run_index: int = 0, ax=None):
        """Plot timing breakdown per rank.

        Parameters
        ----------
        run_index : int, optional
            Index of run to plot (default: 0)
        ax : matplotlib axes, optional
            Axes to plot on (creates new figure if None)

        Returns
        -------
        matplotlib axes
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        timings = self.runs[run_index]['timings']
        ranks = sorted(timings.keys())

        compute_times = [np.sum(timings[r]['compute_times']) for r in ranks]
        mpi_comm_times = [np.sum(timings[r]['mpi_comm_times']) for r in ranks]
        halo_times = [np.sum(timings[r]['halo_exchange_times']) for r in ranks]

        x = np.arange(len(ranks))
        width = 0.25

        ax.bar(x - width, compute_times, width, label='Compute')
        ax.bar(x, mpi_comm_times, width, label='MPI Comm')
        ax.bar(x + width, halo_times, width, label='Halo Exchange')

        ax.set_xlabel('Rank')
        ax.set_ylabel('Time (s)')
        ax.set_title('Timing Breakdown per Rank')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{r}' for r in ranks])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        return ax

    def plot_strong_scaling(self, ax=None):
        """Plot strong scaling efficiency.

        Requires multiple runs with different MPI sizes.

        Parameters
        ----------
        ax : matplotlib axes, optional
            Axes to plot on (creates new figure if None)

        Returns
        -------
        matplotlib axes
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))

        df = self.to_dataframe()

        # Get total wall time (max compute + mpi across ranks)
        df['total_time'] = df['max_compute_time'] + df['max_mpi_comm_time']

        # Compute speedup relative to smallest run
        baseline_time = df.loc[df['mpi_size'].idxmin(), 'total_time']
        df['speedup'] = baseline_time / df['total_time']

        # Ideal scaling
        baseline_size = df['mpi_size'].min()
        df['ideal_speedup'] = df['mpi_size'] / baseline_size

        # Plot
        ax.plot(df['mpi_size'], df['speedup'], 'o-', linewidth=2, markersize=8, label='Actual')
        ax.plot(df['mpi_size'], df['ideal_speedup'], '--', linewidth=2, alpha=0.7, label='Ideal')

        ax.set_xlabel('Number of MPI Ranks')
        ax.set_ylabel('Speedup')
        ax.set_title('Strong Scaling Efficiency')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def plot_communication_overhead(self, ax=None):
        """Plot communication overhead vs MPI size.

        Requires multiple runs with different MPI sizes.

        Parameters
        ----------
        ax : matplotlib axes, optional
            Axes to plot on (creates new figure if None)

        Returns
        -------
        matplotlib axes
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))

        df = self.to_dataframe()

        # Compute communication fraction
        df['total_time'] = df['max_compute_time'] + df['max_mpi_comm_time']
        df['comm_fraction'] = df['max_mpi_comm_time'] / df['total_time']

        ax.plot(df['mpi_size'], df['comm_fraction'] * 100, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of MPI Ranks')
        ax.set_ylabel('Communication Overhead (%)')
        ax.set_title('Communication Overhead vs MPI Size')
        ax.grid(True, alpha=0.3)

        return ax

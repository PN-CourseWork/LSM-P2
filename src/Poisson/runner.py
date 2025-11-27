"""Run Poisson solver via mpiexec subprocess."""

import json
import subprocess
import tempfile
from pathlib import Path


def run_solver(N: int, n_ranks: int = 1, output: str = None, **kwargs) -> dict:
    """Run solver with N grid points on n_ranks MPI processes.

    Parameters
    ----------
    N : int
        Grid size
    n_ranks : int
        Number of MPI ranks
    output : str, optional
        Path to save HDF5 results (uses temp file if not provided)
    **kwargs
        Extra options: strategy, communicator, max_iter, tol, validate

    Returns
    -------
    dict
        Results with config and metrics (or 'error' key on failure)
    """
    import pandas as pd

    # Use temp file if no output path specified
    use_temp = output is None
    if use_temp:
        tmp = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
        output = tmp.name
        tmp.close()

    config = {"N": N, "output": output, **kwargs}
    cmd = ["mpiexec", "-n", str(n_ranks), "uv", "run", "python", "-m", "Poisson.runner_helper", json.dumps(config)]

    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        return {"error": proc.stderr}

    # Load results from HDF5
    if not Path(output).exists():
        return {"error": "No output file created", "stderr": proc.stderr}

    result = pd.read_hdf(output, key='results').iloc[0].to_dict()

    # Clean up temp file if we created one
    if use_temp:
        Path(output).unlink(missing_ok=True)

    return result

"""I/O utilities for loading and saving simulation data."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd


def load_simulation_data(
    data_dir: Path | str,
    filename_base: str,
    prefer: Literal["parquet", "pickle"] = "parquet",
) -> pd.DataFrame:
    """Load simulation data with automatic fallback between parquet and pickle.

    Parameters
    ----------
    data_dir : Path or str
        Directory containing the data files
    filename_base : str
        Base filename without extension (e.g., 'kdv_two_soliton')
    prefer : {'parquet', 'pickle'}
        Preferred format to try first

    Returns
    -------
    pd.DataFrame
        Loaded dataframe

    Raises
    ------
    FileNotFoundError
        If neither parquet nor pickle file exists

    """
    data_dir = Path(data_dir)
    parquet_path = data_dir / f"{filename_base}.parquet"
    pickle_path = data_dir / f"{filename_base}.pkl"

    if prefer == "parquet":
        primary, secondary = parquet_path, pickle_path
        primary_loader, secondary_loader = pd.read_parquet, pd.read_pickle
        primary_fmt, secondary_fmt = "parquet", "pickle"
    else:
        primary, secondary = pickle_path, parquet_path
        primary_loader, secondary_loader = pd.read_pickle, pd.read_parquet
        primary_fmt, secondary_fmt = "pickle", "parquet"

    if primary.exists():
        print(f"Loading {primary_fmt} data: {primary}")
        return primary_loader(primary)
    elif secondary.exists():
        print(
            f"{primary_fmt.capitalize()} not found; loading {secondary_fmt} data: {secondary}"
        )
        return secondary_loader(secondary)
    else:
        raise FileNotFoundError(
            f"No dataset found at {data_dir / filename_base}.{{parquet,pkl}}. "
            f"Run the corresponding compute script first."
        )


def save_simulation_data(
    df: pd.DataFrame,
    output_path: Path | str,
    format: Literal["parquet", "pickle"] = "parquet",
) -> None:
    """Save simulation data to disk.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save
    output_path : Path or str
        Output file path (should include extension)
    format : {'parquet', 'pickle'}
        Output format

    """
    output_path = Path(output_path)

    if format == "parquet":
        df.to_parquet(output_path, index=False)
    elif format == "pickle":
        df.to_pickle(output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")

    print(f"Saved {format} data → {output_path} ({df.shape})")


def ensure_output_dir(path: Path | str) -> Path:
    """Ensure output directory exists, creating it if necessary.

    Parameters
    ----------
    path : Path or str
        Directory path to create

    Returns
    -------
    Path
        The created/existing directory path

    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

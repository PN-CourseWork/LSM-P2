"""Cleanup utilities for generated files and caches.

Provides configurable cleanup of build artifacts, caches,
and generated data files.
"""

import shutil
from pathlib import Path
from typing import List, Tuple, Optional

from .paths import get_repo_root


def _remove_item(path: Path) -> Tuple[bool, Optional[str]]:
    """Remove a file or directory.

    Returns
    -------
    tuple
        (success, error_message)
    """
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        return True, None
    except Exception as e:
        return False, str(e)


def clean_directories(
    directories: Optional[List[str]] = None,
    repo_root: Optional[Path] = None,
) -> Tuple[int, int]:
    """Clean specified directories.

    Parameters
    ----------
    directories : list of str, optional
        Directories to clean (relative to repo root).
        Uses defaults if not specified.
    repo_root : Path, optional
        Repository root path.

    Returns
    -------
    tuple
        (cleaned_count, failed_count)
    """
    if repo_root is None:
        repo_root = get_repo_root()

    if directories is None:
        directories = [
            "docs/build",
            "docs/source/example_gallery",
            "docs/source/generated",
            "docs/source/gen_modules",
            "plots",
            "build",
            "dist",
            ".pytest_cache",
            ".ruff_cache",
            ".mypy_cache",
        ]

    cleaned, failed = 0, 0
    for d in directories:
        path = repo_root / d
        if path.exists():
            success, _ = _remove_item(path)
            cleaned += success
            failed += not success

    return cleaned, failed


def clean_files(
    files: Optional[List[str]] = None,
    repo_root: Optional[Path] = None,
) -> Tuple[int, int]:
    """Clean specified files.

    Parameters
    ----------
    files : list of str, optional
        Files to clean (relative to repo root).
    repo_root : Path, optional
        Repository root path.

    Returns
    -------
    tuple
        (cleaned_count, failed_count)
    """
    if repo_root is None:
        repo_root = get_repo_root()

    if files is None:
        files = [
            "docs/source/sg_execution_times.rst",
        ]

    cleaned, failed = 0, 0
    for f in files:
        path = repo_root / f
        if path.exists():
            success, _ = _remove_item(path)
            cleaned += success
            failed += not success

    return cleaned, failed


def clean_patterns(
    patterns: Optional[List[str]] = None,
    repo_root: Optional[Path] = None,
) -> Tuple[int, int]:
    """Clean files/directories matching patterns recursively.

    Parameters
    ----------
    patterns : list of str, optional
        Glob patterns to match.
    repo_root : Path, optional
        Repository root path.

    Returns
    -------
    tuple
        (cleaned_count, failed_count)
    """
    if repo_root is None:
        repo_root = get_repo_root()

    if patterns is None:
        patterns = [
            "__pycache__",
            "*.pyc",
            ".DS_Store",
            "mlruns",
            "multirun",
            "output",
            "outputs",
        ]

    cleaned, failed = 0, 0
    for pattern in patterns:
        for path in repo_root.rglob(pattern):
            success, _ = _remove_item(path)
            cleaned += success
            failed += not success

    return cleaned, failed


def clean_data_directory(
    data_dir: str = "data",
    preserve: Optional[List[str]] = None,
    repo_root: Optional[Path] = None,
) -> Tuple[int, int]:
    """Clean data directory contents, preserving specific files.

    Parameters
    ----------
    data_dir : str
        Data directory relative to repo root.
    preserve : list of str, optional
        Filenames to preserve.
    repo_root : Path, optional
        Repository root path.

    Returns
    -------
    tuple
        (cleaned_count, failed_count)
    """
    if repo_root is None:
        repo_root = get_repo_root()

    if preserve is None:
        preserve = ["README.md", ".gitkeep"]

    data_path = repo_root / data_dir
    if not data_path.exists():
        return 0, 0

    cleaned, failed = 0, 0
    for item in data_path.iterdir():
        if item.name not in preserve:
            success, _ = _remove_item(item)
            cleaned += success
            failed += not success

    return cleaned, failed


def clean_experiment_outputs(
    experiments_dir: str = "Experiments",
    output_dir_name: str = "output",
    repo_root: Optional[Path] = None,
) -> Tuple[int, int]:
    """Clean output directories in experiment folders.

    Parameters
    ----------
    experiments_dir : str
        Experiments directory relative to repo root.
    output_dir_name : str
        Name of output subdirectories to clean.
    repo_root : Path, optional
        Repository root path.

    Returns
    -------
    tuple
        (cleaned_count, failed_count)
    """
    if repo_root is None:
        repo_root = get_repo_root()

    exp_path = repo_root / experiments_dir
    if not exp_path.exists():
        return 0, 0

    cleaned, failed = 0, 0
    for output_dir in exp_path.glob(f"*/{output_dir_name}"):
        success, _ = _remove_item(output_dir)
        cleaned += success
        failed += not success

    return cleaned, failed


def clean_all() -> None:
    """Clean all generated files and caches."""
    print("\nCleaning all generated files and caches...")

    total_cleaned = 0
    total_failed = 0

    c, f = clean_directories()
    total_cleaned += c
    total_failed += f

    c, f = clean_files()
    total_cleaned += c
    total_failed += f

    c, f = clean_patterns()
    total_cleaned += c
    total_failed += f

    c, f = clean_data_directory()
    total_cleaned += c
    total_failed += f

    c, f = clean_experiment_outputs()
    total_cleaned += c
    total_failed += f

    if total_cleaned:
        print(f"  ✓ Cleaned {total_cleaned} items")
    if total_failed:
        print(f"  ✗ Failed to clean {total_failed} items")
    if not total_cleaned and not total_failed:
        print("  Nothing to clean")
    print()

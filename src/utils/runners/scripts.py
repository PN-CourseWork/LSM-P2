"""Script discovery and execution utilities.

Provides parallel and sequential script execution with
configurable timeouts and progress reporting.
"""

import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional

from ..config import get_repo_root


def discover_scripts(pattern: str, directory: str = "Experiments") -> List[Path]:
    """Find scripts in a directory matching pattern.

    Parameters
    ----------
    pattern : str
        Pattern to match in script names (e.g., "plot", "compute")
    directory : str, default "Experiments"
        Directory to search in, relative to repo root

    Returns
    -------
    list of Path
        Sorted list of matching script paths
    """
    repo_root = get_repo_root()
    search_dir = repo_root / directory

    if not search_dir.exists():
        return []

    scripts = [
        p
        for p in search_dir.rglob("*.py")
        if p.is_file() and pattern in p.name and p.name != "__init__.py"
    ]

    return sorted(scripts)


def _run_single_script(
    script: Path,
    repo_root: Path,
    timeout: int = 180,
    interpreter: str = "uv run python",
) -> Tuple[Path, bool, Optional[str]]:
    """Run a single script and return its result.

    Parameters
    ----------
    script : Path
        Path to the script
    repo_root : Path
        Repository root for relative path display
    timeout : int
        Timeout in seconds
    interpreter : str
        Command to run the script

    Returns
    -------
    tuple
        (display_path, success, error_message)
    """
    display_path = script.relative_to(repo_root)

    try:
        cmd = interpreter.split() + [str(script)]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(repo_root),
        )

        if result.returncode == 0:
            return (display_path, True, None)
        else:
            error_msg = result.stderr[:200] if result.stderr else ""
            return (display_path, False, f"exit {result.returncode}: {error_msg}")

    except subprocess.TimeoutExpired:
        return (display_path, False, "timeout")
    except Exception as e:
        return (display_path, False, str(e))


def run_scripts_parallel(
    scripts: List[Path],
    timeout: int = 180,
    interpreter: str = "uv run python",
    max_workers: int = None,
) -> Tuple[int, int]:
    """Run scripts in parallel using ThreadPoolExecutor.

    Parameters
    ----------
    scripts : list of Path
        Scripts to run
    timeout : int, default 180
        Timeout per script in seconds
    interpreter : str, default "uv run python"
        Command to run scripts
    max_workers : int, optional
        Maximum number of parallel workers

    Returns
    -------
    tuple
        (success_count, fail_count)
    """
    if not scripts:
        print("  No scripts to run")
        return 0, 0

    repo_root = get_repo_root()
    print(f"\nRunning {len(scripts)} scripts in parallel...\n")

    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_script = {
            executor.submit(
                _run_single_script, script, repo_root, timeout, interpreter
            ): script
            for script in scripts
        }

        for future in as_completed(future_to_script):
            display_path, success, error_msg = future.result()

            if success:
                print(f"  ✓ {display_path}")
                success_count += 1
            else:
                print(f"  ✗ {display_path} ({error_msg})")
                fail_count += 1

    print(f"\n  Summary: {success_count} succeeded, {fail_count} failed\n")
    return success_count, fail_count


def run_scripts_sequential(
    scripts: List[Path],
    timeout: int = 600,
    interpreter: str = "uv run python",
) -> Tuple[int, int]:
    """Run scripts sequentially.

    Parameters
    ----------
    scripts : list of Path
        Scripts to run
    timeout : int, default 600
        Timeout per script in seconds
    interpreter : str, default "uv run python"
        Command to run scripts

    Returns
    -------
    tuple
        (success_count, fail_count)
    """
    if not scripts:
        print("  No scripts to run")
        return 0, 0

    repo_root = get_repo_root()
    print(f"\nRunning {len(scripts)} scripts sequentially...\n")

    success_count = 0
    fail_count = 0

    for script in scripts:
        display_path = script.relative_to(repo_root)
        print(f"  → {display_path}...", end=" ", flush=True)

        _, success, error_msg = _run_single_script(
            script, repo_root, timeout, interpreter
        )

        if success:
            print("✓")
            success_count += 1
        else:
            print(f"✗ ({error_msg})")
            fail_count += 1

    print(f"\n  Summary: {success_count} succeeded, {fail_count} failed\n")
    return success_count, fail_count


def run_plot_scripts() -> Tuple[int, int]:
    """Run all plot scripts in parallel."""
    scripts = discover_scripts("plot")
    return run_scripts_parallel(scripts, timeout=180)


def run_compute_scripts() -> Tuple[int, int]:
    """Run all compute scripts sequentially."""
    scripts = discover_scripts("compute")
    return run_scripts_sequential(scripts, timeout=600)


def copy_to_report(
    source_dir: str = "figures",
    dest_dir: str = "docs/reports/TexReport/figures",
) -> bool:
    """Copy a directory to the report location.

    Parameters
    ----------
    source_dir : str
        Source directory relative to repo root
    dest_dir : str
        Destination directory relative to repo root

    Returns
    -------
    bool
        True if successful
    """
    repo_root = get_repo_root()
    source = repo_root / source_dir
    dest = repo_root / dest_dir

    print(f"\nCopying {source_dir}/ to {dest_dir}/...")

    if not source.exists():
        print(f"  No {source_dir}/ directory found")
        return False

    try:
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(source, dest)
        print(f"  ✓ Copied {source_dir}/ to {dest_dir}/")
        return True
    except Exception as e:
        print(f"  ✗ Failed to copy: {e}")
        return False

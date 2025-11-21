#!/usr/bin/env python3
"""Main script for project management."""

import argparse
import subprocess
import sys
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_repo_root() -> Path:
    """Get repository root directory.

    Returns the repository root by detecting the presence of pyproject.toml.
    Works from any subdirectory of the repository.

    Returns
    -------
    Path
        Absolute path to the repository root

    """
    # Start from this script's location
    current = Path(__file__).resolve().parent

    # Walk up until we find pyproject.toml (marks repo root)
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent

    # Fallback: assume script is in repo root
    return current


# Get repo root once at module level
REPO_ROOT = get_repo_root()


def discover_plot_scripts():
    """Find all plotting scripts in Experiments/ directory."""
    experiments_dir = REPO_ROOT / "Experiments"

    if not experiments_dir.exists():
        return []

    scripts = [
        p
        for p in experiments_dir.rglob("*.py")
        if p.is_file() and "plot" in p.name and p.name != "__init__.py"
    ]

    return sorted(scripts)


def _run_single_plot_script(script):
    """Run a single plotting script and return its result.

    Parameters
    ----------
    script : Path
        Path to the script to run

    Returns
    -------
    tuple
        (display_path, success, error_message)
    """
    display_path = script.relative_to(REPO_ROOT)

    try:
        result = subprocess.run(
            ["uv", "run", "python", str(script)],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=str(REPO_ROOT),
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


def run_plot_scripts():
    """Run plotting scripts in parallel and report results."""
    scripts = discover_plot_scripts()

    if not scripts:
        print("  No plot scripts found")
        return

    print(f"\nRunning {len(scripts)} plot scripts in parallel...\n")

    success_count = 0
    fail_count = 0

    # Run scripts in parallel using thread pool
    with ThreadPoolExecutor() as executor:
        # Submit all scripts
        future_to_script = {
            executor.submit(_run_single_plot_script, script): script
            for script in scripts
        }

        # Process results as they complete
        for future in as_completed(future_to_script):
            display_path, success, error_msg = future.result()

            if success:
                print(f"  ✓ {display_path}")
                success_count += 1
            else:
                print(f"  ✗ {display_path} ({error_msg})")
                fail_count += 1

    print(f"\n  Summary: {success_count} succeeded, {fail_count} failed\n")


def copy_plots():
    """Copy figures/ directory to docs/reports/TexReport/."""
    source_dir = REPO_ROOT / "figures"
    dest_dir = REPO_ROOT / "docs" / "reports" / "TexReport" / "figures"

    print("\nCopying figures/ to docs/reports/TexReport/...")

    if not source_dir.exists():
        print("  No figures/ directory found")
        return

    try:
        # Remove existing destination if it exists, then copy entire directory
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(source_dir, dest_dir)
        print(f"  ✓ Copied figures/ to docs/reports/TexReport/figures/")
    except Exception as e:
        print(f"  ✗ Failed to copy: {e}")

    print()


def build_docs():
    """Build Sphinx documentation."""
    docs_dir = REPO_ROOT / "docs"
    source_dir = docs_dir / "source"
    build_dir = docs_dir / "build"

    print("\nBuilding Sphinx documentation...")

    if not source_dir.exists():
        print(f"  Error: Documentation source directory not found: {source_dir}")
        return False

    try:
        result = subprocess.run(
            [
                "uv",
                "run",
                "sphinx-build",
                "-M",
                "html",
                str(source_dir),
                str(build_dir),
            ],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(REPO_ROOT),
        )

        if result.returncode == 0:
            print("  ✓ Documentation built successfully")
            print(f"  → Open: {build_dir / 'html' / 'index.html'}\n")
            return True
        else:
            print(f"  ✗ Documentation build failed (exit {result.returncode})")
            if result.stderr:
                print(f"    Error: {result.stderr[:500]}")
            return False

    except subprocess.TimeoutExpired:
        print("  ✗ Documentation build timed out")
        return False
    except FileNotFoundError:
        print("  ✗ sphinx-build not found. Install with: uv sync")
        return False
    except Exception as e:
        print(f"  ✗ Documentation build failed: {e}")
        return False


def clean_all():
    """Clean all generated files and caches."""
    print("\nCleaning all generated files and caches...")

    cleaned = []
    failed = []

    # List of directories to clean
    clean_targets = [
        REPO_ROOT / "docs" / "build",
        REPO_ROOT / "docs" / "source" / "example_gallery",
        REPO_ROOT / "docs" / "source" / "generated",
        REPO_ROOT / "docs" / "source" / "gen_modules",
        REPO_ROOT / "plots",
        REPO_ROOT / "build",
        REPO_ROOT / "dist",
        REPO_ROOT / ".pytest_cache",
        REPO_ROOT / ".ruff_cache",
        REPO_ROOT / ".mypy_cache",
    ]

    # Clean directories
    for target_path in clean_targets:
        if target_path.exists():
            try:
                shutil.rmtree(target_path)
                cleaned.append(str(target_path.relative_to(REPO_ROOT)))
            except Exception as e:
                failed.append(f"{target_path.relative_to(REPO_ROOT)}: {e}")

    # Clean specific files
    clean_files = [
        REPO_ROOT / "docs" / "source" / "sg_execution_times.rst",
    ]

    for target_file in clean_files:
        if target_file.exists():
            try:
                target_file.unlink()
                cleaned.append(str(target_file.relative_to(REPO_ROOT)))
            except Exception as e:
                failed.append(f"{target_file.relative_to(REPO_ROOT)}: {e}")

    # Clean __pycache__ directories
    for pycache in REPO_ROOT.rglob("__pycache__"):
        try:
            shutil.rmtree(pycache)
            cleaned.append(str(pycache.relative_to(REPO_ROOT)))
        except Exception as e:
            failed.append(f"{pycache.relative_to(REPO_ROOT)}: {e}")

    # Clean .pyc files
    for pyc in REPO_ROOT.rglob("*.pyc"):
        try:
            pyc.unlink()
            cleaned.append(str(pyc.relative_to(REPO_ROOT)))
        except Exception as e:
            failed.append(f"{pyc.relative_to(REPO_ROOT)}: {e}")

    # Clean .DS_Store files (macOS metadata)
    for ds_store in REPO_ROOT.rglob(".DS_Store"):
        try:
            ds_store.unlink()
            cleaned.append(str(ds_store.relative_to(REPO_ROOT)))
        except Exception as e:
            failed.append(f"{ds_store.relative_to(REPO_ROOT)}: {e}")

    # Clean data directory (but keep README.md)
    data_dir = REPO_ROOT / "data"
    if data_dir.exists():
        for item in data_dir.iterdir():
            if item.name != "README.md" and item.name != ".gitkeep":
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                    cleaned.append(str(item.relative_to(REPO_ROOT)))
                except Exception as e:
                    failed.append(f"{item.relative_to(REPO_ROOT)}: {e}")

    # Clean Experiments/*/output directories
    experiments_dir = REPO_ROOT / "Experiments"
    if experiments_dir.exists():
        for output_dir in experiments_dir.glob("*/output"):
            if output_dir.exists():
                try:
                    shutil.rmtree(output_dir)
                    cleaned.append(str(output_dir.relative_to(REPO_ROOT)))
                except Exception as e:
                    failed.append(f"{output_dir.relative_to(REPO_ROOT)}: {e}")

    # Print results
    if cleaned:
        print(f"  ✓ Cleaned {len(cleaned)} items")
    if failed:
        print(f"  ✗ Failed to clean {len(failed)} items:")
        for fail in failed[:5]:  # Show first 5 failures
            print(f"    - {fail}")
    if not cleaned and not failed:
        print("  Nothing to clean")
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Project management for MPI Poisson Solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --docs                        Build Sphinx documentation
  python main.py --plot                        Run all plotting scripts
  python main.py --copy-plots                  Copy plots to plots/ directory
  python main.py --clean                       Clean all generated files
  python main.py --plot --copy-plots           Generate and copy plots
        """,
    )

    parser.add_argument("--docs", action="store_true", help="Build Sphinx HTML documentation")
    parser.add_argument("--plot", action="store_true", help="Run all plotting scripts")
    parser.add_argument("--copy-plots", action="store_true", help="Copy plots to plots/ directory")
    parser.add_argument("--clean", action="store_true", help="Clean all generated files and caches")

    # Show help if no arguments provided
    if len(sys.argv) == 1:
        parser.print_help()
        print("\n Error: No arguments provided. Please specify at least one option.\n")
        sys.exit(1)

    args = parser.parse_args()

    # Execute commands in logical order
    if args.clean:
        clean_all()

    if args.plot:
        run_plot_scripts()

    if args.copy_plots:
        copy_plots()

    if args.docs:
        build_docs()


if __name__ == "__main__":
    main()

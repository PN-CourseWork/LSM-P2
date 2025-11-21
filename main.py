#!/usr/bin/env python3
"""Main script for project management."""

import argparse
import subprocess
import sys
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


# Repository root (main.py is at repo root)
REPO_ROOT = Path(__file__).resolve().parent


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

    def remove_item(path):
        """Remove file or directory, return (success, error)."""
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            return True, None
        except Exception as e:
            return False, str(e)

    cleaned, failed = 0, 0

    # Directories to clean
    dirs = [
        "docs/build", "docs/source/example_gallery", "docs/source/generated",
        "docs/source/gen_modules", "plots", "build", "dist",
        ".pytest_cache", ".ruff_cache", ".mypy_cache",
    ]
    for d in dirs:
        path = REPO_ROOT / d
        if path.exists():
            success, _ = remove_item(path)
            cleaned += success
            failed += not success

    # Specific files to clean
    files = ["docs/source/sg_execution_times.rst"]
    for f in files:
        path = REPO_ROOT / f
        if path.exists():
            success, _ = remove_item(path)
            cleaned += success
            failed += not success

    # Recursive patterns to clean
    patterns = ["__pycache__", "*.pyc", ".DS_Store"]
    for pattern in patterns:
        for path in REPO_ROOT.rglob(pattern):
            success, _ = remove_item(path)
            cleaned += success
            failed += not success

    # Clean data/ directory contents (preserve README.md and .gitkeep)
    data_dir = REPO_ROOT / "data"
    if data_dir.exists():
        for item in data_dir.iterdir():
            if item.name not in {"README.md", ".gitkeep"}:
                success, _ = remove_item(item)
                cleaned += success
                failed += not success

    # Clean Experiments/*/output directories
    for output_dir in (REPO_ROOT / "Experiments").glob("*/output"):
        success, _ = remove_item(output_dir)
        cleaned += success
        failed += not success

    # Print results
    if cleaned:
        print(f"  ✓ Cleaned {cleaned} items")
    if failed:
        print(f"  ✗ Failed to clean {failed} items")
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

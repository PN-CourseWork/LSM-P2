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


def discover_compute_scripts():
    """Find all compute scripts in Experiments/ directory."""
    experiments_dir = REPO_ROOT / "Experiments"

    if not experiments_dir.exists():
        return []

    scripts = [
        p
        for p in experiments_dir.rglob("*.py")
        if p.is_file() and "compute" in p.name and p.name != "__init__.py"
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


def run_compute_scripts():
    """Run compute scripts sequentially."""
    scripts = discover_compute_scripts()

    if not scripts:
        print("  No compute scripts found")
        return

    print(f"\nRunning {len(scripts)} compute scripts sequentially...\n")

    success_count = 0
    fail_count = 0

    for script in scripts:
        display_path = script.relative_to(REPO_ROOT)
        print(f"  → {display_path}...", end=" ", flush=True)

        try:
            result = subprocess.run(
                ["uv", "run", "python", str(script)],
                capture_output=True,
                text=True,
                timeout=600,  # 10 min timeout for compute scripts
                cwd=str(REPO_ROOT),
            )

            if result.returncode == 0:
                print("✓")
                success_count += 1
            else:
                error_msg = (
                    result.stderr[:200]
                    if result.stderr
                    else f"exit {result.returncode}"
                )
                print(f"✗ ({error_msg})")
                fail_count += 1

        except subprocess.TimeoutExpired:
            print("✗ (timeout)")
            fail_count += 1
        except Exception as e:
            print(f"✗ ({e})")
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
        print("  ✓ Copied figures/ to docs/reports/TexReport/figures/")
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


def hpc_submit_pack(scaling_type: str, dry_run: bool):
    """Generate and optionally submit an LSF job pack."""
    print(f"\nGenerating {scaling_type} scaling job pack...")

    pack_file_name = f"Experiments/05-scaling/{scaling_type}_scaling_jobs.pack"
    pack_file_path = REPO_ROOT / pack_file_name

    # Generate the pack file
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "src.utils.generate_pack",
        "--type",
        scaling_type,
        "--output",
        str(pack_file_path),
        "--config-dir",
        str(REPO_ROOT / "Experiments" / "05-scaling"),
    ]
    # Use standard N values for strong scaling if not specified in generate_pack default
    if scaling_type == "strong":
        # Hardcoded defaults for project consistency
        cmd.extend(["--N", "64", "128", "256"])

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))

    if result.returncode != 0:
        print(f"  ✗ Failed to generate pack file: {result.stderr}")
        return
    print(f"  ✓ {result.stdout.strip()}")

    if dry_run:
        print(f"\n[DRY RUN] Content of {pack_file_name}:")
        print("-" * 40)
        print(pack_file_path.read_text())
        print("-" * 40)
        print(
            f"  To submit manually: bsub -pack {pack_file_name}"
        )
        return

    # Submit the pack file
    print(f"\nSubmitting {pack_file_path} to LSF...")
    
    if not shutil.which("bsub"):
        print("  ✗ 'bsub' command not found. Are you on the HPC login node?")
        return

    submit_cmd = ["bsub", "-pack", str(pack_file_path)]
    
    result = subprocess.run(
        submit_cmd, capture_output=True, text=True, cwd=str(REPO_ROOT)
    )

    if result.returncode != 0:
        print(f"  ✗ Failed to submit jobs: {result.stderr}")
    else:
        print(f"  ✓ Jobs submitted successfully.")
        if result.stdout:
            print(f"    {result.stdout.strip()}")


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


def fetch_mlflow():
    """Fetch artifacts from MLflow for all converged runs."""
    print("\nFetching MLflow artifacts...")

    try:
        # Import locally to avoid hard dependency if not fetching
        # from utils.mlflow_io import download_artifacts_with_naming, setup_mlflow_auth
        from utils.mlflow_io import setup_mlflow_auth

        setup_mlflow_auth()

        # Define download targets - modifying for LSM Project 2 context
        # Assuming we might have experiments named like 'LSM-Project-2/Scaling' or similar
        # For now, we'll use a placeholder or a generic search if available,
        # but matching the ANA-P3 pattern:

        # output_dir = REPO_ROOT / "data" / "downloaded"

        # Example: Fetch from a "Scaling" experiment
        # experiments = ["LSM-Scaling", "LSM-Kernels"]
        # for exp in experiments:
        #     print(f"\n{exp}:")
        #     paths = download_artifacts_with_naming(exp, output_dir / exp)
        #     print(f"  ✓ Downloaded {len(paths)} files to data/downloaded/{exp}/")

        print(
            "  (No experiments configured for auto-fetch yet. Edit main.py to specify experiments.)"
        )
        print()

    except ImportError as e:
        print(f"  ✗ Missing dependency: {e}")
        print("    Install with: uv sync")
    except Exception as e:
        print(f"  ✗ Failed to fetch: {e}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Project management for MPI Poisson Solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --docs                        Build Sphinx documentation
  python main.py --compute                     Run all compute scripts
  python main.py --plot                        Run all plotting scripts
  python main.py --copy-plots                  Copy plots to plots/ directory
  python main.py --clean                       Clean all generated files
  python main.py --hpc strong --dry            Generate strong scaling pack (dry run)
        """,
    )

    # Action Group
    actions = parser.add_argument_group("Actions")
    actions.add_argument(
        "--docs", action="store_true", help="Build Sphinx HTML documentation"
    )
    actions.add_argument(
        "--compute", action="store_true", help="Run all compute scripts (sequentially)"
    )
    actions.add_argument(
        "--plot", action="store_true", help="Run all plotting scripts (in parallel)"
    )
    actions.add_argument(
        "--copy-plots", action="store_true", help="Copy plots to plots/ directory"
    )
    actions.add_argument(
        "--clean", action="store_true", help="Clean all generated files and caches"
    )
    actions.add_argument(
        "--fetch",
        action="store_true",
        help="Fetch artifacts from MLflow for all converged runs",
    )
    actions.add_argument(
        "--hpc",
        choices=["strong", "weak"],
        help="Generate and submit LSF job pack for scaling",
    )

    # Options Group
    options = parser.add_argument_group("Options")
    options.add_argument(
        "--dry",
        action="store_true",
        help="Print generated job pack without submitting (for --hpc)",
    )

    # Show help if no arguments provided
    if len(sys.argv) == 1:
        parser.print_help()
        print("\n Error: No arguments provided. Please specify at least one option.\n")
        sys.exit(1)

    args = parser.parse_args()

    # Execute commands in logical order
    if args.clean:
        clean_all()

    if args.compute:
        run_compute_scripts()

    if args.plot:
        run_plot_scripts()

    if args.copy_plots:
        copy_plots()

    if args.fetch:
        fetch_mlflow()

    # Handle HPC pack submission
    if args.hpc:
        hpc_submit_pack(args.hpc, args.dry)

    # Handle documentation commands

    if args.docs:
        build_docs()


if __name__ == "__main__":
    main()

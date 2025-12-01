#!/usr/bin/env python3
"""Main entry point for project management - CLI driven."""

import argparse
import sys
from pathlib import Path

# Ensure src directory is in python path
sys.path.append(str(Path(__file__).parent / "src"))

from utils import runners
from utils import mlflow as mlflow_utils
from utils.config import get_repo_root, load_project_config, clean_all
from utils.hpc import interactive_generate


def build_docs():
    """Build Sphinx documentation."""
    import subprocess

    repo_root = get_repo_root()
    docs_dir = repo_root / "docs"
    source_dir = docs_dir / "source"
    build_dir = docs_dir / "build"

    print("\nBuilding Sphinx documentation...")

    if not source_dir.exists():
        print(f"  Error: Documentation source directory not found: {source_dir}")
        return False

    try:
        result = subprocess.run(
            ["uv", "run", "sphinx-build", "-M", "html", str(source_dir), str(build_dir)],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(repo_root),
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


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Project management for MPI Poisson Solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    actions = parser.add_argument_group("Actions")
    actions.add_argument("--docs", action="store_true", help="Build Sphinx HTML documentation")
    actions.add_argument("--compute", action="store_true", help="Run all compute scripts (sequentially)")
    actions.add_argument("--plot", action="store_true", help="Run all plotting scripts (in parallel)")
    actions.add_argument("--copy-plots", action="store_true", help="Copy plots to report directory")
    actions.add_argument("--clean", action="store_true", help="Clean all generated files and caches")
    actions.add_argument("--setup-mlflow", action="store_true", help="Interactive MLflow setup (login to Databricks)")
    actions.add_argument("--fetch", action="store_true", help="Fetch artifacts from MLflow (skips existing files)")
    actions.add_argument("--force", action="store_true", help="Force re-download of existing files (use with --fetch)")
    actions.add_argument("--hpc", nargs="?", const="DEFAULT", help="Interactive HPC job generator (optional: config path)")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    # Execute commands in logical order
    if args.clean:
        clean_all()

    if args.setup_mlflow:
        import mlflow
        print("\nSetting up MLflow...")
        mlflow.login(backend="databricks", interactive=True)

    if args.compute:
        runners.run_compute_scripts()

    if args.plot:
        runners.run_plot_scripts()

    if args.copy_plots:
        runners.copy_to_report()

    if args.fetch:
        config = load_project_config()
        mlflow_conf = config.get("mlflow", {})
        repo_root = get_repo_root()

        mlflow_utils.setup_mlflow_tracking()
        output_dir = repo_root / mlflow_conf.get("download_dir", "data")
        mlflow_utils.fetch_project_artifacts(output_dir, force=args.force)

    if args.hpc:
        config = load_project_config()
        default_conf = config.get("hpc", {}).get(
            "default_config", "Experiments/06-scaling/job-packs/packs.yaml"
        )
        config_path = default_conf if args.hpc == "DEFAULT" else args.hpc
        interactive_generate(config_path)

    if args.docs:
        if not build_docs():
            sys.exit(1)


if __name__ == "__main__":
    main()

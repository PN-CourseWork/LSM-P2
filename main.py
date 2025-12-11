#!/usr/bin/env python3
"""Main entry point for project management - CLI driven."""

import argparse
import sys
from pathlib import Path
import os # Added for os.setsid

# Ensure src directory is in python path
sys.path.append(str(Path(__file__).parent / "src"))

from utils import runners
from utils import mlflow as mlflow_utils
from utils.config import get_repo_root, clean_all


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
    actions.add_argument("--mlflow-ui", action="store_true", help="Start MLflow UI and open in browser")

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

    if args.mlflow_ui:
        import subprocess
        import webbrowser
        import time

        print("\nStarting MLflow UI...")
        try:
            # Default to standard file-based backend (./mlruns)
            cmd = ["uv", "run", "mlflow", "ui"]
            
            # Start MLflow UI in the background
            log_file = open("mlflow_ui.log", "w")
            mlflow_process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid # Detach process to run in background
            )
            print(f"MLflow UI started in background with PID: {mlflow_process.pid}")
            print(f"Logs redirected to: mlflow_ui.log")
            
            # Give MLflow UI some time to start up
            time.sleep(3) 

            # Open in browser
            url = "http://localhost:5000"
            webbrowser.open_new_tab(url)
            print(f"Opening MLflow UI in browser: {url}")
            print("Press Ctrl+C to stop MLflow UI process later if it's still running.")
        except FileNotFoundError:
            print("Error: 'uv' command not found. Ensure uv is installed and in PATH.")
        except Exception as e:
            print(f"Error starting MLflow UI: {e}")

    if args.docs:
        if not build_docs():
            sys.exit(1)


if __name__ == "__main__":
    main()

"""Batch/CLI mode for running commands without TUI."""

import argparse
import sys

from utils import runners, mlflow
from utils.config import get_repo_root, load_project_config, clean_all
from utils.tui.actions import hpc, docs


def handle_args():
    """Handle command line arguments for batch/automation mode."""
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
  python main.py --hpc                         Interactive HPC job generator (default config)
  python main.py --hpc my_config.yaml          Interactive HPC job generator (custom config)
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
        nargs="?",
        const="DEFAULT",
        help="Generate and submit LSF job pack (optional: specify config path)",
    )

    args = parser.parse_args()

    # Execute commands in logical order
    if args.clean:
        clean_all()

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

        mlflow.setup_mlflow_tracking() 

        output_dir = repo_root / mlflow_conf.get("download_dir", "data/")
        
        print("Fetching all project artifacts from MLflow...")
        mlflow.fetch_project_artifacts(output_dir)

    if args.hpc:
        config = load_project_config()
        default_conf = config.get("hpc", {}).get(
            "default_config", "Experiments/05-scaling/configs/template.yaml"
        )

        config_path = default_conf if args.hpc == "DEFAULT" else args.hpc

        hpc.run_hpc_menu(config_path)

    if args.docs:
        if not docs.build_docs():
            sys.exit(1)

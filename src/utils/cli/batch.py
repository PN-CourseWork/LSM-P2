import argparse
import sys
from src.utils import manage, mlflow_io
from src.utils.cli.actions import hpc

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
        const="DEFAULT", # Sentinel value to distinguish flag presence from None
        help="Generate and submit LSF job pack (optional: specify config path)",
    )

    args = parser.parse_args()

    # Execute commands in logical order
    if args.clean:
        manage.clean_all()

    if args.compute:
        manage.run_compute_scripts()

    if args.plot:
        manage.run_plot_scripts()

    if args.copy_plots:
        manage.copy_plots()

    if args.fetch:
        config = manage.load_project_config()
        mlflow_conf = config.get("mlflow", {})
        repo_root = manage.get_repo_root()
        
        # Setup Auth
        mlflow_io.setup_mlflow_auth(mlflow_conf.get("tracking_uri"))
        
        # Fetch
        output_dir = repo_root / mlflow_conf.get("download_dir", "data/downloaded")
        experiments = mlflow_conf.get("experiments", [])
        if not experiments:
            print("  (No experiments configured in project_config.yaml)")
        else:
            mlflow_io.fetch_project_artifacts(experiments, output_dir)

    # Handle HPC pack submission
    if args.hpc:
        config = manage.load_project_config()
        default_conf = config.get("hpc", {}).get("default_config", "Experiments/05-scaling/configs/template.yaml")
        
        config_path = default_conf if args.hpc == "DEFAULT" else args.hpc
        
        # Call the menu from actions
        hpc.run_hpc_menu(config_path)

    if args.docs:
        if not manage.build_docs():
            sys.exit(1)
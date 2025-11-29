import questionary
import sys
from src.utils.hpc import _interactive_generate, _interactive_submit
from src.utils.cli.styles import get_custom_style
from src.utils.manage import load_project_config
from pathlib import Path

def run_hpc_menu(config_path_str: str = None):
    """HPC Submenu"""
    
    repo_root = Path.cwd()
    
    project_config = load_project_config()
    hpc_config = project_config.get("hpc", {})
    
    job_packs_dir_str = hpc_config.get("job_packs", "Experiments/05-scaling/job-packs")
    job_packs_dir = repo_root / job_packs_dir_str
    
    if config_path_str:
        config_path = repo_root / config_path_str
    else:
        config_path = job_packs_dir / "packs.yaml"

    while True:
        action = questionary.select(
            "HPC & Scheduling:",
            choices=[
                "Generate Job Pack",
                "Submit Job Pack",
                questionary.Separator(),
                "[b] Back",
                "[q] Quit"
            ],
            style=get_custom_style(),
            use_shortcuts=True
        ).ask()
        
        if action == "Generate Job Pack":
            _interactive_generate(config_path, job_packs_dir)
        elif action == "Submit Job Pack":
            _interactive_submit(job_packs_dir)
        elif action == "[q] Quit":
            sys.exit(0)
        else: # [b] Back or None
            break
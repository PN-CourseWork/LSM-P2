import sys
from src.utils.hpc import _interactive_generate, _interactive_submit
from src.utils.manage import load_project_config
from src.utils.TUI.io import getch, clear_screen
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
        clear_screen()
        print("\n--- HPC & Scheduling ---")
        print("  [g] Generate Job Pack")
        print("  [s] Submit Job Pack")
        print("  -----------------------")
        print("  [b] Back")
        print("  [q] Quit")
        print("\nSelect an action: ", end="", flush=True)
        
        key = getch().lower()
        
        if key == 'g':
            _interactive_generate(config_path, job_packs_dir)
        elif key == 's':
            _interactive_submit(job_packs_dir)
        elif key == 'q':
            clear_screen()
            sys.exit(0)
        elif key == 'b':
            break
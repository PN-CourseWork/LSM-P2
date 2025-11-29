"Utilities for LSF job pack generation."

import yaml
import itertools
import sys
import shutil
import subprocess
import os
import questionary
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Union

def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def generate_pack_lines(config: Dict[str, Any], job_name_base: str, selected_groups: List[str] = None) -> List[str]:
    """Generate list of job pack lines (options + command)."""
    
    # --- Schema Normalization ---
    input_jobs_raw = []

    # 1. Collect raw job definitions from various input schemas
    if "jobs" in config:
        # Preferred Schema (List of Jobs)
        input_jobs_raw = config["jobs"]
    elif "groups" in config:
        # Legacy Schema (List of Groups)
        input_jobs_raw = []
        for g in config["groups"]:
            job_entry = {
                "name": g.get("name", "job"),
                "lsf": g.get("lsf", {}),
                "sweep": g.get("sweep", {})
            }
            if "parameters" in g:
                job_entry["execution"] = {"arguments": g["parameters"]}
            if "mpi_options" in g:
                job_entry.setdefault("lsf", {})["mpi_options"] = g["mpi_options"]
            input_jobs_raw.append(job_entry)
    else:
        # New Dictionary Schema (Keys are Group Names)
        for key, value in config.items():
            if key == "defaults": # Ignore defaults as requested
                continue
            if isinstance(value, dict):
                job_entry = value.copy()
                job_entry["name"] = key 
                input_jobs_raw.append(job_entry)

    # 2. Convert all raw input_jobs entries into the normalized 'execution' format
    normalized_jobs = []
    for input_job in input_jobs_raw:
        normalized_job = {
            "name": input_job.get("name", "unnamed"),
            "lsf": input_job.get("lsf", {}),
            "sweep": input_job.get("sweep", {})
        }

        if "static_args" in input_job:
            normalized_job["execution"] = {
                "script": input_job["static_args"].get("script"),
                "interpreter": input_job["static_args"].get("interpreter", "uv run python"),
                "arguments": {k: v for k, v in input_job["static_args"].items() if k not in ["script", "interpreter"]}
            }
        elif "execution" in input_job:
            normalized_job["execution"] = input_job["execution"]
        else:
            normalized_job["execution"] = {} 
        
        normalized_jobs.append(normalized_job)
    
    # --- End Schema Normalization ---

    lines = []
    global_job_counter = 1
    
    for job in normalized_jobs:
        job_name_group = job.get("name", "job")
        
        # Filter based on selection
        if selected_groups and job_name_group not in selected_groups:
            continue
            
        # Add Group Comment Header
        lines.append(f"\n# --- Group: {job_name_group} ---")

        # 1. LSF Settings
        job_lsf = job.get("lsf", {})
        
        # 2. Execution Settings
        job_exec = job.get("execution", {})
        
        script = job_exec.get("script", "compute_scaling.py") 
        interpreter = job_exec.get("interpreter", "uv run python")
        
        # Base arguments
        current_args_base = job_exec.get("arguments", {})
        
        # Sweep
        sweep = job.get("sweep", {})
        
        if not sweep:
            combinations = [()]
            keys = []
        else:
            keys = list(sweep.keys())
            values = list(sweep.values())
            combinations = list(itertools.product(*values))
        
        for combo in combinations:
            # Create full params for this run
            run_args = current_args_base.copy()
            sweep_args = dict(zip(keys, combo))
            
            # Cartesian Product Logic 
            run_args.update(sweep_args)
            
            # Extract Special Args for Infrastructure
            ranks = run_args.pop("ranks", 1)
            
            # Construct Job Name
            current_job_name = f"{job_name_base}_{job_name_group}_{global_job_counter}"
            
            # Build Script Arguments
            args_list = []
            for k, v in run_args.items():
                if isinstance(v, bool):
                    if v:
                        args_list.append(f"--{k}")
                else:
                    args_list.append(f"--{k} {v}")
            
            # Add Standard Logging Args
            args_list.append(f"--job-name {current_job_name}")
            args_list.append("--log-dir logs")
            args_list.append(f"--experiment-name {job_name_group}")
            
            # Build Command
            mpi_options = job_lsf.get("mpi_options", "")
            
            cmd_parts = [f"mpiexec -n {ranks}"]
            if mpi_options:
                cmd_parts.append(mpi_options)
            
            cmd_parts.append(f"{interpreter} {script}")
            cmd_parts.append(" ".join(args_list))
            
            main_cmd = " ".join(cmd_parts)
            
            # Log Uploader 
            uploader_cmd = f"{interpreter} src/utils/upload_logs.py --job-name {current_job_name} --log-dir logs --experiment-name {job_name_group}"
            cmd = f'({main_cmd}; {uploader_cmd})'
            
            # Build LSF Line
            lsf_opts = []
            lsf_opts.append(f"-J {current_job_name}")
            
            # Cores logic
            cores = ranks 
            if "cores" in job_lsf:
                cores = job_lsf["cores"]
            lsf_opts.append(f"-n {cores}")
            
            # Standard Keys
            val_map = {
                'queue': '-q',
                'walltime': '-W',
                'memory': '-M',
                'email': '-u',
                'project': '-P',
            }
            for key, flag in val_map.items():
                if key in job_lsf:
                    lsf_opts.append(f"{flag} {job_lsf[key]}")
            
            # Boolean mappings
            bool_map = {
                'exclusive': '-x',
                'notify_start': '-B',
                'notify_end': '-N',
            }
            for key, flag in bool_map.items():
                if job_lsf.get(key, False):
                    lsf_opts.append(flag)
            
            # Resources
            resources = job_lsf.get("resources", [])
            if isinstance(resources, str): resources = [resources]
            for r in resources:
                lsf_opts.append(f'-R "{r}"')
            
            # Logs
            lsf_opts.append(f"-o logs/{current_job_name}.out")
            lsf_opts.append(f"-e logs/{current_job_name}.err")
            
            line = " ".join(lsf_opts) + " " + cmd
            lines.append(line)
            global_job_counter += 1
            
    return lines

def write_pack_file(output_path: Path, lines: List[str]):
    """Write LSF pack file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# LSF Job Pack generated by src.utils.hpc\n")
        for line in lines:
            f.write(line + "\n")

def _get_custom_style():
    return questionary.Style([
        ('qmark', 'fg:#673ab7 bold'),       # Token.QuestionMark
        ('question', 'bold'),               # Token.Question
        ('answer', 'fg:#f44336 bold'),      # Token.Answer
        ('pointer', 'fg:#673ab7 bold'),     # Token.Pointer
        ('highlighted', 'fg:#673ab7 bold'), # Token.Highlighted
        ('selected', 'fg:#cc5454'),         # Token.Selected
        ('separator', 'fg:#cc5454'),        # Token.Separator
        ('instruction', '')                 # Token.Instruction
    ])

def _interactive_generate(config_path: Path, job_packs_dir: Path):
    """Handle job pack generation workflow."""
    print(f"\n--- Generate Job Pack ---")
    print(f"Loading config: {config_path}")
    
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        return

    # Extract Group Names based on Schema
    group_names = []
    if "jobs" in config:
        group_names = [j.get("name", "unnamed") for j in config["jobs"]]
    elif "groups" in config:
        group_names = [j.get("name", "unnamed") for j in config["groups"]]
    else:
        # Dictionary Schema
        group_names = [key for key in config.keys() if key != "defaults"]
        
    if not group_names:
        print("No job groups found in configuration.")
        return
    
    # Select Groups
    selected_groups_for_gen = questionary.checkbox( # Renamed to avoid conflict
        "Select experiment groups to generate packs for (Space to select, Enter to confirm):",
        choices=group_names,
        style=_get_custom_style()
    ).ask()
    
    if not selected_groups_for_gen:
        print("No groups selected. returning to menu.")
        return

    generated_files = []
    total_jobs_generated = 0

    for group_name in selected_groups_for_gen:
        # Generate lines ONLY for the current group
        job_name_base = config_path.stem # Base name from config file
        lines = generate_pack_lines(config, job_name_base, [group_name]) # Pass single group
        
        if not lines:
            print(f"  No jobs generated for group '{group_name}'. Skipping.")
            continue

        print(f"\n  Generated {len(lines)} jobs for group '{group_name}'.")
        total_jobs_generated += len(lines)
        
        # Output file named after the group
        output_file = job_packs_dir / f"{group_name}.pack"
        write_pack_file(output_file, lines)
        generated_files.append(output_file)
        print(f"  Pack file saved to: {output_file}")
        
        # Preview for this group
        print(f"\n  --- Pack Content Preview for {group_name}.pack ---")
        print("  " + "-" * 40)
        for line in lines[:5]: # Show fewer lines for individual previews
            print("  " + line)
        if len(lines) > 5:
            print(f"  ... ({len(lines) - 5} more lines)")
        print("  " + "-" * 40)
    
    if generated_files:
        print(f"\n--- Generation Summary ---")
        print(f"Successfully generated {total_jobs_generated} total jobs across {len(generated_files)} pack files:")
        for f in generated_files:
            print(f"  ✓ {f.name}")
    else:
        print("\nNo job packs were generated.")

    print("\n")
    input("Press Enter to continue...")

def _interactive_submit(job_packs_dir: Path):
    """Handle job pack submission workflow."""
    print(f"\n--- Submit Job Pack ---")
    
    if not job_packs_dir.exists():
        print(f"Directory not found: {job_packs_dir}")
        return

    # List pack files, sorted by modification time (newest first)
    pack_files = sorted(job_packs_dir.glob("*.pack"), key=lambda f: f.stat().st_mtime, reverse=True)
    
    if not pack_files:
        print("No .pack files found.")
        return

    choices = [f.name for f in pack_files]
    choices.append("Cancel")
    
    selection = questionary.select(
        "Select a job pack to submit:",
        choices=choices,
        style=_get_custom_style()
    ).ask()
    
    if selection == "Cancel" or not selection:
        return
    
    selected_file = job_packs_dir / selection
    
    # Preview
    print(f"\nSelected: {selected_file}")
    print("-" * 40)
    with open(selected_file, 'r') as f:
        head = [next(f) for _ in range(10)]
    for line in head:
        print(line.strip())
    print("...")
    print("-" * 40)

    # Confirm
    if not shutil.which("bsub"):
        print("Error: 'bsub' command not found. Cannot submit.")
        input("Press Enter to continue...")
        return

    should_submit = questionary.confirm(
        f"Submit {selection} to LSF?",
        default=False,
        style=_get_custom_style()
    ).ask()
    
    if should_submit:
        print(f"Submitting {selection}...")
        try:
            subprocess.run(["bsub", "-pack", str(selected_file)], check=True)
            print("Successfully submitted.")
        except subprocess.CalledProcessError as e:
            print(f"Submission failed: {e}")
    else:
        print("Submission cancelled.")
    
    input("Press Enter to continue...")

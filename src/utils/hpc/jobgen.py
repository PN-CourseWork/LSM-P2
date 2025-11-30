"""Job pack generation utilities for HPC schedulers.

Generates LSF-compatible job pack files based on a declarative YAML configuration
with parameter sweeps.
"""

import itertools
import os
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional

# --- Helper Functions ---

def get_project_root() -> Path:
    """Returns the project root folder (LSM-P2)."""
    return Path(__file__).parents[3] # src/utils/hpc is 3 levels deep from root

def get_job_output_dir(job_name_base: str, create: bool = True) -> Path:
    """Get the directory for job output files (e.g., .out, .err).

    Uses $HPC_OUTPUT_DIR if set, otherwise defaults to project_root/logs/hpc-jobs.
    Creates the directory if it doesn't exist.

    Parameters
    ----------
    job_name_base : str
        Base name for the job group, used to create a subdirectory.
    create : bool
        If True, create the directory if it doesn't exist.

    Returns
    -------
    Path
        Directory path for job outputs
    """
    if "HPC_OUTPUT_DIR" in os.environ:
        base_dir = Path(os.environ["HPC_OUTPUT_DIR"])
    else:
        base_dir = get_project_root() / "logs" / "hpc-jobs"

    output_dir = base_dir / job_name_base
    if create:
        output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file.

    Parameters
    ----------
    config_path : Path
2        Path to YAML config file

    Returns
    -------
    dict
        Parsed configuration

    Raises
    ------
    FileNotFoundError
        If config file doesn't exist
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_pack_lines(
    config: Dict[str, Any],
    job_name_prefix: str,
    selected_groups: Optional[List[str]] = None,
) -> List[str]:
    """Generate list of LSF job pack lines.

    Parameters
    ----------
    config : dict
        Parsed configuration dictionary for job groups.
    job_name_prefix : str
        Prefix for generated job names (e.g., from the pack filename).
    selected_groups : list of str, optional
        If provided, only generate jobs for these specified groups.

    Returns
    -------
    list of str
        Lines for the job pack file, each representing a single LSF job submission.
    """
    lines = []
    
    # Ensure a consistent output directory for this pack generation session
    session_output_dir = get_job_output_dir(job_name_prefix)

    for group_name, group_config in config.items():
        # Filter based on selection
        if selected_groups and group_name not in selected_groups:
            continue

        lines.append(f"\n# --- Group: {group_name} ---")

        lsf_options_templates: List[str] = group_config.get("lsf_options", [])
        executable_template: str = group_config.get("executable", "python")
        script_path: str = group_config.get("script", "")
        static_args: Dict[str, Any] = group_config.get("static_args", {})
        sweep: Dict[str, List[Any]] = group_config.get("sweep", {})
        
        # Prepare sweep combinations
        if not sweep:
            # If no sweep, generate a single combination for static args
            combinations = [{}]
        else:
            sweep_keys = list(sweep.keys())
            sweep_values = list(sweep.values())
            combinations = [dict(zip(sweep_keys, combo)) for combo in itertools.product(*sweep_values)]

        for i, combo_dict in enumerate(combinations):
            # Combine all arguments (static + sweep) into one dictionary for formatting
            # This dictionary will be used to format all template strings
            all_args_for_formatting = {**static_args, **combo_dict}
            
            # Dynamically generate a job name base for output files, using group name and counter
            # Ensure job_name is available for templates (e.g., in lsf_options or command)
            job_name_suffix = "_".join([f"{k}{v}" for k,v in combo_dict.items()]) if combo_dict else "base"
            current_job_name = f"{job_name_prefix}_{group_name}_{job_name_suffix}_{i:03d}"
            all_args_for_formatting["job_name"] = current_job_name # Make it available for formatting
            all_args_for_formatting["LSF_OUTPUT_DIR"] = session_output_dir # Make path available

            # --- Format LSF Options ---
            formatted_lsf_options = []
            for opt_template in lsf_options_templates:
                try:
                    formatted_lsf_options.append(opt_template.format(**all_args_for_formatting))
                except KeyError as e:
                    raise ValueError(f"Missing key in LSF option template '{opt_template}': {e}. Available keys: {list(all_args_for_formatting.keys())}")

            # --- Construct Script Arguments String ---
            script_args_list = []
            # Combine static and sweep args for the script itself
            combined_script_args = {**static_args, **combo_dict}
            for k, v in combined_script_args.items():
                if isinstance(v, bool):
                    if v: # Only add flag if True
                        script_args_list.append(f"--{k}")
                elif v is not None: # Only add if not None
                    script_args_list.append(f"--{k} {v}")
            
            # Also add job-name, log-dir, experiment-name as standard arguments
            script_args_list.append(f"--job-name {current_job_name}")
            script_args_list.append(f"--log-dir logs") # Assumes 'logs' is relative to project root
            script_args_list.append(f"--experiment-name {group_name}")

            formatted_script_args = " ".join(script_args_list)

            # --- Assemble Full Command ---
            # The executable and script path might also contain placeholders
            formatted_executable = executable_template.format(**all_args_for_formatting)
            full_command = f"{formatted_executable} {script_path} {formatted_script_args}"

            # --- Assemble Final LSF Line ---
            final_lsf_line = " ".join(formatted_lsf_options) + " " + full_command
            lines.append(final_lsf_line)
            
    return lines


def write_pack_file(output_path: Path, lines: List[str]) -> None:
    """Write LSF pack file.

    Parameters
    ----------
    output_path : Path
        Path to output file
    lines : list of str
        Lines to write
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"# LSF Job Pack generated by {Path(__file__).name}\n")
        f.write("# Each line is a bsub command (without 'bsub')\n")
        for line in lines:
            f.write(line + "\n")
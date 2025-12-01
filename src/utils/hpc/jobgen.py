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

def get_job_output_dir(absolute: bool = True) -> Path:
    """Get the directory for job output files (e.g., .out, .err).

    Uses $HPC_OUTPUT_DIR if set, otherwise defaults to logs/lsf.

    Parameters
    ----------
    absolute : bool
        If True, return absolute path. If False, return relative path
        (useful for HPC job scripts where cwd is project root).

    Returns
    -------
    Path
        Directory path for job outputs
    """
    if "HPC_OUTPUT_DIR" in os.environ:
        return Path(os.environ["HPC_OUTPUT_DIR"])

    if absolute:
        return get_project_root() / "logs" / "lsf"
    return Path("logs/lsf")


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
    # Use relative path for HPC jobs (assumes cwd is project root)
    output_dir = get_job_output_dir(absolute=False)

    for group_name, group_config in config.items():
        # Filter based on selection
        if selected_groups and group_name not in selected_groups:
            continue

        lines.append(f"\n# --- Group: {group_name} ---")

        lsf_options_templates: List[str] = group_config.get("lsf_options", [])
        command_template: str = group_config.get("command", "")
        static_args: Dict[str, Any] = group_config.get("static_args", {})
        sweep: Dict[str, List[Any]] = group_config.get("sweep", {})
        sweep_paired: Dict[str, List[Any]] = group_config.get("sweep_paired", {})

        if not command_template:
            raise ValueError(f"Group '{group_name}' is missing required 'command' field")

        # Prepare sweep combinations
        # 1. Paired sweep: all lists zipped together (must be same length)
        # 2. Regular sweep: cartesian product
        # Final combinations = cartesian product of (paired_combos × regular_combos)

        paired_combinations = [{}]
        if sweep_paired:
            paired_keys = list(sweep_paired.keys())
            paired_values = list(sweep_paired.values())
            # Validate all lists have same length
            lengths = [len(v) for v in paired_values]
            if len(set(lengths)) > 1:
                raise ValueError(
                    f"Group '{group_name}': sweep_paired lists must have same length, "
                    f"got {dict(zip(paired_keys, lengths))}"
                )
            paired_combinations = [dict(zip(paired_keys, combo)) for combo in zip(*paired_values)]

        regular_combinations = [{}]
        if sweep:
            sweep_keys = list(sweep.keys())
            sweep_values = list(sweep.values())
            regular_combinations = [dict(zip(sweep_keys, combo)) for combo in itertools.product(*sweep_values)]

        # Combine: cartesian product of paired × regular
        combinations = []
        for paired in paired_combinations:
            for regular in regular_combinations:
                combinations.append({**paired, **regular})

        for i, combo_dict in enumerate(combinations):
            # Combine all arguments (static + sweep) into one dictionary for formatting
            all_args = {**static_args, **combo_dict}

            # Generate job name: group_val1_val2_..._idx
            sweep_values = [str(v) for v in combo_dict.values()]
            job_suffix = "_".join(sweep_values) if sweep_values else "base"
            current_job_name = f"{group_name}_{job_suffix}_{i:03d}"

            # Add special variables available for formatting
            all_args["job_name"] = current_job_name
            all_args["experiment_name"] = group_name
            all_args["LSF_OUTPUT_DIR"] = output_dir

            # --- Format LSF Options ---
            formatted_lsf_options = []
            for opt_template in lsf_options_templates:
                try:
                    formatted_lsf_options.append(opt_template.format(**all_args))
                except KeyError as e:
                    raise ValueError(
                        f"Missing key in LSF option template '{opt_template}': {e}. "
                        f"Available keys: {list(all_args.keys())}"
                    )

            # --- Format Command ---
            try:
                formatted_command = command_template.format(**all_args)
            except KeyError as e:
                raise ValueError(
                    f"Missing key in command template: {e}. "
                    f"Available keys: {list(all_args.keys())}"
                )

            # --- Assemble Final LSF Line ---
            final_lsf_line = " ".join(formatted_lsf_options) + " " + formatted_command
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
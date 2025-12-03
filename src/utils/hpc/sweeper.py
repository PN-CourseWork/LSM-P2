"""
HPC Sweeper: Generates LSF Job Arrays from YAML configuration.

Key Features:
1. Smart Grouping: Automatically groups jobs into Arrays based on identical resource requirements.
2. Runtime Lookup: Uses 'lookup.py' to resolve arguments at runtime, keeping the YAML as the source of truth.
3. Pack Generation: Creates a master submission script that submits the Universal Runner Template.
"""

import itertools
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .jobgen import load_config

# Use the universal template path relative to project root
RUNNER_TEMPLATE = Path("src/utils/hpc/runner_template.sh")


def get_combinations(group_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate all flattened parameter combinations for a group."""
    static_args = group_config.get("static_args", {})
    sweep = group_config.get("sweep", {})
    sweep_paired = group_config.get("sweep_paired", {})

    paired_combinations = [{}]
    if sweep_paired:
        keys = list(sweep_paired.keys())
        values = list(sweep_paired.values())
        paired_combinations = [dict(zip(keys, v)) for v in zip(*values)]

    regular_combinations = [{}]
    if sweep:
        keys = list(sweep.keys())
        values = list(sweep.values())
        regular_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    all_combos = []
    for paired in paired_combinations:
        for regular in regular_combinations:
            all_combos.append({**static_args, **paired, **regular})

    return all_combos


def extract_resources(
    combo: Dict[str, Any], resource_template: Dict[str, Any]
) -> Tuple[frozenset, Dict[str, Any]]:
    """
    Determine the resource signature for a specific combination.
    """
    resources = resource_template.copy()
    # Update resources from combo if keys match
    for key, value in combo.items():
        if key in resources:
            resources[key] = value

    signature = frozenset(resources.items())
    return signature, resources


def generate_arrays(config_path: Path, output_dir: Path = None):
    """Parse config and generate array scripts."""
    config = load_config(config_path)

    # Determine Output Directories from Config
    base_dir = config_path.parent

    job_script_dir_str = config.get("job_script_output_dir")
    pack_dir_str = config.get("packs_output_dir")

    if job_script_dir_str:
        job_script_dir = Path(job_script_dir_str)
        # Ensure path is handled relative to CWD if not absolute
    else:
        job_script_dir = base_dir / "generated_jobs"

    if pack_dir_str:
        pack_dir = Path(pack_dir_str)
    else:
        pack_dir = base_dir / "generated_packs"

    if not job_script_dir.exists():
        job_script_dir.mkdir(parents=True, exist_ok=True)
    if not pack_dir.exists():
        pack_dir.mkdir(parents=True, exist_ok=True)

    print(f"Configuration: {config_path}")
    print(f"Index Maps Dir: {job_script_dir}")
    print(f"Submit Script Dir: {pack_dir}")

    submission_lines = []
    submission_lines.append("#!/bin/sh")
    submission_lines.append(f"# Generated from {config_path}")
    submission_lines.append("# Submits Universal Runner Template")
    submission_lines.append("")

    for group_name, group_config in config.items():
        if not isinstance(group_config, dict):
            continue

        if (
            "sweep" not in group_config
            and "static_args" not in group_config
            and "sweep_paired" not in group_config
        ):
            continue

        print(f"Processing group: {group_name}")

        combos = get_combinations(group_config)
        if not combos:
            continue

        res_template = group_config.get("resources", {})

        # Group by Resources
        groups = {}

        for i, combo in enumerate(combos):
            idx = i + 1  # 1-based LSF index
            sig, res = extract_resources(combo, res_template)

            if sig not in groups:
                groups[sig] = {"resources": res, "indices": []}
            groups[sig]["indices"].append(idx)

        base_cmd = group_config.get("command_prefix", "python main.py")

        for i, (sig, data) in enumerate(groups.items()):
            indices = data["indices"]
            resources = data["resources"]

            suffix = f"_sub{i}" if len(groups) > 1 else ""
            job_name = f"{group_name}{suffix}"
            num_jobs = len(indices)

            # 1. Generate Mapping File (.idx)
            map_file = job_script_dir / f"{job_name}.idx"
            with open(map_file, "w") as f:
                for global_idx in indices:
                    f.write(f"{global_idx}\n")

            # 2. Generate Submission Command
            # Construct BSUB CLI args
            # Resources
            bsub_args = []
            bsub_args.append(f'-J "{job_name}[1-{num_jobs}]"')
            bsub_args.append(f"-q {resources.get('queue', 'hpcintro')}")
            bsub_args.append(f"-W {resources.get('walltime', '00:10')}")
            bsub_args.append(f"-n {resources.get('n_cores', 1)}")
            bsub_args.append(f'-R "rusage[mem={resources.get("mem", "4GB")}]"')
            bsub_args.append('-R "span[ptile=24]"')  # Default, could be configurable

            # Output logs
            bsub_args.append("-o logs/lsf/%J_%I.out")
            bsub_args.append("-e logs/lsf/%J_%I.err")

            # Environment Variables
            # Use -env "VAR=VAL,VAR2=VAL2"
            env_vars = []
            env_vars.append(f"SWEEP_CONFIG={config_path}")
            env_vars.append(f"SWEEP_GROUP={group_name}")
            env_vars.append(f"SWEEP_MAP_FILE={map_file}")
            env_vars.append(f"SWEEP_CMD='{base_cmd}'")

            # Quote the env vars string properly
            env_string = ",".join(env_vars)
            bsub_args.append(f'-env "{env_string}"')

            # Command to submit the template
            # Note: We assume RUNNER_TEMPLATE is executable or we run it via sh?
            # LSF runs the script.
            # bsub < script is standard.

            cmd = f"bsub {' '.join(bsub_args)} < {RUNNER_TEMPLATE}"
            submission_lines.append(cmd)

            print(f"  ✓ Queued Array {job_name} (Size: {num_jobs})")

    # Write Master Submit Script
    if len(submission_lines) > 4:  # Header is 4 lines
        submit_file_name = config_path.stem + "_submit.sh"
        submit_file_path = pack_dir / submit_file_name

        with open(submit_file_path, "w") as f:
            for line in submission_lines:
                f.write(line + "\n")

        print(f"\nSubmission Script generated: {submit_file_path}")
        print(f"Run: sh {submit_file_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    generate_arrays(args.config, args.out)

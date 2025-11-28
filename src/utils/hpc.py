"""Utilities for LSF job pack generation."""

import yaml
import itertools
from pathlib import Path
from typing import List, Dict, Any


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_pack_lines(config: Dict[str, Any], job_name_base: str) -> List[str]:
    """Generate list of job pack lines (options + command)."""
    script = config.get("script", "compute_scaling.py")
    base_cmd = f"mpiexec -n {{ranks}} uv run python {script}"
    
    # Global defaults
    global_params = config.get("parameters", {})
    global_lsf = config.get("lsf", {})
    global_sweep = config.get("sweep", {})
    global_mpi = config.get("mpi_options", "")
    
    # Normalize to a list of groups
    if "groups" in config:
        groups = config["groups"]
    else:
        # Legacy/Simple mode: Treat top-level as a single group
        groups = [{
            "name": "default",
            "lsf": {},
            "parameters": {},
            "sweep": global_sweep,
            "mpi_options": ""
        }]

    lines = []
    global_job_counter = 1
    
    for group in groups:
        # Merge configurations (Group overrides Global)
        group_lsf = {**global_lsf, **group.get("lsf", {})}
        group_params = {**global_params, **group.get("parameters", {})}
        group_sweep = group.get("sweep", {})
        group_mpi = group.get("mpi_options", global_mpi)
        
        # If no sweep in group, use global sweep (if it exists and wasn't just the legacy fallback)
        if not group_sweep and "groups" in config:
             group_sweep = global_sweep

        # Prepare sweep combinations
        keys = list(group_sweep.keys())
        values = list(group_sweep.values())
        combinations = list(itertools.product(*values))
        
        group_name = group.get("name", "job")
        
        for combo in combinations:
            current_params = group_params.copy()
            current_sweep = dict(zip(keys, combo))
            
            # Handle scaling logic
            scaling_type = config.get("type", "strong")
            if scaling_type == "weak":
                # For weak scaling, N depends on ranks
                base_N = current_params.get("base_N", 32)
                ranks = current_sweep.get("ranks", 1)
                N = int(round(base_N * (ranks**(1/3))))
                current_params["N"] = N
            
            # Merge sweep parameters
            current_params.update(current_sweep)
            
            # Extract ranks for mpiexec and bsub -n
            ranks = current_params.pop("ranks")
            
            # Calculate job name early to pass to script
            current_job_name = f"{job_name_base}_{group_name}_{global_job_counter}"

            # Build command arguments
            args = []
            for k, v in current_params.items():
                if isinstance(v, bool):
                    if v:
                        args.append(f"--{k}")
                else:
                    args.append(f"--{k} {v}")
            
            # Add logging info for MLflow artifact upload
            args.append(f"--job-name {current_job_name}")
            args.append("--log-dir logs")

            # Construct the actual command to run
            # base_cmd pattern: mpiexec -n {ranks} {mpi_options} uv run python {script} {args}
            cmd_parts = [f"mpiexec -n {ranks}"]
            if group_mpi:
                cmd_parts.append(group_mpi)
            cmd_parts.append(f"uv run python {script}")
            cmd_parts.append(" ".join(args))
            
            main_cmd = " ".join(cmd_parts)
            
            # Chain the log uploader command
            # We use (cmd; uploader) to ensure uploader runs regardless of cmd exit status
            uploader_cmd = f"uv run python src/utils/upload_logs.py --job-name {current_job_name} --log-dir logs"
            cmd = f'({main_cmd}; {uploader_cmd})'
            
            # Build LSF options for this specific job
            # current_job_name is already defined above
            
            # Map config keys to LSF flags
            lsf_opts = []
            lsf_opts.append(f"-J {current_job_name}")
            
            # Cores: driven by ranks (if in sweep/params) or explicit 'cores' config
            cores = ranks if ranks is not None else group_lsf.get('cores', 1)
            lsf_opts.append(f"-n {cores}")
            
            # Value mappings
            val_map = {
                'queue': '-q',
                'walltime': '-W',
                'memory': '-M',
                'email': '-u',
                'project': '-P',
            }
            for key, flag in val_map.items():
                if key in group_lsf:
                    lsf_opts.append(f"{flag} {group_lsf[key]}")
            
            # Boolean mappings
            bool_map = {
                'exclusive': '-x',
                'notify_start': '-B',
                'notify_end': '-N',
            }
            for key, flag in bool_map.items():
                if group_lsf.get(key, False):
                    lsf_opts.append(flag)
            
            # Resource requirements (-R)
            resources = group_lsf.get('resources', [])
            if isinstance(resources, str):
                resources = [resources]
            for r in resources:
                lsf_opts.append(f'-R "{r}"')
            
            # Logs
            lsf_opts.append(f"-o logs/{current_job_name}.out")
            lsf_opts.append(f"-e logs/{current_job_name}.err")
            
            # Combine options and command
            line = " ".join(lsf_opts) + " " + cmd
            lines.append(line)
            global_job_counter += 1
        
    return lines


def write_pack_file(output_path: Path, lines: List[str]):
    """Write LSF pack file."""
    with open(output_path, "w") as f:
        f.write("# LSF Job Pack generated by generate_pack.py\n")
        for line in lines:
            f.write(line + "\n")
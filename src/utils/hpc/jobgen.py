"""Job pack generation utilities for HPC schedulers.

Supports multiple configuration schemas and generates
LSF-compatible job pack files with parameter sweeps.
"""

import itertools
import yaml
from pathlib import Path
from typing import List, Dict, Any


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file.

    Parameters
    ----------
    config_path : Path
        Path to YAML config file

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


def _normalize_jobs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize various config schemas to a standard format.

    Supports three schemas:
    1. Preferred: {"jobs": [{"name": ..., "lsf": ..., "sweep": ...}, ...]}
    2. Legacy: {"groups": [{"name": ..., "parameters": ..., ...}, ...]}
    3. Dictionary: {"job_name": {"lsf": ..., "sweep": ...}, ...}
    """
    input_jobs_raw = []

    if "jobs" in config:
        input_jobs_raw = config["jobs"]
    elif "groups" in config:
        for g in config["groups"]:
            job_entry = {
                "name": g.get("name", "job"),
                "lsf": g.get("lsf", {}),
                "sweep": g.get("sweep", {}),
            }
            if "parameters" in g:
                job_entry["execution"] = {"arguments": g["parameters"]}
            if "mpi_options" in g:
                job_entry.setdefault("lsf", {})["mpi_options"] = g["mpi_options"]
            input_jobs_raw.append(job_entry)
    else:
        for key, value in config.items():
            if key == "defaults":
                continue
            if isinstance(value, dict):
                job_entry = value.copy()
                job_entry["name"] = key
                input_jobs_raw.append(job_entry)

    # Convert to normalized 'execution' format
    normalized_jobs = []
    for input_job in input_jobs_raw:
        normalized_job = {
            "name": input_job.get("name", "unnamed"),
            "lsf": input_job.get("lsf", {}),
            "sweep": input_job.get("sweep", {}),
        }

        if "static_args" in input_job:
            normalized_job["execution"] = {
                "script": input_job["static_args"].get("script"),
                "interpreter": input_job["static_args"].get(
                    "interpreter", "uv run python"
                ),
                "arguments": {
                    k: v
                    for k, v in input_job["static_args"].items()
                    if k not in ["script", "interpreter"]
                },
            }
        elif "execution" in input_job:
            normalized_job["execution"] = input_job["execution"]
        else:
            normalized_job["execution"] = {}

        normalized_jobs.append(normalized_job)

    return normalized_jobs


def generate_pack_lines(
    config: Dict[str, Any],
    job_name_base: str,
    selected_groups: List[str] = None,
) -> List[str]:
    """Generate list of job pack lines (LSF options + command).

    Parameters
    ----------
    config : dict
        Parsed configuration dictionary
    job_name_base : str
        Base name for generated jobs
    selected_groups : list of str, optional
        Only generate jobs for these groups

    Returns
    -------
    list of str
        Lines for the job pack file
    """
    normalized_jobs = _normalize_jobs(config)

    lines = []
    global_job_counter = 1

    for job in normalized_jobs:
        job_name_group = job.get("name", "job")

        # Filter based on selection
        if selected_groups and job_name_group not in selected_groups:
            continue

        # Add Group Comment Header
        lines.append(f"\n# --- Group: {job_name_group} ---")

        # LSF Settings
        job_lsf = job.get("lsf", {})

        # Execution Settings
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
            run_args = current_args_base.copy()
            sweep_args = dict(zip(keys, combo))
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
            uploader_cmd = (
                f"{interpreter} src/utils/upload_logs.py "
                f"--job-name {current_job_name} --log-dir logs "
                f"--experiment-name {job_name_group}"
            )
            cmd = f"({main_cmd}; {uploader_cmd})"

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
                "queue": "-q",
                "walltime": "-W",
                "memory": "-M",
                "email": "-u",
                "project": "-P",
            }
            for key, flag in val_map.items():
                if key in job_lsf:
                    lsf_opts.append(f"{flag} {job_lsf[key]}")

            # Boolean mappings
            bool_map = {
                "exclusive": "-x",
                "notify_start": "-B",
                "notify_end": "-N",
            }
            for key, flag in bool_map.items():
                if job_lsf.get(key, False):
                    lsf_opts.append(flag)

            # Resources
            resources = job_lsf.get("resources", [])
            if isinstance(resources, str):
                resources = [resources]
            for r in resources:
                lsf_opts.append(f'-R "{r}"')

            # Logs
            lsf_opts.append(f"-o logs/{current_job_name}.out")
            lsf_opts.append(f"-e logs/{current_job_name}.err")

            line = " ".join(lsf_opts) + " " + cmd
            lines.append(line)
            global_job_counter += 1

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
        f.write("# LSF Job Pack generated by src.utils.hpc\n")
        for line in lines:
            f.write(line + "\n")
